"""日次バックテストエンジン。

先読みバイアス排除の原則:
- t 日の目標ウェイトは t 日終値までの情報のみで計算(allocator の責務)
- t 日終値で建てたポジションが収益を生むのは t+1 日のリターン
  (このシフトは下のループ構造で実現される)

執行モデル:
- リバランスバンド: 保有と目標の最大乖離が band 以下なら取引しない
- 取引コスト: ターンオーバー(|Δw| の合計)× cost_bps
- ドローダウン・ブレーキ: 戦略エクイティの DD に応じて目標を縮小

トランチング(リバランス・タイミング運の除去):
- 同一戦略でもリバランス日の位相が違うだけで成績が有意にズレる
  (Hoffstein らの "Rebalance Timing Luck")。これは運でありスキルではない。
- n_tranches > 1 のとき、資本を等分した K 本のサブポートフォリオを並走させ、
  トランシェ k は t % K == k の日にのみリバランスを許可する。
  合成リターンは資本加重平均となり、位相という運要素が平均化で消える。
  リターン期待値を変えずに分散だけを下げる、数少ないフリーランチ。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finbot.backtest.metrics import summary
from finbot.config import BotConfig
from finbot.data.base import OHLCFrames
from finbot.portfolio.allocator import compute_target_weights
from finbot.risk.drawdown import drawdown_multiplier


@dataclass
class BacktestResult:
    returns: pd.Series          # 日次ネットリターン
    equity: pd.Series           # エクイティカーブ(初期値 1.0)
    weights: pd.DataFrame       # 各日の保有ウェイト(終値時点)
    turnover: pd.Series         # 日次ターンオーバー
    stats: dict[str, float]


def _simulate_tranche(
    rets: np.ndarray,
    tw: np.ndarray,
    cfg: BotConfig,
    offset: int = 0,
    period: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """単一トランシェのシミュレーション。

    リバランスは t % period == offset の日にのみ許可される
    (period=1, offset=0 で毎日 = トランチングなしの従来挙動)。
    戻り値: (エクイティ系列, ターンオーバー系列, 保有ウェイト履歴)。
    """
    t_len, n = rets.shape
    cost_rate = cfg.cost_bps * 1e-4
    held = np.zeros(n)
    equity = 1.0
    peak = 1.0

    eq_path = np.ones(t_len)
    turnover = np.zeros(t_len)
    held_hist = np.zeros((t_len, n))

    for t in range(1, t_len):
        r = np.nan_to_num(rets[t])

        # 前日終値時点の保有が本日のリターンを生む
        gross = float(held @ r)
        # 保有ウェイトは価格変動でドリフトする
        if abs(1.0 + gross) > 1e-12:
            held = held * (1.0 + r) / (1.0 + gross)

        equity *= 1.0 + gross

        # 本日終値の情報でリバランス判断(効果は翌日以降に現れる)
        peak = max(peak, equity)
        dd = equity / peak - 1.0
        mult = drawdown_multiplier(
            dd, cfg.dd_threshold, cfg.dd_full_cut, cfg.dd_min_exposure
        )
        target = tw[t] * mult

        gap = np.abs(target - held).max() if n else 0.0
        if t % period == offset and gap > cfg.rebalance_band:
            turn = float(np.abs(target - held).sum())
            cost = turn * cost_rate
            equity *= 1.0 - cost
            held = target.copy()
            turnover[t] = turn

        eq_path[t] = equity
        held_hist[t] = held

    return eq_path, turnover, held_hist


def run_backtest(
    prices: pd.DataFrame,
    cfg: BotConfig,
    target_weights: pd.DataFrame | None = None,
    ohlc: OHLCFrames | None = None,
) -> BacktestResult:
    if target_weights is None:
        target_weights = compute_target_weights(prices, cfg, ohlc=ohlc)

    rets = prices.pct_change().to_numpy(dtype=float)
    tw = target_weights.to_numpy(dtype=float)
    t_len, n = rets.shape

    k = max(1, cfg.n_tranches)
    eq_paths = np.empty((k, t_len))
    turns = np.empty((k, t_len))
    helds = np.empty((k, t_len, n))
    for j in range(k):
        eq_paths[j], turns[j], helds[j] = _simulate_tranche(
            rets, tw, cfg, offset=j, period=k
        )

    # 合成: 資本を等分して各トランシェに与え、その後は各自に複利で任せる。
    # 合成エクイティはトランシェ平均、保有/ターンオーバーは資本加重平均。
    eq_comb = eq_paths.mean(axis=0)
    cap_w = eq_paths / eq_paths.sum(axis=0)  # (k, T) 各日の資本シェア
    port_rets = np.zeros(t_len)
    port_rets[1:] = eq_comb[1:] / eq_comb[:-1] - 1.0
    turnover = (cap_w * turns).sum(axis=0)
    held_hist = np.einsum("kt,ktn->tn", cap_w, helds)

    idx = prices.index
    returns = pd.Series(port_rets, index=idx, name="returns")
    eq = (1.0 + returns).cumprod()
    to = pd.Series(turnover, index=idx, name="turnover")

    # ウォームアップ期間(ポジションゼロ)は統計から除外
    active = returns.iloc[cfg.warmup + 1 :]
    active_to = to.iloc[cfg.warmup + 1 :]
    stats = summary(active, cfg.rf_rate, cfg.trading_days, active_to)

    return BacktestResult(
        returns=returns,
        equity=eq,
        weights=pd.DataFrame(held_hist, index=idx, columns=prices.columns),
        turnover=to,
        stats=stats,
    )
