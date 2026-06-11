"""日次バックテストエンジン(設計方針 v2)。

先読みバイアス排除の原則:
- t 日の目標ウェイトは t 日終値までの情報のみで計算(allocator の責務)
- t 日終値で建てたポジションが収益を生むのは t+1 日のリターン
  (このシフトは下のループ構造で実現される)

執行モデル:
- ポジションバッファ(Carver の buffering): 目標の周囲に半幅
  buffer_frac × 平均|目標| の帯を置き、帯内なら取引しない。
  帯外なら「帯の端まで」戻す(中心まで戻さない)。比例コスト下の
  最適方策はバンド端への最小調整であり、バンド幅はコストの3乗根で
  しかスケールしないため固定比率で頑健。
- 取引コスト: ターンオーバー(|Δw| の合計)× cost_bps

ドローダウン・ブレーキは v2 で撤廃した: トレンドシグナルの符号反転と
ボラターゲットが既にデレバレッジを内包しており、追加ブレーキは
OOS シャープに中立〜マイナスでクライシスアルファを毀損する
(docs/DESIGN_V2.md §6)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finbot.backtest.metrics import summary
from finbot.config import BotConfig
from finbot.portfolio.allocator import compute_target_weights


@dataclass
class BacktestResult:
    returns: pd.Series          # 日次ネットリターン
    equity: pd.Series           # エクイティカーブ(初期値 1.0)
    weights: pd.DataFrame       # 各日の保有ウェイト(終値時点)
    turnover: pd.Series         # 日次ターンオーバー
    stats: dict[str, float]


def run_backtest(
    prices: pd.DataFrame,
    cfg: BotConfig,
    target_weights: pd.DataFrame | None = None,
    carry: pd.DataFrame | None = None,
) -> BacktestResult:
    if target_weights is None:
        target_weights = compute_target_weights(prices, cfg, carry)

    rets = prices.pct_change().to_numpy(dtype=float)
    tw = target_weights.to_numpy(dtype=float)
    t_len, n = rets.shape

    cost_rate = cfg.cost_bps * 1e-4
    held = np.zeros(n)
    equity = 1.0

    port_rets = np.zeros(t_len)
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
        net_ret = gross

        # 本日終値の情報でリバランス判断(効果は翌日以降に現れる)
        target = tw[t]
        band = cfg.buffer_frac * np.abs(target).mean() if n else 0.0
        gap = target - held
        outside = np.abs(gap) > band
        if outside.any():
            # 帯外の資産だけを帯の端まで動かす(最小限の取引)
            trade = np.where(outside, gap - np.sign(gap) * band, 0.0)
            turn = float(np.abs(trade).sum())
            cost = turn * cost_rate
            equity *= 1.0 - cost
            net_ret = (1.0 + gross) * (1.0 - cost) - 1.0
            held = held + trade
            turnover[t] = turn

        port_rets[t] = net_ret
        held_hist[t] = held

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
