"""日次バックテストエンジン。

先読みバイアス排除の原則:
- t 日の目標ウェイトは t 日終値までの情報のみで計算(allocator の責務)
- t 日終値で建てたポジションが収益を生むのは t+1 日のリターン
  (このシフトは下のループ構造で実現される)

執行モデル:
- リバランスバンド: 保有と目標の最大乖離が band 以下なら取引しない
- 取引コスト: ターンオーバー(|Δw| の合計)× cost_bps
- ドローダウン・ブレーキ: 戦略エクイティの DD に応じて目標を縮小
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finbot.backtest.metrics import summary
from finbot.config import BotConfig
from finbot.portfolio.allocator import compute_target_weights
from finbot.risk.drawdown import drawdown_multiplier


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
) -> BacktestResult:
    if target_weights is None:
        target_weights = compute_target_weights(prices, cfg)

    rets = prices.pct_change().to_numpy(dtype=float)
    tw = target_weights.to_numpy(dtype=float)
    t_len, n = rets.shape

    cost_rate = cfg.cost_bps * 1e-4
    held = np.zeros(n)
    equity = 1.0
    peak = 1.0

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
        peak = max(peak, equity)
        dd = equity / peak - 1.0
        mult = drawdown_multiplier(
            dd, cfg.dd_threshold, cfg.dd_full_cut, cfg.dd_min_exposure
        )
        target = tw[t] * mult

        gap = np.abs(target - held).max() if n else 0.0
        if gap > cfg.rebalance_band:
            turn = float(np.abs(target - held).sum())
            cost = turn * cost_rate
            equity *= 1.0 - cost
            net_ret = (1.0 + gross) * (1.0 - cost) - 1.0
            held = target.copy()
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
