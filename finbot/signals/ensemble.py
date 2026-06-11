"""シグナル合成。"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from finbot.config import BotConfig
from finbot.signals.momentum import ts_momentum, xs_momentum


def gp_lookback_weights(
    lookbacks: Sequence[int], trade_rate: float
) -> list[float]:
    """Gârleanu-Pedersen の持続性重み(エイム・ポートフォリオ用)。

    GP (2013) では、取引コスト下の最適ポートフォリオは「エイム」へ
    部分的に近づき続けるもので、エイム側では平均回帰速度 φ_i の
    シグナルを 1 / (1 + φ_i / a) 倍する(a は取引速度)。
    減衰の速いシグナルは取引が完了する前に消えてしまうため、
    最初から薄くしか追わないのが最適、という帰結。

    ルックバック ℓ 日のモメンタムの減衰速度を φ ≈ 1/ℓ とみなすと
        weight(ℓ) = 1 / (1 + 1/(ℓ·a)) = ℓ·a / (1 + ℓ·a)
    で、遅いシグナル(大きい ℓ)ほど 1 に近い重みを得る。
    """
    if trade_rate <= 0:
        raise ValueError("trade_rate は正の値が必要")
    return [lb * trade_rate / (1.0 + lb * trade_rate) for lb in lookbacks]


def combined_signal(prices: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    """時系列モメンタムとクロスセクショナルの加重平均。各資産 [-1, 1]。

    gp_enabled のときは時系列モメンタムの各ルックバックを GP の
    持続性重みで加重(遅いシグナルを相対的に過大評価したエイム)。
    """
    lw = (
        gp_lookback_weights(cfg.momentum_lookbacks, cfg.gp_trade_rate)
        if cfg.gp_enabled
        else None
    )
    ts = ts_momentum(prices, cfg.momentum_lookbacks, cfg.vol_span, lookback_weights=lw)
    xs = xs_momentum(prices, cfg.xs_lookback)
    sig = cfg.ts_weight * ts + (1.0 - cfg.ts_weight) * xs
    return sig.clip(-1.0, 1.0)
