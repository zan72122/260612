"""シグナル合成。"""

from __future__ import annotations

import pandas as pd

from finbot.config import BotConfig
from finbot.signals.momentum import ts_momentum, xs_momentum


def combined_signal(prices: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    """時系列モメンタムとクロスセクショナルの加重平均。各資産 [-1, 1]。"""
    ts = ts_momentum(prices, cfg.momentum_lookbacks, cfg.vol_span)
    xs = xs_momentum(prices, cfg.xs_lookback)
    sig = cfg.ts_weight * ts + (1.0 - cfg.ts_weight) * xs
    return sig.clip(-1.0, 1.0)
