"""ボラティリティ・ターゲティング。

ポートフォリオの予測ボラが目標年率ボラに一致するよう全体を
スケールする。リスク量を一定に保つことはリターンの安定化、
すなわちシャープレシオの改善に直結する。
"""

from __future__ import annotations

import numpy as np


def vol_target_leverage(
    weights: np.ndarray,
    cov_daily: np.ndarray,
    target_vol: float,
    max_leverage: float,
    trading_days: int = 252,
) -> float:
    """単一日のウェイトと日次共分散から全体スケール係数を返す。"""
    if np.isnan(cov_daily).any() or not np.isfinite(weights).all():
        return 0.0
    gross = np.abs(weights).sum()
    if gross < 1e-12:
        return 0.0
    var_daily = float(weights @ cov_daily @ weights)
    if var_daily <= 0:
        return 0.0
    forecast_vol = np.sqrt(var_daily * trading_days)
    lev = target_vol / forecast_vol
    # グロスレバレッジ上限でクリップ
    return float(min(lev, max_leverage / gross))
