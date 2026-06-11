"""トレンドシグナル: Baz, Granger, Harvey, Le Roux, Rattray (2015) の EWMAC 構成。

すべて「t 行の値は t 日終値までの情報のみ」で計算される因果的シグナル。
エンジン側がウェイトを 1 日シフトして翌日リターンに適用するため、
ここでは先読みシフトを入れないこと。

構成(独立再現論文 2 件でパラメータ確認済み):
    x_k = EWMA_S(price) - EWMA_L(price)     (S,L) ∈ {(8,24),(16,48),(32,96)}
    y_k = x_k / rolling_std(price, 63日)
    z_k = y_k / rolling_std(y_k, 252日)
    u_k = z_k * exp(-z_k^2 / 4) / 0.89      応答関数: z=±√2 でピーク、極端値で減衰
    trend = (1/3) Σ_k u_k

応答関数が極端なトレンドでポジションを自然に縮めるため、
別途のシグナルキャップはほぼ飾りになる(±1 弱に有界)。
符号ルールに対する優位はアルファではなくターンオーバー削減
(Baltas-Kosowski: 連続シグナルでネットシャープ改善)。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

_RESPONSE_NORM = 0.89  # sup_z |z·exp(-z²/4)| の正規化定数(Baz et al. 2015)


def _halflife(n: int) -> float:
    """タイムスケール n を EWMA halflife に変換: HL = log(0.5)/log(1-1/n)。"""
    return math.log(0.5) / math.log(1.0 - 1.0 / n)


def ewmac_trend(
    prices: pd.DataFrame,
    pairs: Sequence[tuple[int, int]] = ((8, 24), (16, 48), (32, 96)),
    price_vol_window: int = 63,
    signal_vol_window: int = 252,
) -> pd.DataFrame:
    """Baz et al. (2015) のマルチスピード EWMAC トレンド。各資産ほぼ [-1, 1]。"""
    scores = []
    for short, long in pairs:
        x = (
            prices.ewm(halflife=_halflife(short)).mean()
            - prices.ewm(halflife=_halflife(long)).mean()
        )
        price_vol = prices.rolling(price_vol_window).std()
        y = x / price_vol.where(price_vol > 0)
        sig_vol = y.rolling(signal_vol_window).std()
        z = y / sig_vol.where(sig_vol > 0)
        scores.append(z * np.exp(-(z**2) / 4.0) / _RESPONSE_NORM)
    return sum(scores) / len(scores)
