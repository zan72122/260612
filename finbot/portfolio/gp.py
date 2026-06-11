"""Gârleanu-Pedersen 部分調整(Dynamic Trading, JF 2013)。

取引コストが存在するとき、毎日エイム(目標)へ全量リバランスするのは
最適ではない。GP の閉形式解は

    x_t = (1 - τ) · x_{t-1} + τ · aim_t

すなわち「エイムへ毎日 τ だけ近づく」。τ(取引速度)はコストが高いほど
小さく、リスク回避度とアルファ減衰が大きいほど大きくなる。
ここでは τ を設定値 gp_trade_rate として与える。

エイム側のシグナル持続性重み付けは finbot.signals.ensemble.gp_lookback_weights
が担い、本モジュールはポジションの部分調整のみを行う。
"""

from __future__ import annotations

import numpy as np


def partial_adjustment(aim: np.ndarray, trade_rate: float) -> np.ndarray:
    """エイム系列 (T×N) に対する GP 部分調整後のウェイト系列を返す。

    w_t = (1 - τ)·w_{t-1} + τ·aim_t  (w_{-1} = 0)
    凸結合なので、エイムが満たす制約(|w| 上限・グロス上限)は保存される。
    """
    if not 0.0 < trade_rate <= 1.0:
        raise ValueError("trade_rate は (0, 1] の範囲が必要")
    out = np.empty_like(aim)
    w = np.zeros(aim.shape[1])
    for t in range(aim.shape[0]):
        w = (1.0 - trade_rate) * w + trade_rate * aim[t]
        out[t] = w
    return out
