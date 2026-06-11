"""モメンタムシグナル。

すべて「t 行の値は t 日終値までの情報のみ」で計算される因果的シグナル。
エンジン側がウェイトを 1 日シフトして翌日リターンに適用するため、
ここでは先読みシフトを入れないこと。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def ts_momentum(
    prices: pd.DataFrame,
    lookbacks: Sequence[int] = (21, 63, 126, 252),
    vol_span: int = 60,
    lookback_weights: Sequence[float] | None = None,
) -> pd.DataFrame:
    """ボラ調整済み時系列モメンタム。各資産 [-1, 1]。

    複数ルックバックの「リターン/ボラ」スコアを tanh で squash して平均。
    単一ルックバックへの過学習を避けるためのアンサンブル。
    lookback_weights を与えると等加重の代わりに加重平均する
    (Gârleanu-Pedersen の持続性重み。合計 1 に正規化して使う)。
    """
    rets = prices.pct_change()
    daily_vol = rets.ewm(span=vol_span, min_periods=vol_span // 2).std()
    if lookback_weights is None:
        lookback_weights = [1.0] * len(lookbacks)
    if len(lookback_weights) != len(lookbacks):
        raise ValueError("lookback_weights は lookbacks と同じ長さが必要")
    total = float(sum(lookback_weights))
    scores = []
    for lb, w in zip(lookbacks, lookback_weights):
        ret_lb = prices / prices.shift(lb) - 1.0
        # lb 日間の期待標準偏差で正規化した t 統計量風スコア
        norm = daily_vol * np.sqrt(lb)
        scores.append((w / total) * np.tanh(ret_lb / norm.where(norm > 0)))
    return sum(scores)


def xs_momentum(prices: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """クロスセクショナル・モメンタム。同日の資産間で順位付けし [-1, 1] に写像。

    相対的な強さを取るため市場全体の方向に依存しにくく、
    時系列モメンタムと相関の低い超過収益源になる。
    """
    ret_lb = prices / prices.shift(lookback) - 1.0
    ranks = ret_lb.rank(axis=1)
    n = ret_lb.notna().sum(axis=1)
    # 順位を [-1, 1] に線形変換(最下位 -1、最上位 +1)
    sig = (2.0 * (ranks.sub(1, axis=0)).div(n - 1, axis=0)) - 1.0
    return sig.where(n.ge(2), 0.0)
