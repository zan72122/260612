"""ドローダウン・ブレーキ。

戦略自身のエクイティカーブのドローダウンが深くなるほど
エクスポージャを線形に縮小する。テールイベント時の損失の
複利的拡大(=ボラの右裾)を抑え、シャープの分母を守る。
"""

from __future__ import annotations


def drawdown_multiplier(
    drawdown: float,
    threshold: float,
    full_cut: float,
    min_exposure: float,
) -> float:
    """現在のドローダウン(負値, 例 -0.12)からエクスポージャ係数 [min_exposure, 1] を返す。"""
    dd = abs(min(drawdown, 0.0))
    if dd <= threshold:
        return 1.0
    if dd >= full_cut:
        return min_exposure
    frac = (dd - threshold) / (full_cut - threshold)
    return 1.0 - frac * (1.0 - min_exposure)
