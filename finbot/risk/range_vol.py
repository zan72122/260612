"""レンジベース・ボラティリティ推定(Yang-Zhang 2000)。

終値同士のリターンだけを使う推定は高値・安値・始値の情報を捨てており、
レンジを使う推定量は同じ窓長で統計効率が数倍高い(= 推定ノイズが小さい)。
ボラターゲティングの分母が滑らかかつ速く真のボラに追随するため、
実現ボラの安定化を通じてシャープレシオを直接改善する。

Yang-Zhang はオーバーナイトギャップとドリフトの両方に頑健な唯一の
レンジ推定量で、以下の3成分の合成:

    σ²_YZ = σ²_overnight + k·σ²_open_to_close + (1-k)·σ²_RS

- σ²_overnight   : o_t = ln(O_t / C_{t-1}) の分散
- σ²_open_to_close: c_t = ln(C_t / O_t) の分散
- σ²_RS          : Rogers-Satchell 項 u(u-c) + d(d-c) の平均
                   (u = ln(H/O), d = ln(L/O))
- k = 0.34 / (1.34 + (n+1)/(n-1))

ここでは移動窓の代わりに EWMA を使う(因果的・逐次更新可能)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finbot.data.base import OHLCFrames


def yang_zhang_variance(ohlc: OHLCFrames, span: int) -> pd.DataFrame:
    """日次分散の推定系列 (T×N) を返す。t 行は t 日までの OHLC のみ使用。"""
    o = np.log(ohlc.open / ohlc.close.shift(1))  # オーバーナイト
    c = np.log(ohlc.close / ohlc.open)           # 場中(始値→終値)
    u = np.log(ohlc.high / ohlc.open)
    d = np.log(ohlc.low / ohlc.open)
    rs = u * (u - c) + d * (d - c)               # Rogers-Satchell(日次の分散推定)

    minp = max(2, span // 2)
    var_o = o.ewm(span=span, min_periods=minp).var()
    var_c = c.ewm(span=span, min_periods=minp).var()
    mean_rs = rs.ewm(span=span, min_periods=minp).mean()

    k = 0.34 / (1.34 + (span + 1) / (span - 1))
    yz = var_o + k * var_c + (1.0 - k) * mean_rs
    # RS 項は理論上非負だが数値誤差に備えて下限 0 を保証
    return yz.clip(lower=0.0)


def yang_zhang_ann_vol(
    ohlc: OHLCFrames, span: int, trading_days: int = 252
) -> pd.DataFrame:
    """年率ボラティリティの推定系列 (T×N) を返す。"""
    return np.sqrt(yang_zhang_variance(ohlc, span) * trading_days)
