"""EWMA 共分散の逐次計算。

S_t = lam * S_{t-1} + (1 - lam) * r_t r_t'
t 行の共分散は t 日までのリターンのみを使う(因果的)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_covariance(returns: pd.DataFrame, span: int = 120) -> np.ndarray:
    """日次リターン (T×N) から日次 EWMA 共分散 (T×N×N) を返す。

    初期値はバーンイン期間(span/4 日)の標本共分散。NaN 行は直前値を維持。
    """
    lam = 1.0 - 2.0 / (span + 1.0)
    r = returns.to_numpy(dtype=float)
    t_len, n = r.shape
    out = np.full((t_len, n, n), np.nan)

    burn = max(n + 2, span // 4)
    valid = ~np.isnan(r).any(axis=1)
    seed_rows = np.where(valid)[0]
    if len(seed_rows) < burn:
        return out
    seed_end = seed_rows[burn - 1]
    s = np.cov(r[seed_rows[:burn]].T, ddof=0)
    out[seed_end] = s
    for t in range(seed_end + 1, t_len):
        if valid[t]:
            rt = r[t]
            s = lam * s + (1.0 - lam) * np.outer(rt, rt)
        out[t] = s
    return out
