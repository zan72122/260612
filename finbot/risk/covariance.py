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


def rescale_diagonal(cov: np.ndarray, target_var: np.ndarray) -> np.ndarray:
    """共分散 (T×N×N) の分散(対角)を target_var (T×N) に合わせて再スケールする。

    相関構造は EWMA 推定を維持したまま、各資産のボラだけをより効率の高い
    推定量(例: Yang-Zhang)に差し替える標準的な合成:
        S' = D_target · C_ewma · D_target  ⇔  S'_ij = S_ij · s_i s_j,
        s_i = sqrt(v_i / S_ii)
    target_var が NaN または非正の要素はスケール 1(元の分散を維持)。
    """
    t_len, n, _ = cov.shape
    diag = cov[:, np.arange(n), np.arange(n)]
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.sqrt(target_var / diag)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return cov * scale[:, :, None] * scale[:, None, :]
