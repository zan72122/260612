"""リスク推定: ボラティリティと相関の分離推定。

ボラは速く(短期 EWMA + 長期アンカーのブレンド)、相関は遅く
(長 halflife EWMA + 定相関シュリンク)推定し、Σ = D·C·D で再合成する。
Barra USE4(ボラ HL42日 / 相関 HL200日)や Carver のブレンドボラと
同じ思想。t 行の推定値は t 日までのリターンのみを使う(因果的)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def blended_vol(
    returns: pd.DataFrame,
    halflife: int = 20,
    blend: float = 0.7,
    long_min_periods: int = 252,
    trading_days: int = 252,
) -> pd.DataFrame:
    """年率ボラ = blend × 短期 EWMA ボラ + (1-blend) × 長期アンカー。

    長期アンカー(expanding 平均ボラ)は静穏期の後にポジションが
    過大化するのを防ぐ(ボラの長期平均回帰)。
    """
    short = returns.ewm(halflife=halflife, min_periods=halflife).std()
    long = returns.expanding(min_periods=long_min_periods).std()
    vol = blend * short + (1.0 - blend) * long.fillna(short)
    return vol * np.sqrt(trading_days)


def ewma_correlation(
    returns: pd.DataFrame,
    halflife: int = 200,
    shrinkage: float = 0.2,
) -> np.ndarray:
    """日次 EWMA 相関 (T×N×N)。定相関ターゲットへ線形シュリンクする。

    初期値はバーンイン期間(halflife/2 日)の標本相関。NaN 行は直前値を維持。
    """
    lam = 0.5 ** (1.0 / halflife)
    r = returns.to_numpy(dtype=float)
    t_len, n = r.shape
    out = np.full((t_len, n, n), np.nan)

    burn = max(n + 2, halflife // 2)
    valid = ~np.isnan(r).any(axis=1)
    seed_rows = np.where(valid)[0]
    if len(seed_rows) < burn:
        return out

    seed_end = seed_rows[burn - 1]
    s = np.cov(r[seed_rows[:burn]].T, ddof=0)
    out[seed_end] = _shrunk_corr(s, shrinkage)
    for t in range(seed_end + 1, t_len):
        if valid[t]:
            rt = r[t]
            s = lam * s + (1.0 - lam) * np.outer(rt, rt)
        out[t] = _shrunk_corr(s, shrinkage)
    return out


def _shrunk_corr(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    d = np.where(d > 1e-12, d, 1.0)
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    if shrinkage > 0:
        n = corr.shape[0]
        off = ~np.eye(n, dtype=bool)
        target = np.full_like(corr, corr[off].mean())
        np.fill_diagonal(target, 1.0)
        corr = (1.0 - shrinkage) * corr + shrinkage * target
    return corr


def build_covariance(
    ann_vol: pd.DataFrame,
    corr: np.ndarray,
    trading_days: int = 252,
) -> np.ndarray:
    """年率ボラ (T×N) と相関 (T×N×N) から日次共分散 (T×N×N) を組む。"""
    daily_vol = ann_vol.to_numpy(dtype=float) / np.sqrt(trading_days)
    d = daily_vol[:, :, None] * daily_vol[:, None, :]
    return corr * d
