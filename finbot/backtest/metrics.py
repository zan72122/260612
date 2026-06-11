"""パフォーマンス指標。

シャープ系の点推定に加えて、多重検定を補正する
Probabilistic / Deflated Sharpe Ratio(Bailey & López de Prado 2014)を提供する。
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

_NORM = NormalDist()
_EULER_GAMMA = 0.5772156649015329


def annualized_return(returns: pd.Series, trading_days: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (trading_days / len(returns)) - 1.0


def annualized_vol(returns: pd.Series, trading_days: int = 252) -> float:
    return float(returns.std(ddof=1)) * np.sqrt(trading_days)


def sharpe_ratio(
    returns: pd.Series, rf_rate: float = 0.0, trading_days: int = 252
) -> float:
    excess = returns - rf_rate / trading_days
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd) * np.sqrt(trading_days)


def sortino_ratio(
    returns: pd.Series, rf_rate: float = 0.0, trading_days: int = 252
) -> float:
    excess = returns - rf_rate / trading_days
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    dd = np.sqrt(float((downside**2).mean()))
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd) * np.sqrt(trading_days)


def max_drawdown(returns: pd.Series) -> float:
    """最大ドローダウン(負値, 例 -0.18)。"""
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def calmar_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("inf")
    return annualized_return(returns, trading_days) / mdd


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR = P(真のシャープ > sr_benchmark)。

    PSR(SR*) = Φ[(ŜR − SR*)·√(T−1) / √(1 − γ₃ŜR + ((γ₄−1)/4)ŜR²)]
    ŜR と sr_benchmark は「日次(非年率)」のシャープであることに注意。
    歪度・尖度によるシャープ推定量の標準誤差の補正が入っている。
    """
    r = returns.dropna()
    t_len = len(r)
    sd = float(r.std(ddof=1))
    if t_len < 3 or sd == 0 or np.isnan(sd):
        return 0.0
    sr = float(r.mean()) / sd
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0  # pandas は超過尖度を返す
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return 0.0
    z = (sr - sr_benchmark) * np.sqrt(t_len - 1) / np.sqrt(denom)
    return float(_NORM.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """N 回の独立試行でゼロスキルでも出てしまう最大シャープの期待値(日次単位)。

    SR₀ = √V[SR]·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))], γ = Euler-Mascheroni
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    q1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    q2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * np.e))
    return float(
        np.sqrt(sr_variance) * ((1.0 - _EULER_GAMMA) * q1 + _EULER_GAMMA * q2)
    )


def deflated_sharpe_ratio(
    returns: pd.Series, trial_sharpes_daily: "list[float] | np.ndarray"
) -> float:
    """DSR = PSR(SR₀)。試行回数と試行間バラつきでデフレートしたシャープの有意性。

    trial_sharpes_daily: 探索で試した全コンフィグの「日次」シャープのリスト。
    0.95 を超えて初めて「選択バイアスを補正してもスキルが残る」と言える。
    """
    trials = np.asarray(trial_sharpes_daily, dtype=float)
    trials = trials[~np.isnan(trials)]
    sr0 = expected_max_sharpe(len(trials), float(np.var(trials, ddof=1)) if len(trials) > 1 else 0.0)
    return probabilistic_sharpe_ratio(returns, sr0)


def summary(
    returns: pd.Series,
    rf_rate: float = 0.0,
    trading_days: int = 252,
    turnover: pd.Series | None = None,
) -> dict[str, float]:
    out = {
        "ann_return": annualized_return(returns, trading_days),
        "ann_vol": annualized_vol(returns, trading_days),
        "sharpe": sharpe_ratio(returns, rf_rate, trading_days),
        "sortino": sortino_ratio(returns, rf_rate, trading_days),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, trading_days),
        "hit_rate": float((returns > 0).mean()),
        "n_days": float(len(returns)),
    }
    if turnover is not None and len(turnover):
        out["ann_turnover"] = float(turnover.mean()) * trading_days
    return out


def format_report(stats: dict[str, float], title: str = "Backtest") -> str:
    lines = [f"=== {title} ===" ]
    fmt = {
        "ann_return": ("年率リターン", "{:+.2%}"),
        "ann_vol": ("年率ボラティリティ", "{:.2%}"),
        "sharpe": ("シャープレシオ", "{:.2f}"),
        "sortino": ("ソルティノレシオ", "{:.2f}"),
        "max_drawdown": ("最大ドローダウン", "{:+.2%}"),
        "calmar": ("カルマーレシオ", "{:.2f}"),
        "hit_rate": ("勝率(日次)", "{:.1%}"),
        "ann_turnover": ("年間ターンオーバー", "{:.1f}x"),
        "n_days": ("評価日数", "{:.0f}"),
    }
    for key, (label, f) in fmt.items():
        if key in stats:
            lines.append(f"{label:<12}: {f.format(stats[key])}")
    return "\n".join(lines)
