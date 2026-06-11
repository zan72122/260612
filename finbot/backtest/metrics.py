"""パフォーマンス指標。"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
