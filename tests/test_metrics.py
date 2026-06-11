import numpy as np
import pandas as pd
import pytest

from finbot.backtest.metrics import (
    annualized_return,
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summary,
)


def test_sharpe_known_value():
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(0.001, 0.01, 252 * 4))
    expected = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(daily) == pytest.approx(expected)


def test_sharpe_zero_vol_returns_zero():
    assert sharpe_ratio(pd.Series([0.0] * 100)) == 0.0


def test_sharpe_rf_reduces_sharpe():
    rets = pd.Series([0.001] * 200 + [-0.0005] * 100)
    assert sharpe_ratio(rets, rf_rate=0.05) < sharpe_ratio(rets, rf_rate=0.0)


def test_max_drawdown_constructed():
    # 100 -> 110 -> 55 -> 66 : 最大DDはピーク110から55の -50%
    rets = pd.Series([0.10, -0.50, 0.20])
    assert max_drawdown(rets) == pytest.approx(-0.50)


def test_max_drawdown_monotonic_up_is_zero():
    assert max_drawdown(pd.Series([0.01] * 50)) == 0.0


def test_annualized_return_constant():
    rets = pd.Series([0.001] * 252)
    assert annualized_return(rets) == pytest.approx(1.001**252 - 1)


def test_annualized_vol():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0, 0.01, 5000))
    assert annualized_vol(rets) == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_sortino_ignores_upside_vol():
    base = pd.Series([0.001, -0.001] * 100)
    upside_heavy = pd.Series([0.01, -0.001] * 100)
    assert sortino_ratio(upside_heavy) > sortino_ratio(base)


def test_summary_keys():
    rets = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 500))
    stats = summary(rets, turnover=pd.Series([0.01] * 500))
    for key in ("ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
                "calmar", "hit_rate", "ann_turnover"):
        assert key in stats
