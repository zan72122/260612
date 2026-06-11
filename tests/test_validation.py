"""CPCV と PSR/DSR の検証。"""

import math

import numpy as np
import pandas as pd
import pytest

from finbot.backtest.cpcv import run_cpcv
from finbot.backtest.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from finbot.config import BotConfig
from finbot.data import SyntheticSource


def _returns(mean, std, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def test_psr_high_for_genuine_skill():
    r = _returns(0.001, 0.01)  # 日次SR 0.1 ≈ 年率1.6
    assert probabilistic_sharpe_ratio(r, 0.0) > 0.99


def test_psr_near_half_for_zero_skill():
    r = _returns(0.0, 0.01, seed=1)
    assert 0.05 < probabilistic_sharpe_ratio(r, 0.0) < 0.95


def test_expected_max_sharpe_grows_with_trials():
    v = 0.001
    assert expected_max_sharpe(100, v) > expected_max_sharpe(10, v) > 0.0
    assert expected_max_sharpe(1, v) == 0.0


def test_dsr_deflates_selection_bias():
    """ゼロスキル試行群から最良を選んでも DSR は高くならない。"""
    rng = np.random.default_rng(7)
    trials = [pd.Series(rng.normal(0, 0.01, 1500)) for _ in range(20)]
    daily_srs = [float(t.mean() / t.std(ddof=1)) for t in trials]
    best = trials[int(np.argmax(daily_srs))]
    psr = probabilistic_sharpe_ratio(best, 0.0)
    dsr = deflated_sharpe_ratio(best, daily_srs)
    assert dsr < psr  # デフレーションが効いている
    assert dsr < 0.95  # 偽発見を採用基準で弾ける


def test_cpcv_runs_and_reports_distribution():
    cfg = BotConfig()
    src = SyntheticSource(days=2520, seed=42)
    res = run_cpcv(
        src.load(), cfg, n_groups=8, k_test=2, carry=src.load_carry()
    )
    # パス数 φ = C(N-1, k-1)
    assert len(res.path_sharpes) == math.comb(7, 1)
    assert 0.0 <= res.pbo <= 1.0
    assert 0.0 <= res.dsr <= 1.0
    assert len(res.combos) == math.comb(8, 2)
    # 各組み合わせの選択は purge/embargo 済みの学習データに基づく
    assert all("omega" in c for c in res.combos)


def test_cpcv_raises_on_short_history():
    cfg = BotConfig()
    prices = SyntheticSource(days=600, seed=1).load()
    with pytest.raises(ValueError):
        run_cpcv(prices, cfg, n_groups=10, k_test=2)
