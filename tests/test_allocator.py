import numpy as np
import pandas as pd

from finbot.config import BotConfig
from finbot.data import SyntheticSource
from finbot.portfolio import compute_target_weights
from finbot.risk.drawdown import drawdown_multiplier
from finbot.risk.vol_target import vol_target_leverage


def test_weights_respect_constraints():
    cfg = BotConfig()
    prices = SyntheticSource(days=1000, seed=5).load()
    w = compute_target_weights(prices, cfg)
    assert (w.abs() <= cfg.max_weight + 1e-9).all().all()
    assert (w.abs().sum(axis=1) <= cfg.max_leverage + 1e-6).all()


def test_weights_zero_during_warmup():
    cfg = BotConfig()
    prices = SyntheticSource(days=1000, seed=5).load()
    w = compute_target_weights(prices, cfg)
    assert (w.iloc[: cfg.warmup] == 0.0).all().all()


def test_weights_causality():
    """t 日のウェイトは未来の価格に依存しない。"""
    cfg = BotConfig()
    prices = SyntheticSource(days=900, seed=11).load()
    w_full = compute_target_weights(prices, cfg)

    cut = 700
    altered = prices.copy()
    altered.iloc[cut:] *= 3.0
    w_altered = compute_target_weights(altered, cfg)

    pd.testing.assert_frame_equal(w_full.iloc[:cut], w_altered.iloc[:cut])


def test_vol_target_leverage_scales_to_target():
    n = 4
    cov = np.eye(n) * (0.01**2)  # 日次ボラ1%
    w = np.full(n, 0.25)
    lev = vol_target_leverage(w, cov, target_vol=0.10, max_leverage=10.0)
    port_vol = np.sqrt(float((w * lev) @ cov @ (w * lev)) * 252)
    assert port_vol == np.float64(port_vol)
    assert abs(port_vol - 0.10) < 1e-9


def test_vol_target_respects_max_leverage():
    n = 2
    cov = np.eye(n) * (0.0001**2)  # ほぼ無リスク → 巨大レバレッジを要求
    w = np.full(n, 0.5)
    lev = vol_target_leverage(w, cov, target_vol=0.10, max_leverage=1.5)
    assert np.abs(w * lev).sum() <= 1.5 + 1e-9


def test_vol_target_zero_on_nan_cov():
    cov = np.full((2, 2), np.nan)
    assert vol_target_leverage(np.array([0.5, 0.5]), cov, 0.1, 1.5) == 0.0


def test_drawdown_multiplier_profile():
    assert drawdown_multiplier(0.0, 0.08, 0.20, 0.25) == 1.0
    assert drawdown_multiplier(-0.05, 0.08, 0.20, 0.25) == 1.0
    assert drawdown_multiplier(-0.30, 0.08, 0.20, 0.25) == 0.25
    mid = drawdown_multiplier(-0.14, 0.08, 0.20, 0.25)
    assert 0.25 < mid < 1.0
