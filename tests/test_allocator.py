import numpy as np
import pandas as pd

from finbot.config import BotConfig
from finbot.data import SyntheticSource
from finbot.portfolio import compute_target_weights
from finbot.risk.covariance import blended_vol, build_covariance, ewma_correlation
from finbot.risk.vol_target import vol_target_leverage


def test_weights_respect_constraints():
    cfg = BotConfig()
    prices = SyntheticSource(days=1200, seed=5).load()
    w = compute_target_weights(prices, cfg)
    assert (w.abs() <= cfg.max_weight + 1e-9).all().all()
    assert (w.abs().sum(axis=1) <= cfg.max_leverage + 1e-6).all()


def test_weights_zero_during_warmup():
    cfg = BotConfig()
    prices = SyntheticSource(days=1200, seed=5).load()
    w = compute_target_weights(prices, cfg)
    assert (w.iloc[: cfg.warmup] == 0.0).all().all()


def test_weights_causality():
    """t 日のウェイトは未来の価格に依存しない。"""
    cfg = BotConfig()
    src = SyntheticSource(days=1100, seed=11)
    prices, carry = src.load(), src.load_carry()
    w_full = compute_target_weights(prices, cfg, carry)

    cut = 900
    altered = prices.copy()
    altered.iloc[cut:] *= 3.0
    w_altered = compute_target_weights(altered, cfg, carry)

    pd.testing.assert_frame_equal(w_full.iloc[:cut], w_altered.iloc[:cut])


def test_carry_changes_weights():
    """キャリーを与えるとトレンド単独と異なる配分になる(統合層が効いている)。"""
    cfg = BotConfig()
    src = SyntheticSource(days=1100, seed=13)
    prices, carry = src.load(), src.load_carry()
    w_trend_only = compute_target_weights(prices, cfg)
    w_with_carry = compute_target_weights(prices, cfg, carry)
    diff = (w_trend_only - w_with_carry).abs().to_numpy().max()
    assert diff > 1e-4


def test_vol_target_leverage_scales_to_target():
    n = 4
    cov = np.eye(n) * (0.01**2)  # 日次ボラ1%
    w = np.full(n, 0.25)
    lev = vol_target_leverage(w, cov, target_vol=0.10, max_leverage=10.0)
    port_vol = np.sqrt(float((w * lev) @ cov @ (w * lev)) * 252)
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


def test_correlation_matrix_well_formed():
    prices = SyntheticSource(days=900, seed=17).load()
    rets = prices.pct_change()
    corr = ewma_correlation(rets, halflife=200, shrinkage=0.2)
    last = corr[-1]
    n = last.shape[0]
    np.testing.assert_allclose(np.diag(last), np.ones(n))
    np.testing.assert_allclose(last, last.T)
    assert (np.abs(last) <= 1.0 + 1e-9).all()


def test_blended_vol_floors_quiet_periods():
    """静穏期でも長期アンカーがボラ推定の床になる(過大レバレッジ防止)。"""
    dates = pd.bdate_range("2018-01-01", periods=1000)
    rng = np.random.default_rng(0)
    # 前半は高ボラ、後半はほぼ無ボラ
    r = np.concatenate([rng.normal(0, 0.02, 500), rng.normal(0, 0.001, 500)])
    rets = pd.DataFrame({"A": r}, index=dates)
    vol = blended_vol(rets, halflife=20, blend=0.7, long_min_periods=252)
    short_only = rets.ewm(halflife=20, min_periods=20).std() * np.sqrt(252)
    assert vol["A"].iloc[-1] > short_only["A"].iloc[-1]


def test_build_covariance_consistent():
    prices = SyntheticSource(days=900, seed=19).load()
    rets = prices.pct_change()
    vol = blended_vol(rets).fillna(0.0)
    corr = ewma_correlation(rets)
    cov = build_covariance(vol, corr)
    t = -1
    implied_vol = np.sqrt(np.diag(cov[t]) * 252)
    np.testing.assert_allclose(implied_vol, vol.iloc[t].to_numpy(), rtol=1e-9)
