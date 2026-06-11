import numpy as np
import pandas as pd

from finbot.config import BotConfig
from finbot.data import SyntheticSource
from finbot.risk.covariance import blended_vol
from finbot.signals import carry_signal, combined_forecast, ewmac_trend, scale_forecast


def _trending_prices(n_days=500, n_assets=3, slope=0.002, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rets = slope + rng.normal(0, 0.005, (n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_ewmac_positive_on_uptrend():
    sig = ewmac_trend(_trending_prices(slope=0.002))
    assert (sig.iloc[-1] > 0).all()


def test_ewmac_negative_on_downtrend():
    sig = ewmac_trend(_trending_prices(slope=-0.002))
    assert (sig.iloc[-1] < 0).all()


def test_ewmac_response_saturates_on_extreme_trend():
    """応答関数 z·exp(-z²/4) は極端トレンド(|z|≫√2)でシグナルを縮める。"""
    smooth = _trending_prices(slope=0.003, seed=2)
    # ほぼノイズなしの直線的トレンド → z が大きく、応答は小さくなる
    n = len(smooth)
    dates = smooth.index
    pure = pd.DataFrame(
        {"A0": 100 * np.exp(0.003 * np.arange(n) + 1e-4 * np.sin(np.arange(n)))},
        index=dates,
    )
    sig = ewmac_trend(pure).dropna()
    assert sig["A0"].iloc[-1] < 0.5  # サインルールなら 1.0 になるところ


def test_ewmac_bounded_by_response_function():
    """応答関数 z·exp(-z²/4)/0.89 の上限は |u| = √2·e^{-1/2}/0.89 < 1。"""
    prices = SyntheticSource(days=800, seed=7).load()
    sig = ewmac_trend(prices).dropna()
    assert (sig.abs() <= 1.0 + 1e-9).all().all()


def test_carry_signal_sign_follows_carry():
    dates = pd.bdate_range("2020-01-01", periods=300)
    carry = pd.DataFrame(
        {"A": 0.05, "B": -0.05}, index=dates
    )
    vol = pd.DataFrame({"A": 0.10, "B": 0.10}, index=dates)
    sig = carry_signal(carry, vol, smooth_span=30).dropna()
    assert (sig["A"] > 0).all()
    assert (sig["B"] < 0).all()


def test_scale_forecast_targets_abs_10_and_caps():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=1000)
    raw = pd.DataFrame(rng.normal(0, 0.5, (1000, 4)), index=dates)
    f = scale_forecast(raw, target_abs=10.0, cap=20.0, burn=63)
    tail = f.iloc[-250:]
    assert 7.0 < tail.abs().mean().mean() < 13.0
    assert (f.abs().fillna(0.0) <= 20.0 + 1e-9).all().all()


def test_combined_forecast_causality():
    """t 日のフォーキャストは t 日以前の価格・キャリーのみに依存する(先読みなし)。"""
    cfg = BotConfig()
    src = SyntheticSource(days=900, seed=3)
    prices, carry = src.load(), src.load_carry()
    vol = blended_vol(prices.pct_change())
    f_full = combined_forecast(prices, cfg, vol, carry)

    cut = 700
    altered = prices.copy()
    altered.iloc[cut:] *= 5.0  # 未来の価格を大幅に変える
    vol_alt = blended_vol(altered.pct_change())
    f_altered = combined_forecast(altered, cfg, vol_alt, carry)

    pd.testing.assert_frame_equal(f_full.iloc[:cut], f_altered.iloc[:cut])


def test_combined_forecast_works_without_carry():
    cfg = BotConfig()
    prices = SyntheticSource(days=700, seed=4).load()
    vol = blended_vol(prices.pct_change())
    f = combined_forecast(prices, cfg, vol, carry=None)
    assert f.iloc[-1].notna().all()
