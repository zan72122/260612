import numpy as np
import pandas as pd

from finbot.config import BotConfig
from finbot.data import SyntheticSource
from finbot.signals import combined_signal, ts_momentum, xs_momentum


def _trending_prices(n_days=400, n_assets=3, slope=0.002, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rets = slope + rng.normal(0, 0.005, (n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_ts_momentum_positive_on_uptrend():
    sig = ts_momentum(_trending_prices(slope=0.002))
    assert (sig.iloc[-1] > 0.5).all()


def test_ts_momentum_negative_on_downtrend():
    sig = ts_momentum(_trending_prices(slope=-0.002))
    assert (sig.iloc[-1] < -0.5).all()


def test_ts_momentum_bounded():
    prices = SyntheticSource(days=600, seed=7).load()
    sig = ts_momentum(prices).dropna()
    assert (sig.abs() <= 1.0 + 1e-9).all().all()


def test_xs_momentum_ranks_best_and_worst():
    prices = _trending_prices(n_assets=4, slope=0.0)
    # A0 を強い上昇、A3 を強い下落に上書き
    n = len(prices)
    prices["A0"] = 100 * np.exp(np.linspace(0, 1.0, n))
    prices["A3"] = 100 * np.exp(np.linspace(0, -1.0, n))
    sig = xs_momentum(prices, lookback=126)
    assert sig["A0"].iloc[-1] == 1.0
    assert sig["A3"].iloc[-1] == -1.0
    assert (sig.dropna().abs() <= 1.0).all().all()


def test_combined_signal_causality():
    """t 日のシグナルは t 日以前の価格のみに依存する(先読みなし)。"""
    cfg = BotConfig()
    prices = SyntheticSource(days=800, seed=3).load()
    sig_full = combined_signal(prices, cfg)

    cut = 600
    altered = prices.copy()
    altered.iloc[cut:] *= 5.0  # 未来の価格を大幅に変える
    sig_altered = combined_signal(altered, cfg)

    pd.testing.assert_frame_equal(
        sig_full.iloc[:cut], sig_altered.iloc[:cut]
    )
