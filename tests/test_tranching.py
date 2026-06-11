import numpy as np
import pandas as pd

from finbot.backtest.engine import _simulate_tranche, run_backtest
from finbot.config import BotConfig
from finbot.data import SyntheticSource
from finbot.portfolio import compute_target_weights


def test_timing_luck_exists_across_phases():
    """位相だけが違う同一戦略の成績は実際にズレる(=タイミング運の存在)。"""
    cfg = BotConfig()
    prices = SyntheticSource(days=1500, seed=31).load()
    tw = compute_target_weights(prices, cfg)
    rets = prices.pct_change().to_numpy(dtype=float)
    finals = [
        _simulate_tranche(rets, tw.to_numpy(dtype=float), cfg, offset=j, period=5)[0][-1]
        for j in range(5)
    ]
    assert np.std(finals) > 0.0
    assert len(set(np.round(finals, 12))) > 1


def test_single_tranche_allows_daily_rebalance():
    """n_tranches=1 は毎日リバランス可(従来挙動)。"""
    cfg = BotConfig().with_overrides(n_tranches=1)
    prices = SyntheticSource(days=1200, seed=32).load()
    res = run_backtest(prices, cfg)
    assert res.stats["ann_turnover"] > 0


def test_tranched_differs_from_single():
    prices = SyntheticSource(days=1500, seed=33).load()
    one = run_backtest(prices, BotConfig().with_overrides(n_tranches=1))
    five = run_backtest(prices, BotConfig().with_overrides(n_tranches=5))
    assert not np.allclose(one.returns.to_numpy(), five.returns.to_numpy())


def test_tranched_no_lookahead():
    """トランチング有効時も未来の価格改変が過去のリターンに影響しない。"""
    cfg = BotConfig().with_overrides(n_tranches=5)
    prices = SyntheticSource(days=1200, seed=34).load()
    res_full = run_backtest(prices, cfg)

    cut = 1000
    altered = prices.copy()
    altered.iloc[cut:] *= 0.5
    res_altered = run_backtest(altered, cfg)
    pd.testing.assert_series_equal(
        res_full.returns.iloc[:cut], res_altered.returns.iloc[:cut]
    )


def test_tranched_equity_consistent_with_returns():
    cfg = BotConfig().with_overrides(n_tranches=5)
    prices = SyntheticSource(days=1000, seed=35).load()
    res = run_backtest(prices, cfg)
    np.testing.assert_allclose(
        res.equity.to_numpy(), (1.0 + res.returns).cumprod().to_numpy()
    )


def test_tranched_vol_near_target():
    cfg = BotConfig().with_overrides(n_tranches=5)
    prices = SyntheticSource(days=2520, seed=42).load()
    res = run_backtest(prices, cfg)
    assert 0.5 * cfg.target_vol < res.stats["ann_vol"] < 1.5 * cfg.target_vol


def test_combined_equity_is_mean_of_tranches():
    """合成エクイティはトランシェ・エクイティの等加重平均と一致する。"""
    cfg = BotConfig().with_overrides(n_tranches=4)
    prices = SyntheticSource(days=1200, seed=36).load()
    tw = compute_target_weights(prices, cfg)
    rets = prices.pct_change().to_numpy(dtype=float)
    paths = np.array(
        [
            _simulate_tranche(rets, tw.to_numpy(dtype=float), cfg, offset=j, period=4)[0]
            for j in range(4)
        ]
    )
    res = run_backtest(prices, cfg, target_weights=tw)
    np.testing.assert_allclose(
        res.equity.to_numpy(), paths.mean(axis=0), rtol=1e-10
    )
