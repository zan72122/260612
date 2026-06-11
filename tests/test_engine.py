import numpy as np
import pandas as pd

from finbot.backtest.engine import run_backtest
from finbot.backtest.walkforward import run_walkforward
from finbot.config import BotConfig
from finbot.data import SyntheticSource


def test_no_lookahead():
    """未来の価格を改変しても、それ以前の戦略リターンは一切変わらない。"""
    cfg = BotConfig()
    prices = SyntheticSource(days=1200, seed=21).load()
    res_full = run_backtest(prices, cfg)

    cut = 1000
    altered = prices.copy()
    altered.iloc[cut:] *= 0.5
    res_altered = run_backtest(altered, cfg)

    pd.testing.assert_series_equal(
        res_full.returns.iloc[:cut], res_altered.returns.iloc[:cut]
    )


def test_equity_consistent_with_returns():
    cfg = BotConfig()
    prices = SyntheticSource(days=800, seed=22).load()
    res = run_backtest(prices, cfg)
    np.testing.assert_allclose(
        res.equity.to_numpy(), (1.0 + res.returns).cumprod().to_numpy()
    )


def test_realized_vol_near_target():
    """ボラターゲティングにより実現ボラが目標近傍に収まる。"""
    cfg = BotConfig()
    prices = SyntheticSource(days=2520, seed=42).load()
    res = run_backtest(prices, cfg)
    realized = res.stats["ann_vol"]
    assert 0.5 * cfg.target_vol < realized < 1.5 * cfg.target_vol


def test_costs_reduce_returns():
    prices = SyntheticSource(days=1500, seed=23).load()
    free = run_backtest(prices, BotConfig().with_overrides(cost_bps=0.0))
    costly = run_backtest(prices, BotConfig().with_overrides(cost_bps=50.0))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_rebalance_band_reduces_turnover():
    prices = SyntheticSource(days=1500, seed=24).load()
    tight = run_backtest(prices, BotConfig().with_overrides(rebalance_band=0.0))
    wide = run_backtest(prices, BotConfig().with_overrides(rebalance_band=0.10))
    assert wide.stats["ann_turnover"] < tight.stats["ann_turnover"]


def test_warmup_has_no_positions():
    cfg = BotConfig()
    prices = SyntheticSource(days=900, seed=25).load()
    res = run_backtest(prices, cfg)
    assert (res.returns.iloc[: cfg.warmup] == 0.0).all()
    assert (res.weights.iloc[: cfg.warmup] == 0.0).all().all()


def test_walkforward_runs_and_reports_oos():
    cfg = BotConfig()
    prices = SyntheticSource(days=1600, seed=26).load()
    res = run_walkforward(prices, cfg, train_days=756, test_days=252)
    assert len(res.folds) >= 2
    assert "sharpe" in res.stats
    assert len(res.oos_returns) == sum(1 for _ in res.oos_returns)
    # OOS リターンはテスト窓のみ(train と重複しない)
    first_test_start = pd.Timestamp(res.folds[0]["test_start"])
    assert res.oos_returns.index.min() >= first_test_start
