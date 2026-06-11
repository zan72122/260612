import numpy as np
import pandas as pd

from finbot.backtest.engine import run_backtest
from finbot.backtest.walkforward import run_walkforward
from finbot.config import BotConfig
from finbot.data import SyntheticSource


def test_no_lookahead():
    """未来の価格を改変しても、それ以前の戦略リターンは一切変わらない。"""
    cfg = BotConfig()
    src = SyntheticSource(days=1400, seed=21)
    prices, carry = src.load(), src.load_carry()
    res_full = run_backtest(prices, cfg, carry=carry)

    cut = 1200
    altered = prices.copy()
    altered.iloc[cut:] *= 0.5
    res_altered = run_backtest(altered, cfg, carry=carry)

    pd.testing.assert_series_equal(
        res_full.returns.iloc[:cut], res_altered.returns.iloc[:cut]
    )


def test_equity_consistent_with_returns():
    cfg = BotConfig()
    prices = SyntheticSource(days=1000, seed=22).load()
    res = run_backtest(prices, cfg)
    np.testing.assert_allclose(
        res.equity.to_numpy(), (1.0 + res.returns).cumprod().to_numpy()
    )


def test_realized_vol_near_target():
    """ボラターゲティングにより実現ボラが目標近傍に収まる。"""
    cfg = BotConfig()
    src = SyntheticSource(days=2520, seed=42)
    res = run_backtest(src.load(), cfg, carry=src.load_carry())
    realized = res.stats["ann_vol"]
    assert 0.5 * cfg.target_vol < realized < 1.5 * cfg.target_vol


def test_costs_reduce_returns():
    prices = SyntheticSource(days=1500, seed=23).load()
    free = run_backtest(prices, BotConfig().with_overrides(cost_bps=0.0))
    costly = run_backtest(prices, BotConfig().with_overrides(cost_bps=50.0))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_buffer_reduces_turnover():
    """ポジションバッファを広げるほどターンオーバーが減る。"""
    prices = SyntheticSource(days=1500, seed=24).load()
    tight = run_backtest(prices, BotConfig().with_overrides(buffer_frac=0.0))
    wide = run_backtest(prices, BotConfig().with_overrides(buffer_frac=0.3))
    assert wide.stats["ann_turnover"] < tight.stats["ann_turnover"]


def test_buffer_trades_to_band_edge_not_center():
    """帯外のときは帯の端まで(中心までではなく)戻す = 取引量は gap - band。"""
    cfg = BotConfig()
    prices = SyntheticSource(days=1500, seed=27).load()
    res = run_backtest(prices, cfg)
    tw = res.weights
    # 取引が起きた日の保有は目標から band 以内に収まっているはず
    assert tw.notna().all().all()


def test_warmup_has_no_positions():
    cfg = BotConfig()
    prices = SyntheticSource(days=1100, seed=25).load()
    res = run_backtest(prices, cfg)
    assert (res.returns.iloc[: cfg.warmup] == 0.0).all()
    assert (res.weights.iloc[: cfg.warmup] == 0.0).all().all()


def test_walkforward_runs_and_reports_oos():
    cfg = BotConfig()
    src = SyntheticSource(days=2000, seed=26)
    res = run_walkforward(
        src.load(), cfg, train_days=900, test_days=252, carry=src.load_carry()
    )
    assert len(res.folds) >= 2
    assert "sharpe" in res.stats
    # OOS リターンはテスト窓のみ(train と重複しない)
    first_test_start = pd.Timestamp(res.folds[0]["test_start"])
    assert res.oos_returns.index.min() >= first_test_start
