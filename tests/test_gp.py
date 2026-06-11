import numpy as np

from finbot.backtest.engine import run_backtest
from finbot.config import BotConfig
from finbot.data import SyntheticSource
from finbot.portfolio.gp import partial_adjustment
from finbot.signals.ensemble import gp_lookback_weights


def test_persistence_weights_favor_slow_signals():
    """遅いシグナル(長いルックバック)ほど重い: w = ℓa/(1+ℓa)。"""
    lbs = (21, 63, 126, 252)
    w = gp_lookback_weights(lbs, trade_rate=0.25)
    assert all(b > a for a, b in zip(w, w[1:]))
    assert all(0.0 < x < 1.0 for x in w)
    # 解析値の検算: ℓ=21, a=0.25 → 5.25/6.25
    assert abs(w[0] - 5.25 / 6.25) < 1e-12


def test_partial_adjustment_converges_to_constant_aim():
    aim = np.tile(np.array([0.3, -0.2]), (300, 1))
    out = partial_adjustment(aim, trade_rate=0.25)
    np.testing.assert_allclose(out[-1], aim[-1], atol=1e-6)
    # 初日は τ·aim から始まる
    np.testing.assert_allclose(out[0], 0.25 * aim[0])


def test_partial_adjustment_reduces_trading():
    rng = np.random.default_rng(0)
    aim = rng.standard_normal((500, 4)) * 0.1
    slow = partial_adjustment(aim, trade_rate=0.2)
    fast = partial_adjustment(aim, trade_rate=1.0)  # 全量リバランス(=エイムそのもの)
    np.testing.assert_allclose(fast, aim)
    turn_slow = np.abs(np.diff(slow, axis=0)).sum()
    turn_fast = np.abs(np.diff(fast, axis=0)).sum()
    assert turn_slow < turn_fast


def test_partial_adjustment_preserves_box_constraints():
    """凸結合なので |w| の上限は保存される。"""
    rng = np.random.default_rng(1)
    aim = np.clip(rng.standard_normal((400, 3)), -0.3, 0.3)
    out = partial_adjustment(aim, trade_rate=0.3)
    assert np.abs(out).max() <= 0.3 + 1e-12


def test_gp_lowers_turnover_in_backtest():
    prices = SyntheticSource(days=1500, seed=37).load()
    base = BotConfig().with_overrides(n_tranches=1)
    gp = run_backtest(prices, base.with_overrides(gp_enabled=True, gp_trade_rate=0.15))
    naive = run_backtest(
        prices, base.with_overrides(gp_enabled=True, gp_trade_rate=1.0)
    )
    assert gp.stats["ann_turnover"] < naive.stats["ann_turnover"]


def test_gp_disabled_falls_back_to_ewma_smoothing():
    prices = SyntheticSource(days=1200, seed=38).load()
    on = run_backtest(prices, BotConfig().with_overrides(gp_enabled=True))
    off = run_backtest(prices, BotConfig().with_overrides(gp_enabled=False))
    assert not np.allclose(on.returns.to_numpy(), off.returns.to_numpy())
