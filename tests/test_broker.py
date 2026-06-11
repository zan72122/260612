import pandas as pd
import pytest

from finbot.config import BotConfig
from finbot.data import SyntheticSource
from finbot.live.paper_broker import PaperBroker
from finbot.live.runner import run_paper_tick


def test_rebalance_reaches_target_weights():
    broker = PaperBroker(initial_cash=1_000_000, cost_bps=0.0)
    prices = pd.Series({"A": 100.0, "B": 50.0})
    target = pd.Series({"A": 0.3, "B": -0.2})
    broker.rebalance_to(target, prices)
    equity = broker.portfolio_value(prices)
    assert broker.positions["A"] * 100.0 / equity == pytest.approx(0.3, abs=1e-6)
    assert broker.positions["B"] * 50.0 / equity == pytest.approx(-0.2, abs=1e-6)


def test_second_rebalance_is_noop():
    broker = PaperBroker(initial_cash=1_000_000, cost_bps=5.0)
    prices = pd.Series({"A": 100.0, "B": 50.0})
    target = pd.Series({"A": 0.3, "B": -0.2})
    broker.rebalance_to(target, prices)
    orders = broker.rebalance_to(target, prices)
    assert orders == []


def test_save_load_roundtrip(tmp_path):
    broker = PaperBroker(initial_cash=500_000, cost_bps=3.0)
    prices = pd.Series({"A": 100.0})
    broker.rebalance_to(pd.Series({"A": 0.5}), prices)
    path = tmp_path / "state.json"
    broker.save(path)
    loaded = PaperBroker.load(path)
    assert loaded.cash == broker.cash
    assert loaded.positions == broker.positions
    assert loaded.cost_bps == broker.cost_bps


def test_paper_tick_end_to_end(tmp_path):
    cfg = BotConfig()
    source = SyntheticSource(days=600, seed=9)
    state = tmp_path / "state.json"
    broker, orders, target = run_paper_tick(source, cfg, state_path=state)
    assert state.exists()
    assert (target.abs() <= cfg.max_weight + 1e-9).all()
    # 同一データでの2ティック目は実質ノーオペ
    _, orders2, _ = run_paper_tick(source, cfg, state_path=state)
    assert orders2 == []
