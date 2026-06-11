"""ペーパートレードの実行ループ(1ティック = 1営業日終値後の処理)。

cron 等で日次実行する想定:
    python -m finbot paper --state state.json

バックテストと同一の compute_target_weights を使うため、
検証済みのロジックがそのまま動く。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finbot.config import BotConfig
from finbot.data.base import DataSource
from finbot.live.paper_broker import Order, PaperBroker
from finbot.portfolio.allocator import compute_target_weights


def run_paper_tick(
    source: DataSource,
    cfg: BotConfig,
    state_path: str | Path = "paper_state.json",
    initial_cash: float = 1_000_000.0,
) -> tuple[PaperBroker, list[Order], pd.Series]:
    """最新データで目標ウェイトを計算し、ペーパーブローカで執行する。"""
    prices = source.load()
    if len(prices) < cfg.warmup + 1:
        raise ValueError(f"履歴不足: 最低 {cfg.warmup + 1} 営業日の価格が必要")

    state_path = Path(state_path)
    if state_path.exists():
        broker = PaperBroker.load(state_path)
    else:
        broker = PaperBroker(initial_cash=initial_cash, cost_bps=cfg.cost_bps)

    target = compute_target_weights(prices, cfg).iloc[-1]
    latest = prices.iloc[-1]
    orders = broker.rebalance_to(target, latest)
    broker.save(state_path)
    return broker, orders, target
