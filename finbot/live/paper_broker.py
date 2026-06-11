"""ペーパートレード用ブローカ。

実ブローカへ差し替える際はこのクラスと同じインタフェース
(portfolio_value / rebalance_to)を実装したアダプタを用意する。
状態は JSON に永続化され、ティックをまたいで保持される。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Order:
    asset: str
    units: float       # 正: 買い, 負: 売り
    price: float

    @property
    def notional(self) -> float:
        return self.units * self.price


class PaperBroker:
    def __init__(self, initial_cash: float = 1_000_000.0, cost_bps: float = 5.0):
        self.cash = initial_cash
        self.cost_bps = cost_bps
        self.positions: dict[str, float] = {}  # 資産名 -> 保有口数

    def portfolio_value(self, prices: pd.Series) -> float:
        pos_value = sum(
            units * prices[asset] for asset, units in self.positions.items()
        )
        return self.cash + pos_value

    def rebalance_to(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        min_weight_gap: float = 0.001,
    ) -> list[Order]:
        """目標ウェイトへリバランスし、執行した注文一覧を返す。

        min_weight_gap: ウェイト乖離がこれ未満の資産は取引しない
        (コストだけ発生する微小注文の抑制)。
        """
        equity = self.portfolio_value(prices)
        orders: list[Order] = []
        for asset, w in target_weights.items():
            price = float(prices[asset])
            current_units = self.positions.get(asset, 0.0)
            target_units = equity * float(w) / price
            delta = target_units - current_units
            if abs(delta * price) < min_weight_gap * equity:
                continue
            cost = abs(delta * price) * self.cost_bps * 1e-4
            self.cash -= delta * price + cost
            self.positions[asset] = target_units
            orders.append(Order(asset=asset, units=delta, price=price))
        return orders

    # --- 永続化 ---

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"cash": self.cash, "cost_bps": self.cost_bps, "positions": self.positions},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "PaperBroker":
        data = json.loads(Path(path).read_text())
        broker = cls(initial_cash=data["cash"], cost_bps=data["cost_bps"])
        broker.positions = dict(data["positions"])
        return broker
