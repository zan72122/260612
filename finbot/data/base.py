"""データソースの抽象インタフェース。

すべてのソースは「日付 index × 資産 columns の終値 DataFrame」を返す。
バックテスト・ペーパートレード・実運用が同じ形式を共有するため、
ソースを差し替えるだけで全パイプラインがそのまま動く。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """終値の DataFrame(index: DatetimeIndex, columns: 資産名)を返す。"""

    @staticmethod
    def validate(prices: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.set_index(pd.DatetimeIndex(prices.index))
        prices = prices.sort_index()
        if prices.isna().any().any():
            prices = prices.ffill().dropna()
        if (prices <= 0).any().any():
            raise ValueError("価格に 0 以下の値が含まれています")
        return prices
