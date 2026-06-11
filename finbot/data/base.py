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

    def load_carry(self) -> pd.DataFrame | None:
        """年率キャリー(価格が動かない場合の期待リターン)の DataFrame を返す。

        形式は load() と同じ(index: 日付, columns: 資産)。t 行は t 日時点で
        観測可能な値であること(因果性)。提供できないソースは None を返し、
        その場合パイプラインはトレンド単独で動く。
        """
        return None

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
