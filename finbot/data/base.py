"""データソースの抽象インタフェース。

すべてのソースは「日付 index × 資産 columns の終値 DataFrame」を返す。
バックテスト・ペーパートレード・実運用が同じ形式を共有するため、
ソースを差し替えるだけで全パイプラインがそのまま動く。

OHLC を提供できるソースは load_ohlc() を実装する(任意)。
OHLC があると Yang-Zhang レンジボラ推定が有効になり、
終値のみのソースでは従来の終値ベース推定に自動フォールバックする。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import pandas as pd


class OHLCFrames(NamedTuple):
    """始値・高値・安値・終値の DataFrame の組(index/columns は共通)。"""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame

    def window(self, start: int, stop: int) -> "OHLCFrames":
        """iloc ベースの行スライス(ウォークフォワードのフォールド分割用)。"""
        return OHLCFrames(*(df.iloc[start:stop] for df in self))

    def aligned_to(self, prices: pd.DataFrame) -> "OHLCFrames":
        """終値 DataFrame と同じ index/columns に整列する。"""
        return OHLCFrames(
            *(df.reindex(index=prices.index, columns=prices.columns) for df in self)
        )


class DataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """終値の DataFrame(index: DatetimeIndex, columns: 資産名)を返す。"""

    def load_ohlc(self) -> OHLCFrames | None:
        """OHLC を返す。提供できないソースは None(終値ベースにフォールバック)。"""
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
