"""CSV からの価格読み込み(実データ持ち込み用)。

フォーマット: 1列目が日付、以降の列が資産ごとの終値。
例:
    date,EQ_US,BOND_GOV,GOLD
    2020-01-02,100.0,50.0,150.0
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finbot.data.base import DataSource


class CSVSource(DataSource):
    def __init__(self, path: str | Path, carry_path: str | Path | None = None):
        self.path = Path(path)
        self.carry_path = Path(carry_path) if carry_path else None

    def load(self) -> pd.DataFrame:
        prices = pd.read_csv(self.path, index_col=0, parse_dates=True)
        return self.validate(prices)

    def load_carry(self) -> pd.DataFrame | None:
        """年率キャリーの CSV(価格と同形式、値は例: 0.03 = 年率3%)。"""
        if self.carry_path is None:
            return None
        carry = pd.read_csv(self.carry_path, index_col=0, parse_dates=True)
        return carry.sort_index().ffill()
