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
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        prices = pd.read_csv(self.path, index_col=0, parse_dates=True)
        return self.validate(prices)
