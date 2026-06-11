"""Yahoo Finance アダプタ(実運用環境用)。

この開発環境からは市場データAPIへのネットワークアクセスが
遮断されているため未検証。実運用環境で `pip install yfinance`
の上で使用すること。
"""

from __future__ import annotations

import pandas as pd

from finbot.data.base import DataSource


class YFinanceSource(DataSource):
    def __init__(self, tickers: dict[str, str], period: str = "10y"):
        """tickers: {資産名: Yahooティッカー} 例 {"EQ_US": "SPY", "BOND_GOV": "TLT"}"""
        self.tickers = tickers
        self.period = period

    def load(self) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance が必要です: pip install yfinance"
            ) from e

        raw = yf.download(
            list(self.tickers.values()), period=self.period, auto_adjust=True
        )["Close"]
        inv = {v: k for k, v in self.tickers.items()}
        prices = raw.rename(columns=inv)[list(self.tickers.keys())]
        return self.validate(pd.DataFrame(prices))
