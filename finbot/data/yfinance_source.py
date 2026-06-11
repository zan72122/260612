"""Yahoo Finance アダプタ(実運用環境用)。

この開発環境からは市場データAPIへのネットワークアクセスが
遮断されているため未検証。実運用環境で `pip install yfinance`
の上で使用すること。
"""

from __future__ import annotations

import pandas as pd

from finbot.data.base import DataSource, OHLCFrames


class YFinanceSource(DataSource):
    def __init__(self, tickers: dict[str, str], period: str = "10y"):
        """tickers: {資産名: Yahooティッカー} 例 {"EQ_US": "SPY", "BOND_GOV": "TLT"}"""
        self.tickers = tickers
        self.period = period

    def _download(self) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance が必要です: pip install yfinance"
            ) from e

        return yf.download(
            list(self.tickers.values()), period=self.period, auto_adjust=True
        )

    def _rename(self, raw: pd.DataFrame) -> pd.DataFrame:
        inv = {v: k for k, v in self.tickers.items()}
        return pd.DataFrame(raw.rename(columns=inv)[list(self.tickers.keys())])

    def load(self) -> pd.DataFrame:
        return self.validate(self._rename(self._download()["Close"]))

    def load_ohlc(self) -> OHLCFrames:
        raw = self._download()
        close = self.validate(self._rename(raw["Close"]))
        ohlc = OHLCFrames(
            open=self._rename(raw["Open"]),
            high=self._rename(raw["High"]),
            low=self._rename(raw["Low"]),
            close=close,
        )
        return ohlc.aligned_to(close)
