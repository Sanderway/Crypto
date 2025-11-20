from typing import List, Optional, Union

import pandas as pd
import requests

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
TICKER_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"


INTERVALS = {
    "1d": "1d",
    "4h": "4h",
    "1h": "1h",
    "15m": "15m",
}


class BinanceClient:
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def fetch_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        if interval not in INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")

        params = {"symbol": symbol.upper(), "interval": INTERVALS[interval], "limit": limit}
        response = self.session.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data: List[List[Union[str, float, int]]] = response.json()

        df = pd.DataFrame(
            data,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # Binance futures klines include the currently-forming candle; drop any kline whose
        # close_time is still in the future to ensure we only analyze closed candles.
        now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
        df = df[df["close_time"] <= now_utc]
        if df.empty:
            raise ValueError("No closed klines returned; please retry with a larger limit or later time.")

        return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]

    def fetch_latest_close(self, symbol: str, interval: str) -> float:
        df = self.fetch_klines(symbol, interval, limit=2)
        return float(df["close"].iloc[-1])

    def fetch_mark_price(self, symbol: str) -> float:
        """Fetch the current mark price for a futures symbol (near real-time)."""

        params = {"symbol": symbol.upper()}
        response = self.session.get(TICKER_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data["markPrice"])


