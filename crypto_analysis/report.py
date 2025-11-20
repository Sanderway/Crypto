from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from . import indicators
from .binance_client import BinanceClient
from .support_resistance import recent_levels


@dataclass
class TimeframeReport:
    interval: str
    close: float
    ema20: float
    ema50: float
    ema200: float
    macd: float
    macd_signal: float
    rsi: float
    support: float | None
    resistance: float | None
    trend: str
    suggestion: str


@dataclass
class SymbolReport:
    symbol: str
    timeframes: List[TimeframeReport]


class Analyzer:
    def __init__(self, client: BinanceClient) -> None:
        self.client = client

    def analyze_symbol(self, symbol: str, intervals: List[str], limit: int = 240) -> SymbolReport:
        tf_reports: List[TimeframeReport] = []
        for interval in intervals:
            df = self.client.fetch_klines(symbol, interval, limit=limit)
            tf_reports.append(self._analyze_timeframe(df, symbol, interval))
        return SymbolReport(symbol=symbol, timeframes=tf_reports)

    def _analyze_timeframe(self, df: pd.DataFrame, symbol: str, interval: str) -> TimeframeReport:
        closes = df["close"]
        ema20 = indicators.ema(closes, 20)
        ema50 = indicators.ema(closes, 50)
        ema200 = indicators.ema(closes, 200)
        macd_df = indicators.macd(closes)
        rsi_series = indicators.rsi(closes)
        support, resistance = recent_levels(df)

        last_row = df.iloc[-1]
        price = float(last_row["close"])
        ema20_val = float(ema20.iloc[-1])
        ema50_val = float(ema50.iloc[-1])
        ema200_val = float(ema200.iloc[-1])
        macd_val = float(macd_df["macd"].iloc[-1])
        macd_signal_val = float(macd_df["signal"].iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])

        trend = self._determine_trend(price, ema50_val, ema200_val, macd_val)
        suggestion = self._make_suggestion(price, trend, support, resistance, rsi_val)

        return TimeframeReport(
            interval=interval,
            close=price,
            ema20=ema20_val,
            ema50=ema50_val,
            ema200=ema200_val,
            macd=macd_val,
            macd_signal=macd_signal_val,
            rsi=rsi_val,
            support=support,
            resistance=resistance,
            trend=trend,
            suggestion=suggestion,
        )

    def _determine_trend(self, price: float, ema50: float, ema200: float, macd_value: float) -> str:
        bullish = price > ema200 and ema50 > ema200 and macd_value > 0
        bearish = price < ema200 and ema50 < ema200 and macd_value < 0
        if bullish:
            return "bullish"
        if bearish:
            return "bearish"
        return "sideways"

    def _make_suggestion(
        self, price: float, trend: str, support: float | None, resistance: float | None, rsi_value: float
    ) -> str:
        if trend == "bullish" and support:
            if price <= support * 1.01:
                return "多头趋势，靠近支撑，可考虑分批做多/逢低买入"
            if resistance and price >= resistance * 0.99:
                return "多头趋势但临近阻力，注意减仓或等待突破确认"
            return "多头趋势，持有或逢回调加仓"
        if trend == "bearish" and resistance:
            if price >= resistance * 0.99:
                return "空头趋势，靠近阻力，可考虑逢高做空或减仓"
            if support and price <= support * 1.01:
                return "空头趋势但接近支撑，谨慎追空，关注反弹"
            return "空头趋势，反弹承压时考虑减仓/做空"

        if rsi_value < 35 and support:
            return "震荡区间但RSI低位，若守住支撑可尝试低吸"
        if rsi_value > 65 and resistance:
            return "震荡区间且RSI偏高，注意阻力位附近的回落风险"
        return "走势震荡，等待关键位突破"


def build_summary(reports: List[SymbolReport]) -> Dict[str, Dict[str, Dict[str, float | str | None]]]:
    summary: Dict[str, Dict[str, Dict[str, float | str | None]]] = {}
    for report in reports:
        symbol_entry: Dict[str, Dict[str, float | str | None]] = {}
        for tf in report.timeframes:
            symbol_entry[tf.interval] = {
                "close": tf.close,
                "ema20": tf.ema20,
                "ema50": tf.ema50,
                "ema200": tf.ema200,
                "macd": tf.macd,
                "macd_signal": tf.macd_signal,
                "rsi": tf.rsi,
                "support": tf.support,
                "resistance": tf.resistance,
                "trend": tf.trend,
                "suggestion": tf.suggestion,
            }
        summary[report.symbol] = symbol_entry
    return summary


