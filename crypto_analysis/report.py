import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from . import indicators
from .binance_client import BinanceClient
from .support_resistance import fibonacci_levels, recent_levels


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
    support: Optional[float]
    resistance: Optional[float]
    fib_levels: Optional[Dict[str, float]]
    trend: str
    suggestion: str
    llm_note: Optional[str]


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
        fib_levels = fibonacci_levels(support, resistance) if support is not None and resistance is not None else None

        last_row = df.iloc[-1]
        price = float(last_row["close"])
        ema20_val = float(ema20.iloc[-1])
        ema50_val = float(ema50.iloc[-1])
        ema200_val = float(ema200.iloc[-1])
        macd_val = float(macd_df["macd"].iloc[-1])
        macd_signal_val = float(macd_df["signal"].iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])

        trend = self._determine_trend(price, ema50_val, ema200_val, macd_val)
        suggestion = self._make_suggestion(
            price,
            trend,
            support,
            resistance,
            rsi_val,
            macd_val,
            macd_signal_val,
            fib_levels,
        )
        llm_note = self._maybe_llm_comment(
            symbol=symbol,
            interval=interval,
            price=price,
            trend=trend,
            support=support,
            resistance=resistance,
            fib_levels=fib_levels,
            ema20=ema20_val,
            ema50=ema50_val,
            ema200=ema200_val,
            macd=macd_val,
            macd_signal=macd_signal_val,
            rsi=rsi_val,
        )

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
            fib_levels=fib_levels,
            trend=trend,
            suggestion=suggestion,
            llm_note=llm_note,
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
        self,
        price: float,
        trend: str,
        support: Optional[float],
        resistance: Optional[float],
        rsi_value: float,
        macd_value: float,
        macd_signal: float,
        fib_levels: Optional[Dict[str, float]],
    ) -> str:
        notes: List[str] = []

        if trend == "bullish":
            notes.append("总体偏多，关注回踩做多机会")
            if support is not None:
                notes.append(f"靠近支撑 {support:.2f} 可分批低吸，跌破则止损")
            if fib_levels is not None and "0.382" in fib_levels:
                notes.append(f"斐波 0.382 回撤位 {fib_levels['0.382']:.2f} 可作为二次补仓/防守")
            if resistance is not None:
                notes.append(f"首要目标/减仓位看阻力 {resistance:.2f}")
            if macd_value < macd_signal:
                notes.append("MACD 处于回落，追多需等待柱子重新翻红")
        elif trend == "bearish":
            notes.append("总体偏空，逢高寻找做空/减仓点")
            if resistance is not None:
                notes.append(f"靠近阻力 {resistance:.2f} 可尝试轻仓空，站上则止损")
            if fib_levels is not None and "0.618" in fib_levels:
                notes.append(f"斐波 0.618 回撤 {fib_levels['0.618']:.2f} 为反弹重要压制")
            if support is not None:
                notes.append(f"下方支撑 {support:.2f}，接近时不宜盲目追空")
            if rsi_value < 35:
                notes.append("RSI 已接近超卖，等待反弹承压再布局")
        else:
            notes.append("当前震荡，优先等待方向选择")
            if support is not None and resistance is not None:
                notes.append(f"区间参考 {support:.2f} - {resistance:.2f}，下沿试多，上沿试空")
            if fib_levels is not None and "0.5" in fib_levels:
                notes.append(f"0.5 回撤位 {fib_levels['0.5']:.2f} 可作为区间中轴观察")
            if rsi_value > 70:
                notes.append("RSI 高位，警惕上沿附近回落")
            elif rsi_value < 30:
                notes.append("RSI 低位，可关注下沿企稳后的反弹")

        return "；".join(notes)

    def _maybe_llm_comment(
        self,
        *,
        symbol: str,
        interval: str,
        price: float,
        trend: str,
        support: Optional[float],
        resistance: Optional[float],
        fib_levels: Optional[Dict[str, float]],
        ema20: float,
        ema50: float,
        ema200: float,
        macd: float,
        macd_signal: float,
        rsi: float,
    ) -> Optional[str]:
        """Optional GPT-based commentary if openai is installed and API key provided."""

        if importlib.util.find_spec("openai") is None:
            return None
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        import openai

        client = openai.OpenAI(api_key=api_key)
        fib_text = ", ".join([f"{k}:{v:.2f}" for k, v in (fib_levels or {}).items()]) or "无"
        payload: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "你是一名数字货币技术分析助手。请基于给定的指标，用中文给出更细的操作建议，"
                    "包含潜在入场、止损和目标位，不要夸大收益，强调风险。"  # 指令
                    f"\n币种: {symbol}, 周期: {interval}, 收盘: {price:.2f}, 趋势: {trend}, "
                    f"EMA20/50/200: {ema20:.2f}/{ema50:.2f}/{ema200:.2f}, "
                    f"MACD: {macd:.4f} (signal {macd_signal:.4f}), RSI: {rsi:.2f}, "
                    f"支撑: {support}, 阻力: {resistance}, 斐波位: {fib_text}"
                ),
            }
        ]
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=payload,
            temperature=0.4,
            max_tokens=300,
        )
        message = completion.choices[0].message.content
        return message.strip() if message else None


def build_summary(reports: List[SymbolReport]) -> Dict[str, Dict[str, Dict[str, Union[float, str, None]]]]:
    summary: Dict[str, Dict[str, Dict[str, Union[float, str, None, Dict[str, float]]]]] = {}
    for report in reports:
        symbol_entry: Dict[str, Dict[str, Union[float, str, None, Dict[str, float]]]] = {}
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
                "fib_levels": tf.fib_levels,
                "llm_note": tf.llm_note,
            }
        summary[report.symbol] = symbol_entry
    return summary


