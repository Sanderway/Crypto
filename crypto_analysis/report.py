import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from . import indicators
from .binance_client import BinanceClient
from .support_resistance import (
    detect_patterns,
    fibonacci_levels,
    pivot_points,
    recent_levels,
    volume_profile_levels,
)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "__float__"):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


@dataclass
class TimeframeReport:
    interval: str
    close: float
    close_time: pd.Timestamp
    mark_price: Optional[float]
    ema20: float
    ema50: float
    ema200: float
    sma20: Optional[float]
    sma50: Optional[float]
    macd: float
    macd_signal: float
    macd_histogram: float
    rsi: float
    kdj_k: Optional[float]
    kdj_d: Optional[float]
    kdj_j: Optional[float]
    boll_mid: Optional[float]
    boll_upper: Optional[float]
    boll_lower: Optional[float]
    atr: Optional[float]
    vma20: Optional[float]
    vma50: Optional[float]
    support: Optional[float]
    resistance: Optional[float]
    fib_levels: Optional[Dict[str, float]]
    pivot_levels: Optional[Dict[str, float]]
    volume_profile: Optional[Dict[str, float]]
    patterns: List[str]
    trend: str
    suggestion: str
    llm_note: Optional[str]


@dataclass
class SymbolReport:
    symbol: str
    timeframes: List[TimeframeReport]
    mark_price: Optional[float]
    llm_prompt: str


class Analyzer:
    def __init__(self, client: BinanceClient) -> None:
        self.client = client

    def analyze_symbol(self, symbol: str, intervals: List[str], limit: int = 240) -> SymbolReport:
        tf_reports: List[TimeframeReport] = []
        mark_price: Optional[float] = None
        try:
            mark_price = self.client.fetch_mark_price(symbol)
        except Exception:
            mark_price = None

        for interval in intervals:
            df = self.client.fetch_klines(symbol, interval, limit=limit)
            tf_reports.append(self._analyze_timeframe(df, symbol, interval, mark_price=mark_price))

        prompt = self._build_llm_prompt(symbol, tf_reports)
        return SymbolReport(symbol=symbol, timeframes=tf_reports, mark_price=mark_price, llm_prompt=prompt)

    def _analyze_timeframe(self, df: pd.DataFrame, symbol: str, interval: str, mark_price: Optional[float]) -> TimeframeReport:
        closes = df["close"]
        ema20 = indicators.ema(closes, 20)
        ema50 = indicators.ema(closes, 50)
        ema200 = indicators.ema(closes, 200)
        sma20 = indicators.sma(closes, 20)
        sma50 = indicators.sma(closes, 50)
        macd_df = indicators.macd(closes)
        rsi_series = indicators.rsi(closes)
        boll = indicators.bollinger_bands(closes)
        kdj_df = indicators.kdj(df)
        atr_series = indicators.atr(df)
        vol_ma_df = indicators.volume_ma(df["volume"])
        support, resistance = recent_levels(df)
        fib_levels = fibonacci_levels(support, resistance) if support is not None and resistance is not None else None
        pivot_level = pivot_points(df)
        volume_profile = volume_profile_levels(df)
        patterns = detect_patterns(df)

        last_row = df.iloc[-1]
        price = float(last_row["close"])
        close_time = pd.to_datetime(last_row["close_time"])
        ema20_val = float(ema20.iloc[-1])
        ema50_val = float(ema50.iloc[-1])
        ema200_val = float(ema200.iloc[-1])
        sma20_val = _safe_float(sma20.iloc[-1])
        sma50_val = _safe_float(sma50.iloc[-1])
        macd_val = float(macd_df["macd"].iloc[-1])
        macd_signal_val = float(macd_df["signal"].iloc[-1])
        macd_hist_val = float(macd_df["histogram"].iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])
        boll_mid = _safe_float(boll["mid"].iloc[-1])
        boll_upper = _safe_float(boll["upper"].iloc[-1])
        boll_lower = _safe_float(boll["lower"].iloc[-1])
        k_val = _safe_float(kdj_df["k"].iloc[-1])
        d_val = _safe_float(kdj_df["d"].iloc[-1])
        j_val = _safe_float(kdj_df["j"].iloc[-1])
        atr_val = _safe_float(atr_series.iloc[-1])
        vma20_val = _safe_float(vol_ma_df["vma20"].iloc[-1]) if "vma20" in vol_ma_df else None
        vma50_val = _safe_float(vol_ma_df["vma50"].iloc[-1]) if "vma50" in vol_ma_df else None

        trend = self._determine_trend(price, ema20_val, ema50_val, ema200_val, macd_val, rsi_val)
        suggestion = self._make_suggestion(
            price=price,
            interval=interval,
            trend=trend,
            support=support,
            resistance=resistance,
            pivot=pivot_level,
            fib_levels=fib_levels,
            volume_profile=volume_profile,
            rsi_value=rsi_val,
            macd_value=macd_val,
            macd_signal=macd_signal_val,
            macd_hist=macd_hist_val,
            boll_mid=boll_mid,
            boll_upper=boll_upper,
            boll_lower=boll_lower,
            k_val=k_val,
            d_val=d_val,
            j_val=j_val,
            atr=atr_val,
            patterns=patterns,
        )
        llm_note = self._maybe_llm_comment(
            symbol=symbol,
            interval=interval,
            price=price,
            close_time=close_time,
            trend=trend,
            support=support,
            resistance=resistance,
            fib_levels=fib_levels,
            pivot_levels=pivot_level,
            volume_profile=volume_profile,
            mark_price=mark_price,
            ema20=ema20_val,
            ema50=ema50_val,
            ema200=ema200_val,
            sma20=sma20_val,
            sma50=sma50_val,
            macd=macd_val,
            macd_signal=macd_signal_val,
            macd_hist=macd_hist_val,
            rsi=rsi_val,
            k=k_val,
            d=d_val,
            j=j_val,
            atr=atr_val,
            boll_mid=boll_mid,
            boll_upper=boll_upper,
            boll_lower=boll_lower,
        )

        return TimeframeReport(
            interval=interval,
            close=price,
            close_time=close_time,
            mark_price=mark_price,
            ema20=ema20_val,
            ema50=ema50_val,
            ema200=ema200_val,
            sma20=sma20_val,
            sma50=sma50_val,
            macd=macd_val,
            macd_signal=macd_signal_val,
            macd_histogram=macd_hist_val,
            rsi=rsi_val,
            kdj_k=k_val,
            kdj_d=d_val,
            kdj_j=j_val,
            boll_mid=boll_mid,
            boll_upper=boll_upper,
            boll_lower=boll_lower,
            atr=atr_val,
            vma20=vma20_val,
            vma50=vma50_val,
            support=support,
            resistance=resistance,
            fib_levels=fib_levels,
            pivot_levels=pivot_level,
            volume_profile=volume_profile,
            patterns=patterns,
            trend=trend,
            suggestion=suggestion,
            llm_note=llm_note,
        )

    def _determine_trend(self, price: float, ema20: float, ema50: float, ema200: float, macd_value: float, rsi_value: float) -> str:
        bullish_stack = price > ema200 and ema20 > ema50 > ema200 and macd_value > 0 and rsi_value > 55
        bearish_stack = price < ema200 and ema20 < ema50 < ema200 and macd_value < 0 and rsi_value < 45
        if bullish_stack:
            return "bullish"
        if bearish_stack:
            return "bearish"
        return "sideways"

    def _make_suggestion(
        self,
        *,
        price: float,
        interval: str,
        trend: str,
        support: Optional[float],
        resistance: Optional[float],
        pivot: Optional[Dict[str, float]],
        fib_levels: Optional[Dict[str, float]],
        volume_profile: Optional[Dict[str, float]],
        rsi_value: float,
        macd_value: float,
        macd_signal: float,
        macd_hist: float,
        boll_mid: Optional[float],
        boll_upper: Optional[float],
        boll_lower: Optional[float],
        k_val: Optional[float],
        d_val: Optional[float],
        j_val: Optional[float],
        atr: Optional[float],
        patterns: List[str],
    ) -> str:
        notes: List[str] = []

        # Trend context
        if trend == "bullish":
            notes.append("偏多，关注回踩多的机会")
        elif trend == "bearish":
            notes.append("偏空，逢高做空或减仓")
        else:
            notes.append("震荡，等待方向或高低抛吸")

        # Key price references
        if support is not None and resistance is not None:
            notes.append(f"区间参考 {support:.2f}-{resistance:.2f}")
        if pivot:
            notes.append(f"Pivot {pivot['pivot']:.2f}，R1 {pivot['r1']:.2f} / S1 {pivot['s1']:.2f}")
        if fib_levels:
            fib_382 = fib_levels.get("0.382")
            fib_618 = fib_levels.get("0.618")
            fib_texts: List[str] = []
            if fib_382 is not None:
                fib_texts.append(f"0.382:{fib_382:.2f}")
            if fib_618 is not None:
                fib_texts.append(f"0.618:{fib_618:.2f}")
            if fib_texts:
                notes.append("斐波 " + " / ".join(fib_texts))
        if volume_profile:
            if "hvn" in volume_profile:
                notes.append(f"成交密集区(HVN) {volume_profile['hvn']:.2f}")
            if "lvn" in volume_profile:
                notes.append(f"低成交密集区(LVN) {volume_profile['lvn']:.2f}，易成突破点")

        # Momentum & oscillators
        if macd_value > macd_signal and macd_hist > 0:
            notes.append("MACD 多头动能增强")
        elif macd_value < macd_signal and macd_hist < 0:
            notes.append("MACD 空头动能延续")
        else:
            notes.append("MACD 动能钝化，等待再次放量")

        if rsi_value >= 70:
            notes.append("RSI 高位，谨防回落")
        elif rsi_value <= 30:
            notes.append("RSI 低位，可等待止跌反弹")

        if k_val is not None and d_val is not None:
            if k_val > d_val and (j_val or 0) > 80:
                notes.append("KDJ 高位金叉，追多需谨慎")
            elif k_val < d_val and (j_val or 0) < 20:
                notes.append("KDJ 低位死叉，做空注意止盈")

        if boll_upper is not None and price >= boll_upper:
            notes.append("价格触及布林上轨，防回踩")
        if boll_lower is not None and price <= boll_lower:
            notes.append("价格触及布林下轨，关注反弹")

        # ATR based stop suggestion
        if atr is not None:
            stop_range = atr * 1.2
            notes.append(f"ATR 防守约 {stop_range:.2f}，可用于止损距离")

        # Pattern hints
        if patterns:
            notes.append("形态: " + "，".join(patterns))

        # Tactical bias per interval
        if interval in ("1h", "15m"):
            if trend == "bullish" and support is not None:
                notes.append("短线回踩支撑或0.382附近尝试顺势做多")
            elif trend == "bearish" and resistance is not None:
                notes.append("短线反弹至阻力/0.618附近尝试顺势做空")
            else:
                notes.append("短线以区间高抛低吸为主")
        else:
            notes.append("此周期偏向趋势框架，具体执行参考1H/15M")

        return "；".join(notes)

    def _maybe_llm_comment(
        self,
        *,
        symbol: str,
        interval: str,
        price: float,
        close_time: pd.Timestamp,
        trend: str,
        support: Optional[float],
        resistance: Optional[float],
        fib_levels: Optional[Dict[str, float]],
        pivot_levels: Optional[Dict[str, float]],
        volume_profile: Optional[Dict[str, float]],
        mark_price: Optional[float],
        ema20: float,
        ema50: float,
        ema200: float,
        sma20: Optional[float],
        sma50: Optional[float],
        macd: float,
        macd_signal: float,
        macd_hist: float,
        rsi: float,
        k: Optional[float],
        d: Optional[float],
        j: Optional[float],
        atr: Optional[float],
        boll_mid: Optional[float],
        boll_upper: Optional[float],
        boll_lower: Optional[float],
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
        pivot_text = ", ".join([f"{k}:{v:.2f}" for k, v in (pivot_levels or {}).items()]) or "无"
        vp_text = ", ".join([f"{k}:{v:.2f}" for k, v in (volume_profile or {}).items()]) or "无"
        payload: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "你是一名数字货币技术分析助手。请基于给定的指标，用中文给出更细的操作建议，"
                    "包含潜在入场、止损和目标位，不要夸大收益，强调风险。"  # 指令
                    f"\n币种: {symbol}, 周期: {interval}, 收盘: {price:.2f} (K线截至UTC {close_time}), 标记价: {mark_price}, 趋势: {trend}"
                    f"\nEMA20/50/200: {ema20:.2f}/{ema50:.2f}/{ema200:.2f}, SMA20/50: {sma20} / {sma50}"
                    f"\nMACD: {macd:.4f} (signal {macd_signal:.4f}, hist {macd_hist:.4f}), RSI: {rsi:.2f}"
                    f"\nKDJ: {k} / {d} / {j}, ATR: {atr}, Boll 中轨: {boll_mid}, 上轨: {boll_upper}, 下轨: {boll_lower}"
                    f"\n支撑: {support}, 阻力: {resistance}, Pivot: {pivot_text}, 斐波位: {fib_text}, VolumeProfile: {vp_text}"
                ),
            }
        ]
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=payload,
            temperature=0.4,
            max_tokens=320,
        )
        message = completion.choices[0].message.content
        return message.strip() if message else None

    def _build_llm_prompt(self, symbol: str, reports: List[TimeframeReport]) -> str:
        lines: List[str] = [
            "你是一名数字货币分析师，请根据以下多周期指标和初步建议，输出更细的交易计划（入场条件、止损、分批减仓/目标位），并强调风险与仓位控制。",
            "禁止提供投资承诺，保持客观。",
            f"标的: {symbol} (币安USDT永续合约)",
        ]
        for tf in reports:
            fib_text = ", ".join([f"{k}:{v:.2f}" for k, v in (tf.fib_levels or {}).items()]) or "无"
            pivot_text = ", ".join([f"{k}:{v:.2f}" for k, v in (tf.pivot_levels or {}).items()]) or "无"
            vp_text = ", ".join([f"{k}:{v:.2f}" for k, v in (tf.volume_profile or {}).items()]) or "无"
            patterns = ",".join(tf.patterns) if tf.patterns else "无"
            lines.append(
                (
                    f"\n[周期 {tf.interval}] 收盘:{tf.close:.2f} (K线UTC:{tf.close_time}) 标记价:{tf.mark_price} 趋势:{tf.trend}"
                    f" | EMA20/50/200:{tf.ema20:.2f}/{tf.ema50:.2f}/{tf.ema200:.2f}"
                    f" | SMA20/50:{tf.sma20} / {tf.sma50}"
                    f" | MACD:{tf.macd:.4f}/{tf.macd_signal:.4f}/{tf.macd_histogram:.4f}"
                    f" | RSI:{tf.rsi:.2f} | KDJ:{tf.kdj_k}/{tf.kdj_d}/{tf.kdj_j}"
                    f" | Boll:{tf.boll_lower}/{tf.boll_mid}/{tf.boll_upper} | ATR:{tf.atr}"
                    f" | 支撑/阻力:{tf.support}/{tf.resistance} | Pivot:{pivot_text}"
                    f" | 斐波:{fib_text} | 成交量均线:{tf.vma20}/{tf.vma50} | VP:{vp_text}"
                    f" | 形态:{patterns}"
                    f" | 初步建议:{tf.suggestion}"
                )
            )
        return "\n".join(lines)


def build_summary(reports: List[SymbolReport]) -> Dict[str, Dict[str, Dict[str, Union[float, str, None, Dict[str, float], List[str]]]]]:
    summary: Dict[str, Dict[str, Dict[str, Union[float, str, None, Dict[str, float], List[str]]]]] = {}
    for report in reports:
        symbol_entry: Dict[str, Dict[str, Union[float, str, None, Dict[str, float], List[str]]]] = {}
        for tf in report.timeframes:
            symbol_entry[tf.interval] = {
                "close": tf.close,
                "close_time": tf.close_time.isoformat(),
                "mark_price": report.mark_price,
                "ema20": tf.ema20,
                "ema50": tf.ema50,
                "ema200": tf.ema200,
                "sma20": tf.sma20,
                "sma50": tf.sma50,
                "macd": tf.macd,
                "macd_signal": tf.macd_signal,
                "macd_histogram": tf.macd_histogram,
                "rsi": tf.rsi,
                "kdj_k": tf.kdj_k,
                "kdj_d": tf.kdj_d,
                "kdj_j": tf.kdj_j,
                "boll_mid": tf.boll_mid,
                "boll_upper": tf.boll_upper,
                "boll_lower": tf.boll_lower,
                "atr": tf.atr,
                "vma20": tf.vma20,
                "vma50": tf.vma50,
                "support": tf.support,
                "resistance": tf.resistance,
                "trend": tf.trend,
                "suggestion": tf.suggestion,
                "fib_levels": tf.fib_levels,
                "pivot_levels": tf.pivot_levels,
                "volume_profile": tf.volume_profile,
                "patterns": tf.patterns,
                "llm_note": tf.llm_note,
            }
        symbol_entry["llm_prompt"] = {"prompt": report.llm_prompt}
        summary[report.symbol] = symbol_entry
    return summary
