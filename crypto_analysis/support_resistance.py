from typing import Dict, List, Optional, Tuple

import pandas as pd


def swing_highs_lows(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    highs = df["high"]
    lows = df["low"]
    swing_high = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    swing_low = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
    swing_high = swing_high.rolling(window=window, min_periods=1).max()
    swing_low = swing_low.rolling(window=window, min_periods=1).min()
    return pd.DataFrame({"swing_high": swing_high, "swing_low": swing_low})


def recent_levels(df: pd.DataFrame, lookback: int = 50) -> Tuple[Optional[float], Optional[float]]:
    if len(df) < 5:
        return None, None

    windowed = df.tail(lookback)
    recent_high = windowed["high"].max()
    recent_low = windowed["low"].min()
    return float(recent_low), float(recent_high)


def fibonacci_levels(low: float, high: float) -> Dict[str, float]:
    """Return common Fibonacci retracement/extension levels between a swing low/high."""

    diff = high - low
    levels = {
        "0.236": high - diff * 0.236,
        "0.382": high - diff * 0.382,
        "0.5": high - diff * 0.5,
        "0.618": high - diff * 0.618,
        "0.786": high - diff * 0.786,
        "1.272": high + diff * 0.272,
    }
    return levels


def pivot_points(df: pd.DataFrame, lookback: int = 2) -> Optional[Dict[str, float]]:
    """Classic pivot points using the previous candle (or previous day/period)."""

    if len(df) < lookback:
        return None
    prev = df.iloc[-lookback]
    high = float(prev["high"])
    low = float(prev["low"])
    close = float(prev["close"])
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return {"pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2}


def volume_profile_levels(df: pd.DataFrame, bins: int = 12, lookback: int = 200) -> Optional[Dict[str, float]]:
    """Approximate volume profile to find high/low volume nodes as support/resistance."""

    if len(df) < 10:
        return None
    windowed = df.tail(lookback)
    typical_price = (windowed["high"] + windowed["low"] + windowed["close"]) / 3
    min_price = typical_price.min()
    max_price = typical_price.max()
    if min_price == max_price:
        return None
    bucket = pd.cut(typical_price, bins=bins, labels=False, include_lowest=True)
    volume_by_bucket = windowed.groupby(bucket)["volume"].sum()
    if volume_by_bucket.empty:
        return None
    hvn_bucket = volume_by_bucket.idxmax()
    lvn_bucket = volume_by_bucket.idxmin()
    bin_size = (max_price - min_price) / bins
    hvn_price = float(min_price + bin_size * (hvn_bucket + 0.5))
    lvn_price = float(min_price + bin_size * (lvn_bucket + 0.5))
    return {"hvn": hvn_price, "lvn": lvn_price}


def detect_patterns(df: pd.DataFrame, lookback: int = 80) -> List[str]:
    """Lightweight pattern hints: double top/bottom and channel drift."""

    patterns: List[str] = []
    recent = df.tail(lookback)
    highs = recent["high"]
    lows = recent["low"]
    close = recent["close"]

    # Double top/bottom approximation
    top_level = highs.max()
    bottom_level = lows.min()
    last_close = close.iloc[-1]
    if ((highs > top_level * 0.995) & (highs < top_level * 1.005)).sum() >= 2 and last_close < top_level:
        patterns.append("可能的双顶/压力区")
    if ((lows > bottom_level * 0.995) & (lows < bottom_level * 1.005)).sum() >= 2 and last_close > bottom_level:
        patterns.append("可能的双底/支撑区")

    # Trend channel hint via linear regression slope
    if len(recent) >= 20:
        idx = pd.Series(range(len(recent)), index=recent.index)
        slope = ((idx - idx.mean()) * (close - close.mean())).sum() / ((idx - idx.mean()) ** 2).sum()
        if slope > 0:
            patterns.append("近段时间呈上行通道")
        elif slope < 0:
            patterns.append("近段时间呈下行通道")

    return patterns


