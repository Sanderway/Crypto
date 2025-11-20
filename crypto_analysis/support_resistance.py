from typing import Dict, Optional, Tuple

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


