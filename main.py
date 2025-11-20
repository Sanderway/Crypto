import argparse
from typing import List

from crypto_analysis import Analyzer, BinanceClient, build_summary

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
DEFAULT_INTERVALS = ["1d", "4h", "1h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto multi-timeframe analyzer (Binance)")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS, help="Symbols to analyze, default USDT pairs")
    parser.add_argument("--intervals", nargs="*", default=DEFAULT_INTERVALS, help="Intervals, e.g. 1d 4h 1h")
    parser.add_argument("--limit", type=int, default=240, help="Number of candles to fetch per timeframe")
    return parser.parse_args()


def format_report(report) -> str:
    lines: List[str] = [f"=== {report.symbol} ==="]
    for tf in report.timeframes:
        lines.append(f"[{tf.interval}] 收盘价: {tf.close:.2f} | 趋势: {tf.trend}")
        lines.append(
            f"  EMA20/50/200: {tf.ema20:.2f} / {tf.ema50:.2f} / {tf.ema200:.2f} | MACD: {tf.macd:.4f} ({tf.macd_signal:.4f}) | RSI: {tf.rsi:.2f}"
        )
        if tf.support is not None and tf.resistance is not None:
            lines.append(f"  支撑: {tf.support:.2f} | 阻力: {tf.resistance:.2f}")
        if tf.fib_levels:
            fib_text = ", ".join([f"{k}:{v:.2f}" for k, v in tf.fib_levels.items()])
            lines.append(f"  斐波位: {fib_text}")
        lines.append(f"  建议: {tf.suggestion}")
        if tf.llm_note:
            lines.append(f"  GPT补充: {tf.llm_note}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    client = BinanceClient()
    analyzer = Analyzer(client)

    reports = [analyzer.analyze_symbol(symbol, args.intervals, limit=args.limit) for symbol in args.symbols]
    summary = build_summary(reports)

    for report in reports:
        print(format_report(report))
        print()

    print("汇总（可供 API/前端使用）:")
    print(summary)


if __name__ == "__main__":
    main()

