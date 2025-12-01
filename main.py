import argparse
import os
import sys
import time
from typing import List

from crypto_analysis import Analyzer, BinanceClient, build_summary
from crypto_analysis.notifier import call_ark, format_actions_for_alert, parse_actions, send_telegram_alert

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_INTERVALS = ["1d", "4h", "1h", "15m"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto multi-timeframe analyzer (Binance)")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS, help="Symbols to analyze, default USDT pairs")
    parser.add_argument("--intervals", nargs="*", default=DEFAULT_INTERVALS, help="Intervals, e.g. 1d 4h 1h")
    parser.add_argument("--limit", type=int, default=240, help="Number of candles to fetch per timeframe")
    parser.add_argument("--watch", action="store_true", help="Enable near-real-time polling loop")
    parser.add_argument("--refresh-seconds", type=int, default=90, help="Polling interval when --watch is enabled")
    parser.add_argument("--ark-api-key", default=os.environ.get("ARK_API_KEY"), help="Ark API key (or set ARK_API_KEY)")
    parser.add_argument("--ark-model", default="doubao-seed-1-6-251015", help="Ark model name")
    parser.add_argument("--ark-max-tokens", type=int, default=2048, help="Max completion tokens for Ark")
    parser.add_argument(
        "--ark-reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort for Ark completions",
    )
    parser.add_argument("--telegram-token", default=os.environ.get("TELEGRAM_BOT_TOKEN"), help="Telegram bot token")
    parser.add_argument("--telegram-chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"), help="Telegram chat id")
    return parser.parse_args()


def format_report(report) -> str:
    lines: List[str] = [f"=== {report.symbol} ==="]
    if report.mark_price is not None:
        lines.append(f"最新标记价(实时): {report.mark_price:.4f}")
    for tf in report.timeframes:
        lines.append(
            f"[{tf.interval}] 收盘价: {tf.close:.2f} | K线截止(UTC): {tf.close_time} | 本地: {tf.close_time_local} | 趋势: {tf.trend}"
        )
        lines.append(
            f"  EMA20/50/200: {tf.ema20:.2f} / {tf.ema50:.2f} / {tf.ema200:.2f} | SMA20/50: {tf.sma20} / {tf.sma50}"
        )
        lines.append(
            f"  MACD: {tf.macd:.4f} (signal {tf.macd_signal:.4f}) hist {tf.macd_histogram:.4f} | RSI: {tf.rsi:.2f}"
        )
        lines.append(
            f"  KDJ: {tf.kdj_k} / {tf.kdj_d} / {tf.kdj_j} | Boll下/中/上: {tf.boll_lower} / {tf.boll_mid} / {tf.boll_upper} | ATR: {tf.atr}"
        )
        lines.append(f"  成交量均线: vma20={tf.vma20}, vma50={tf.vma50}")
        if tf.support is not None and tf.resistance is not None:
            lines.append(f"  支撑: {tf.support:.2f} | 阻力: {tf.resistance:.2f}")
        if tf.fib_levels:
            fib_text = ", ".join([f"{k}:{v:.2f}" for k, v in tf.fib_levels.items()])
            lines.append(f"  斐波位: {fib_text}")
        if tf.pivot_levels:
            pivot_text = ", ".join([f"{k}:{v:.2f}" for k, v in tf.pivot_levels.items()])
            lines.append(f"  Pivot: {pivot_text}")
        if tf.volume_profile:
            vp_text = ", ".join([f"{k}:{v:.2f}" for k, v in tf.volume_profile.items()])
            lines.append(f"  成交密集区: {vp_text}")
        if tf.patterns:
            lines.append("  形态: " + "，".join(tf.patterns))
        lines.append(f"  建议: {tf.suggestion}")
        if tf.llm_note:
            lines.append(f"  GPT补充: {tf.llm_note}")
    lines.append("---")
    lines.append("可直接投喂给 GPT 的分析提示:")
    lines.append(report.llm_prompt)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    client = BinanceClient()
    analyzer = Analyzer(client)

    iteration = 0
    while True:
        iteration += 1
        try:
            reports = [analyzer.analyze_symbol(symbol, args.intervals, limit=args.limit) for symbol in args.symbols]
        except Exception as exc:
            print(f"分析失败: {exc}", file=sys.stderr)
            if not args.watch:
                raise
            time.sleep(args.refresh_seconds)
            continue

        summary = build_summary(reports)

        print(f"=== 轮询第 {iteration} 轮，UTC 时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} ===")
        for report in reports:
            print(format_report(report))
            if args.ark_api_key:
                try:
                    response_text = call_ark(
                        prompt=report.ark_prompt,
                        api_key=args.ark_api_key,
                        model=args.ark_model,
                        max_tokens=args.ark_max_tokens,
                        reasoning_effort=args.ark_reasoning_effort,
                    )
                    actions = parse_actions(response_text)
                    if actions:
                        alert_text = format_actions_for_alert(report.symbol, actions)
                        print("AI操作建议(JSON):", actions)
                        if args.telegram_token and args.telegram_chat_id and alert_text:
                            send_telegram_alert(args.telegram_token, args.telegram_chat_id, alert_text)
                            print("已发送 Telegram 警报")
                    else:
                        print("Ark 返回空/无效的 actions")
                except Exception as exc:  # pragma: no cover - network errors
                    print(f"Ark 调用失败: {exc}", file=sys.stderr)
            print()

        print("汇总（可供 API/前端使用）:")
        print(summary)

        if not args.watch:
            break
        time.sleep(args.refresh_seconds)


if __name__ == "__main__":
    main()

