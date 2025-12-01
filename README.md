# Crypto Analysis Prototype

多周期技术分析脚本，默认针对 BTC/ETH（币安 USDT 永续），从币安合约 REST API 获取 K 线数据并输出趋势、支撑/阻力、斐波位等，可选调用 GPT 生成更细化的交易提示。

## 功能
- 支持 1d / 4h / 1h / 15m 多周期分析（1d/4h 用于大势研判，1h/15m 给出执行建议）。
- 指标：EMA/SMA(20/50/200)、MACD、RSI、KDJ、布林带、ATR、成交量均线；支阻：近高低点、Pivot、斐波回撤/扩展、成交量分布近似（HVN/LVN）。
- 自动输出形态提示（双顶/双底、通道倾向），并据此生成更细的中文建议。
- 终端展示时附带两段 prompt（只保留高价值指标以节省 Token）：
  - `llm_prompt`：可投喂 GPT（openai 风格）。
  - `ark_prompt`：可直接投喂火山引擎 Ark 模型（JSON 输出，要求至少给出 1 条 actions，不允许空数组）。
- 末尾提供结构化汇总字典（包含指标、支阻位、形态、prompt 等）便于 API/前端复用。
- 支持近似实时轮询（`--watch`），可选将 `ark_prompt` 发送给 Ark 模型并把解析出的操作建议推送到 Telegram。

## 运行
### 1) 环境准备
- Python 3.9+，可选创建虚拟环境：
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```
  如需 GPT 补充说明，请确保网络可访问 OpenAI，并在环境中设置 `OPENAI_API_KEY`。

### 2) 执行脚本
按下例命令即可启动分析（需要外网访问币安 Futures）：
```bash
python main.py --symbols BTCUSDT ETHUSDT --intervals 1d 4h 1h 15m --limit 240
```

如需低频轮询（建议 1h/4h 节奏）并把分析交给 Ark 模型 + Telegram 推送：

```bash
python main.py --watch --refresh-seconds 3600 \
  --ark-api-key $ARK_API_KEY \
  --telegram-token $TELEGRAM_BOT_TOKEN \
  --telegram-chat-id $TELEGRAM_CHAT_ID
```

参数说明：
- `--symbols`：币种列表（默认为 BTCUSDT、ETHUSDT 永续）。
- `--intervals`：时间周期，可混合 `1d`、`4h`、`1h`、`15m` 等。
- `--limit`：每个周期抓取的 K 线数量（默认 240 根）。
- `--watch` 与 `--refresh-seconds`：开启轮询与轮询周期（秒），建议设置为 3600（1h）或 14400（4h）以匹配中长线节奏。内部会强制最少 900 秒，避免误设高频消耗 Token。
- `--ark-*`：指定 Ark API Key / 模型名 / 生成长度与 reasoning_effort，用于把 `ark_prompt` 投喂给 Ark。
- `--telegram-token` / `--telegram-chat-id`：提供后会把 Ark 返回的 JSON 操作建议转换成报警文本并推送 Telegram。

运行后终端会输出每个币种的多周期指标、支撑/阻力与操作建议，并附带：
- 可直接投喂 GPT 的 prompt（仅保留核心指标与初步建议，减少 Token 占用）。
- 一个结构化汇总字典，便于 API/前端使用。

## 说明
- 数据来源：币安合约 REST `/fapi/v1/klines`（USDT 永续），抓取后会过滤掉尚未收盘的当前 K 线，只保留已收盘的数据进行分析。
- 收盘价来自“最近一根已收盘 K 线”，各周期的 `close_time (UTC)` 会在输出中体现；终端输出也会显示本地时区时间，便于核对当下时刻。
- 若需要接近实时价格，可参考输出顶部的“标记价”，该值来自 `/fapi/v1/premiumIndex`，并非持久订阅；如需自动刷新可开启 `--watch`。
- 当前实现不包含 WebSocket 推送，如需毫秒级行情可扩展 WebSocket 监听或自行传入最新成交价进行重算。
- 该脚本仅供技术分析参考，不构成投资建议。
