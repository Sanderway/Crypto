# Crypto Analysis Prototype

初版的加密货币多周期技术分析脚本，默认针对 BTC/ETH/BNB/SOL（USDT 现货对），从币安 REST API 获取 K 线数据并输出趋势、支撑/阻力、斐波位与操作建议，可选调用 GPT 生成更细化的交易提示。

## 功能
- 支持 1d / 4h / 1h 多周期分析，可通过参数调整。
- 计算 EMA20/50/200、MACD、RSI。
- 依据 EMA/ MACD 判定趋势，结合近 50 根 K 线高低点推导支撑/阻力并输出常见斐波回撤/扩展位（0.236/0.382/0.5/0.618/0.786/1.272）。
- 输出更细的中文建议（包含入场/止损/目标位逻辑），末尾提供结构化字典便于 API/前端复用。
- 如安装 `openai` 且设置 `OPENAI_API_KEY`，会附带 GPT 补充说明；否则跳过。

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
按下例命令即可启动分析（需要外网访问币安）：
```bash
python main.py --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT --intervals 1d 4h 1h --limit 240
```

参数说明：
- `--symbols`：币种列表（默认为 BTC/ETH/BNB/SOL USDT 现货对）。
- `--intervals`：时间周期，可混合 `1d`、`4h`、`1h`、`15m` 等。
- `--limit`：每个周期抓取的 K 线数量（默认 240 根）。

运行后终端会输出每个币种的多周期指标、支撑/阻力与操作建议，并附带一个结构化汇总字典便于后续 API/前端使用。

## 说明
- 数据来源：币安公开 REST `/api/v3/klines`。
- 该脚本仅供技术分析参考，不构成投资建议。

