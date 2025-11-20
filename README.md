# Crypto Analysis Prototype

初版的加密货币多周期技术分析脚本，默认针对 BTC/ETH/BNB/SOL（USDT 现货对），从币安 REST API 获取 K 线数据并输出趋势、支撑/阻力与操作建议。

## 功能
- 支持 1d / 4h / 1h 多周期分析，可通过参数调整。
- 计算 EMA20/50/200、MACD、RSI。
- 依据 EMA/ MACD 判定趋势，结合近 50 根 K 线高低点推导支撑/阻力。
- 输出中文建议，附关键指标值，末尾提供结构化字典便于 API/前端复用。

## 运行
```bash
pip install -r requirements.txt
python main.py --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT --intervals 1d 4h 1h --limit 240
```

## 说明
- 数据来源：币安公开 REST `/api/v3/klines`。
- 该脚本仅供技术分析参考，不构成投资建议。

