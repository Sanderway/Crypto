import json
from typing import Any, Dict, List

import requests

ARK_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"


def build_ark_messages(prompt: str) -> List[Dict[str, Any]]:
    system_text = (
        "你是量化交易与风控助手，仅输出 JSON。"
        "JSON 字段: actions 数组，每项包含 timeframe, bias(多/空/观望), entry, stop, targets 数组, confidence(0-100), note。"
        "必须至少输出 1 条 actions；若无信号，也需给出观望方案并标注关键支撑/阻力。"
        "禁止输出除 JSON 外的文字。"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def call_ark(
    *,
    prompt: str,
    api_key: str,
    model: str = "doubao-seed-1-6-251015",
    max_tokens: int = 2048,
    reasoning_effort: str = "medium",
) -> str:
    payload = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "messages": build_ark_messages(prompt),
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    response = requests.post(ARK_URL, headers=headers, json=payload, timeout=20)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Empty response from Ark model")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(texts)
    return ""


def parse_actions(response_text: str) -> List[Dict[str, Any]]:
    try:
        return json.loads(response_text).get("actions", [])  # type: ignore[arg-type]
    except json.JSONDecodeError:
        pass
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(response_text[start : end + 1]).get("actions", [])  # type: ignore[arg-type]
        except Exception:
            return []
    return []


def format_actions_for_alert(symbol: str, actions: List[Dict[str, Any]]) -> str:
    if not actions:
        return ""
    lines: List[str] = [f"AI操作建议 - {symbol}"]
    for action in actions:
        timeframe = action.get("timeframe", "?")
        bias = action.get("bias", "?")
        entry = action.get("entry")
        stop = action.get("stop")
        targets = action.get("targets") or []
        confidence = action.get("confidence")
        note = action.get("note")
        lines.append(
            f"[{timeframe}] {bias} | 入场: {entry} | 止损: {stop} | 目标: {targets} | 置信: {confidence} | 说明: {note}"
        )
    return "\n".join(lines)


def send_telegram_alert(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()

