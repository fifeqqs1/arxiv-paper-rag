from __future__ import annotations

import os
from datetime import datetime


def get_system_model_info_impl() -> str:
    provider = os.getenv("DEFAULT_PROVIDER", "unknown")
    model = os.getenv("DEFAULT_MODEL", "unknown")

    if provider == "unknown" or model == "unknown":
        return "无法获取当前的系统模型配置，可能是环境变量未正确加载。"

    return f"当前使用的模型提供商(Provider)是: {provider}，具体型号(Model)是: {model}。"


def get_current_time_impl(*, now_fn=datetime.now) -> str:
    now = now_fn()
    return f"当前本地系统时间是: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def calculator_impl(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"表达式 '{expression}' 的计算结果是: {result}"
    except Exception as exc:
        return f"计算出错，请检查表达式格式。错误信息: {str(exc)}"
