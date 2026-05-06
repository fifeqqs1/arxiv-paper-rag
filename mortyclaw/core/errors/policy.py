from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorKind(str, Enum):
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_AUTH = "provider_auth"
    CONTEXT_OVERFLOW = "context_overflow"
    TOOL_SCHEMA_ERROR = "tool_schema_error"
    TOOL_RUNTIME_ERROR = "tool_runtime_error"
    APPROVAL_BLOCKED = "approval_blocked"
    EMPTY_LLM_RESPONSE = "empty_llm_response"
    UNSAFE_TOOL_SCOPE = "unsafe_tool_scope"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    RETRY = "retry"
    COMPRESS_AND_RETRY = "compress_and_retry"
    ABORT = "abort"
    WAIT_FOR_APPROVAL = "wait_for_approval"
    REPLAN = "replan"


@dataclass(frozen=True)
class RetryPolicy:
    retryable: bool
    max_attempts: int
    backoff_seconds: float = 0.0


@dataclass
class ClassifiedError:
    kind: ErrorKind
    recovery_action: RecoveryAction
    retry_policy: RetryPolicy
    message: str = ""
    user_visible_hint: str = ""
    state_updates: dict[str, Any] = field(default_factory=dict)

    @property
    def retryable(self) -> bool:
        return self.retry_policy.retryable


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _match_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


EXPLICIT_FAILURE_PREFIXES = (
    "执行失败",
    "失败：",
    "错误：",
    "异常：",
    "❌",
    "error:",
    "failed:",
    "failure:",
    "exception:",
    "timeout",
    "permission denied",
    "not found",
    "traceback",
)

TRACEBACK_MARKERS = (
    "traceback (most recent call last)",
    "\ntraceback",
    "\nerror:",
    "\nfailed:",
)

WRITE_BLOCKING_FAILURE_PATTERNS = (
    "没有可用的写文件工具",
    "文件尚未写入",
    "文件不存在",
    "无法验证",
    "没法实际创建文件",
    "cannot verify",
    "file does not exist",
    "no available file write tool",
    "no writable file tool",
)


def looks_like_explicit_failure_text(value: Any) -> bool:
    text = _normalize_text(value)
    if not text:
        return True

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    if any(line.startswith(EXPLICIT_FAILURE_PREFIXES) for line in lines):
        return True

    lowered = text.lower()
    return any(marker in lowered for marker in TRACEBACK_MARKERS) or any(
        pattern in lowered for pattern in WRITE_BLOCKING_FAILURE_PATTERNS
    )


def default_policy_for_kind(kind: ErrorKind) -> tuple[RecoveryAction, RetryPolicy]:
    if kind in {ErrorKind.PROVIDER_TIMEOUT, ErrorKind.PROVIDER_RATE_LIMIT, ErrorKind.TOOL_RUNTIME_ERROR, ErrorKind.EMPTY_LLM_RESPONSE}:
        return RecoveryAction.RETRY, RetryPolicy(retryable=True, max_attempts=2, backoff_seconds=0.0)
    if kind == ErrorKind.CONTEXT_OVERFLOW:
        return RecoveryAction.COMPRESS_AND_RETRY, RetryPolicy(retryable=True, max_attempts=1, backoff_seconds=0.0)
    if kind == ErrorKind.APPROVAL_BLOCKED:
        return RecoveryAction.WAIT_FOR_APPROVAL, RetryPolicy(retryable=False, max_attempts=0)
    if kind in {ErrorKind.TOOL_SCHEMA_ERROR, ErrorKind.UNSAFE_TOOL_SCOPE, ErrorKind.PROVIDER_AUTH}:
        return RecoveryAction.ABORT, RetryPolicy(retryable=False, max_attempts=0)
    return RecoveryAction.REPLAN, RetryPolicy(retryable=False, max_attempts=0)


def classify_error(
    *,
    exc: Exception | None = None,
    message: str = "",
    state: dict[str, Any] | None = None,
    tool_name: str = "",
    provider: str = "",
) -> ClassifiedError:
    state = state or {}
    text = _normalize_text(message or str(exc or ""))
    lowered = text.lower()

    if state.get("pending_approval") and state.get("run_status") in {"waiting_user", "awaiting_approval_response", "awaiting_step_approval"}:
        action, retry = default_policy_for_kind(ErrorKind.APPROVAL_BLOCKED)
        return ClassifiedError(
            kind=ErrorKind.APPROVAL_BLOCKED,
            recovery_action=action,
            retry_policy=retry,
            message=text,
            user_visible_hint="当前步骤正在等待审批确认。",
        )

    if not text and exc is None:
        action, retry = default_policy_for_kind(ErrorKind.EMPTY_LLM_RESPONSE)
        return ClassifiedError(
            kind=ErrorKind.EMPTY_LLM_RESPONSE,
            recovery_action=action,
            retry_policy=retry,
            message="LLM 返回了空响应。",
            user_visible_hint="模型返回了空响应，我会尝试重新组织并继续。",
        )

    if isinstance(exc, TimeoutError) or _match_any(lowered, ("timeout", "timed out", "超时")):
        action, retry = default_policy_for_kind(ErrorKind.PROVIDER_TIMEOUT)
        return ClassifiedError(
            kind=ErrorKind.PROVIDER_TIMEOUT,
            recovery_action=action,
            retry_policy=retry,
            message=text,
            user_visible_hint="模型或工具请求超时，我会尝试重试。",
        )

    if _match_any(lowered, ("rate limit", "too many requests", "429", "throttl", "限流")):
        action, retry = default_policy_for_kind(ErrorKind.PROVIDER_RATE_LIMIT)
        return ClassifiedError(
            kind=ErrorKind.PROVIDER_RATE_LIMIT,
            recovery_action=action,
            retry_policy=retry,
            message=text,
            user_visible_hint="当前请求触发了限流，我会稍作调整后重试。",
        )

    if _match_any(lowered, ("401", "403", "unauthorized", "forbidden", "invalid api key", "authentication", "鉴权")):
        action, retry = default_policy_for_kind(ErrorKind.PROVIDER_AUTH)
        return ClassifiedError(
            kind=ErrorKind.PROVIDER_AUTH,
            recovery_action=action,
            retry_policy=retry,
            message=text,
            user_visible_hint="模型提供商鉴权失败，需要检查当前账号或密钥。",
        )

    if _match_any(lowered, ("context length", "too many tokens", "maximum context", "prompt is too long", "context window", "超过最大长度", "上下文长度")):
        action, retry = default_policy_for_kind(ErrorKind.CONTEXT_OVERFLOW)
        return ClassifiedError(
            kind=ErrorKind.CONTEXT_OVERFLOW,
            recovery_action=action,
            retry_policy=retry,
            message=text,
            user_visible_hint="上下文过长，我会先压缩上下文再继续。",
        )

    if _match_any(lowered, ("schema", "validation error", "missing required", "invalid arguments", "字段", "参数校验")):
        action, retry = default_policy_for_kind(ErrorKind.TOOL_SCHEMA_ERROR)
        return ClassifiedError(
            kind=ErrorKind.TOOL_SCHEMA_ERROR,
            recovery_action=action,
            retry_policy=retry,
            message=text,
            user_visible_hint=f"工具 {tool_name or '调用'} 的参数格式不正确，需要调整参数。",
        )

    if _match_any(lowered, ("越界工具调用", "禁止工具", "out-of-scope tool", "unsafe tool scope")):
        action, retry = default_policy_for_kind(ErrorKind.UNSAFE_TOOL_SCOPE)
        return ClassifiedError(
            kind=ErrorKind.UNSAFE_TOOL_SCOPE,
            recovery_action=action,
            retry_policy=retry,
            message=text,
            user_visible_hint="当前步骤触发了越界工具调用，不能继续按这个动作执行。",
        )

    if looks_like_explicit_failure_text(text):
        action, retry = default_policy_for_kind(ErrorKind.TOOL_RUNTIME_ERROR)
        return ClassifiedError(
            kind=ErrorKind.TOOL_RUNTIME_ERROR,
            recovery_action=action,
            retry_policy=retry,
            message=text,
            user_visible_hint=f"{tool_name or '工具'} 执行失败，我会评估是否重试。",
        )

    action, retry = default_policy_for_kind(ErrorKind.UNKNOWN)
    return ClassifiedError(
        kind=ErrorKind.UNKNOWN,
        recovery_action=action,
        retry_policy=retry,
        message=text or _normalize_text(provider),
        user_visible_hint="遇到未分类异常，我会尝试重新规划当前步骤。",
    )


def serialize_classified_error(item: ClassifiedError) -> dict[str, Any]:
    return {
        "kind": item.kind.value,
        "recovery_action": item.recovery_action.value,
        "retryable": item.retryable,
        "max_attempts": item.retry_policy.max_attempts,
        "backoff_seconds": item.retry_policy.backoff_seconds,
        "message": item.message,
        "user_visible_hint": item.user_visible_hint,
        "state_updates": dict(item.state_updates or {}),
    }


def deserialize_classified_error(data: dict[str, Any] | None) -> ClassifiedError | None:
    if not isinstance(data, dict):
        return None
    try:
        kind = ErrorKind(str(data.get("kind") or ErrorKind.UNKNOWN.value))
        action = RecoveryAction(str(data.get("recovery_action") or RecoveryAction.REPLAN.value))
    except ValueError:
        return None
    return ClassifiedError(
        kind=kind,
        recovery_action=action,
        retry_policy=RetryPolicy(
            retryable=bool(data.get("retryable", False)),
            max_attempts=int(data.get("max_attempts", 0) or 0),
            backoff_seconds=float(data.get("backoff_seconds", 0.0) or 0.0),
        ),
        message=_normalize_text(data.get("message")),
        user_visible_hint=_normalize_text(data.get("user_visible_hint")),
        state_updates=dict(data.get("state_updates") or {}),
    )
