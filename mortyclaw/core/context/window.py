from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


MESSAGE_TOKEN_OVERHEAD = 8
TOOL_MESSAGE_EXTRA_TOKEN_OVERHEAD = 12


def _serialize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_serialize_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)
    return str(value)


class _FallbackEncoder:
    def encode(self, text: str) -> list[int]:
        return list(str(text or "").encode("utf-8", errors="ignore"))


@lru_cache(maxsize=64)
def _resolve_token_encoder(model_name: str):
    if tiktoken is None:  # pragma: no cover
        return _FallbackEncoder()

    normalized = str(model_name or "").strip()
    if normalized:
        try:
            return tiktoken.encoding_for_model(normalized)
        except Exception:
            pass

    for encoding_name in ("o200k_base", "cl100k_base"):
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception:
            continue
    return _FallbackEncoder()


def _estimate_message_tokens(message: BaseMessage, *, encoder) -> int:
    text = _serialize_content(getattr(message, "content", ""))
    estimated = len(encoder.encode(text)) + MESSAGE_TOKEN_OVERHEAD
    if isinstance(message, ToolMessage):
        estimated += TOOL_MESSAGE_EXTRA_TOKEN_OVERHEAD
    return estimated


def _estimate_messages_tokens(messages: list[BaseMessage], *, encoder) -> int:
    return sum(_estimate_message_tokens(message, encoder=encoder) for message in messages)


def _effective_token_budget(raw_budget: int | None, reserve_tokens: int) -> int | None:
    if raw_budget is None:
        return None
    if raw_budget <= 0:
        return None
    return max(raw_budget - max(reserve_tokens, 0), 1)


def _trim_messages_to_token_budget(
    messages: list[BaseMessage],
    keep_tokens: int,
    reserve_tokens: int,
    model_name: str,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    effective_keep_tokens = _effective_token_budget(keep_tokens, reserve_tokens)
    if effective_keep_tokens is None:
        return messages, []

    encoder = _resolve_token_encoder(model_name)
    message_costs = [
        _estimate_message_tokens(message, encoder=encoder)
        for message in messages
    ]
    total_tokens = sum(message_costs)
    if total_tokens <= effective_keep_tokens:
        return messages, []

    latest_human_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            latest_human_index = index
            break

    removable_indices = [
        index
        for index in range(len(messages))
        if index != latest_human_index
        and not isinstance(messages[index], SystemMessage)
    ]
    keep_indices = set(range(len(messages)))

    for index in removable_indices:
        if total_tokens <= effective_keep_tokens:
            break
        keep_indices.remove(index)
        total_tokens -= message_costs[index]

    kept = [message for index, message in enumerate(messages) if index in keep_indices]
    discarded = [message for index, message in enumerate(messages) if index not in keep_indices]
    return kept, discarded


def _trim_messages_to_budget(
    messages: list[BaseMessage],
    keep_messages: int,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    if keep_messages <= 0 or len(messages) <= keep_messages:
        return messages, []

    latest_human_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            latest_human_index = index
            break

    keep_indices = set(range(max(0, len(messages) - keep_messages), len(messages)))
    if latest_human_index >= 0:
        keep_indices.add(latest_human_index)

    while len(keep_indices) > keep_messages:
        removable = [index for index in sorted(keep_indices) if index != latest_human_index]
        if not removable:
            break
        keep_indices.remove(removable[0])

    kept = [message for index, message in enumerate(messages) if index in keep_indices]
    discarded = [message for index, message in enumerate(messages) if index not in keep_indices]
    return kept, discarded


def trim_context_messages(
    messages: list[BaseMessage],
    trigger_turns: int = 8,
    keep_turns: int = 4,
    *,
    trigger_messages: int | None = None,
    keep_messages: int | None = None,
    trigger_tokens: int | None = None,
    keep_tokens: int | None = None,
    reserve_tokens: int = 0,
    model_name: str = "",
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    first_system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    if not non_system_msgs:
        return ([first_system] if first_system else []), []

    if trigger_tokens is not None and keep_tokens is not None:
        token_messages = ([first_system] if first_system else []) + list(non_system_msgs)
        encoder = _resolve_token_encoder(model_name)
        effective_trigger_tokens = _effective_token_budget(trigger_tokens, reserve_tokens)
        message_tokens = _estimate_messages_tokens(token_messages, encoder=encoder)
        token_triggered = (
            effective_trigger_tokens is not None
            and message_tokens >= effective_trigger_tokens
        )
        if token_triggered:
            final_messages, discarded_messages = _trim_messages_to_token_budget(
                token_messages,
                keep_tokens=keep_tokens,
                reserve_tokens=reserve_tokens,
                model_name=model_name,
            )
            return final_messages, discarded_messages
        return token_messages, []

    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []

    for msg in non_system_msgs:
        if isinstance(msg, HumanMessage):
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            if current_turn:
                current_turn.append(msg)

    if current_turn:
        turns.append(current_turn)

    total_turns = len(turns)
    turn_triggered = total_turns >= trigger_turns
    message_triggered = (
        trigger_messages is not None
        and trigger_messages > 0
        and len(non_system_msgs) >= trigger_messages
    )

    if turn_triggered:
        recent_turns = turns[-keep_turns:]
        discarded_turns = turns[:-keep_turns]
        kept_non_system: list[BaseMessage] = []
        for turn in recent_turns:
            kept_non_system.extend(turn)

        discarded_messages: list[BaseMessage] = []
        for turn in discarded_turns:
            discarded_messages.extend(turn)
    else:
        kept_non_system = list(non_system_msgs)
        discarded_messages = []

    if keep_messages is not None and keep_messages > 0:
        should_trim_by_budget = message_triggered or len(kept_non_system) > keep_messages
        if should_trim_by_budget:
            kept_non_system, extra_discarded = _trim_messages_to_budget(kept_non_system, keep_messages)
            discarded_messages.extend(extra_discarded)

    if not discarded_messages:
        final_messages = ([first_system] if first_system else []) + kept_non_system
        return final_messages, []

    final_messages: list[BaseMessage] = []
    if first_system:
        final_messages.append(first_system)
    final_messages.extend(kept_non_system)
    return final_messages, discarded_messages
