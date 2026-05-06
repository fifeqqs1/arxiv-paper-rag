import hashlib
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable


SESSION_MEMORY_PROMPT_LIMIT = 5
LONG_TERM_MEMORY_PROMPT_LIMIT = 4
LONG_TERM_MEMORY_TYPES = (
    "user_preference",
    "project_fact",
    "workflow_preference",
    "safety_preference",
)
LEGACY_LONG_TERM_MEMORY_TYPES = ("user_preference_note",)
SEARCHABLE_LONG_TERM_MEMORY_TYPES = LONG_TERM_MEMORY_TYPES + LEGACY_LONG_TERM_MEMORY_TYPES

LONG_TERM_MEMORY_TYPE_LABELS = {
    "user_preference": "用户偏好",
    "project_fact": "项目事实",
    "workflow_preference": "工作流偏好",
    "safety_preference": "安全偏好",
    "user_preference_note": "用户偏好",
}

LONG_TERM_MEMORY_SUBJECT_LABELS = {
    "response_language": "回复语言",
    "answer_style": "回答风格",
    "addressing": "称呼偏好",
    "project_path": "项目路径",
    "project_context": "项目上下文",
    "testing_workflow": "测试流程",
    "implementation_order": "执行顺序",
    "workflow_policy": "工作流规则",
    "approval_policy": "审批规则",
    "sandbox_policy": "沙盒规则",
    "safety_policy": "安全规则",
}

ABSOLUTE_PATH_PATTERN = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s`\"']+")
LIKELY_UNROOTED_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./~-])"
    r"((?:mnt|tmp|home|Users|opt|var|srv)/[^\s`\"']+)"
)

LONG_TERM_RECALL_HINTS = (
    "记住",
    "偏好",
    "喜欢",
    "习惯",
    "之前",
    "以前",
    "还记得",
    "根据我的",
    "我的设置",
    "我的风格",
    "profile",
    "preference",
    "preferences",
    "remember",
)


class MemoryPromptCache:
    def __init__(self, max_entries: int = 128):
        self.max_entries = max(1, max_entries)
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple, str] = OrderedDict()

    def get_or_build(self, key: tuple, builder: Callable[[], str]) -> str:
        with self._lock:
            if key in self._entries:
                value = self._entries.pop(key)
                self._entries[key] = value
                return value

        value = builder()

        with self._lock:
            self._entries[key] = value
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _store_cache_namespace(store) -> tuple:
    return (
        getattr(store, "db_path", ""),
        id(store),
        getattr(store, "revision", 0),
    )


def _profile_file_mtime(memory_dir: str) -> int:
    profile_path = os.path.join(memory_dir, "user_profile.md")
    try:
        return os.stat(profile_path).st_mtime_ns
    except OSError:
        return 0


def session_memory_id(scope: str, memory_type: str) -> str:
    safe_scope = re.sub(r"[^a-zA-Z0-9_-]+", "_", scope) or "default"
    safe_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", memory_type) or "memory"
    return f"session::{safe_scope}::{safe_type}"


def _strip_path_trailing_punctuation(path_value: str) -> str:
    return (path_value or "").rstrip(".,);:!?]}>'\"，。；：！？】）》")


def _looks_like_path_command(path_value: str) -> bool:
    normalized = _strip_path_trailing_punctuation(path_value)
    if not normalized:
        return False
    if normalized in {"office", "~"}:
        return True
    drive_match = re.match(r"^[A-Za-z]:[\\/]", normalized)
    if drive_match:
        tail = normalized[drive_match.end():]
        return bool(tail and any(sep in tail for sep in ("\\", "/")))
    if normalized.startswith("/"):
        segments = [segment for segment in normalized.split("/") if segment]
        return len(segments) >= 2
    if normalized.startswith(("mnt/", "tmp/", "home/", "Users/", "opt/", "var/", "srv/")):
        segments = [segment for segment in normalized.split("/") if segment]
        return len(segments) >= 2
    return False


def _normalize_path_candidate(path_value: str) -> str:
    candidate = _strip_path_trailing_punctuation(path_value)
    if not candidate:
        return ""
    if candidate == "office":
        return candidate
    if candidate.startswith(("mnt/", "tmp/", "home/", "Users/", "opt/", "var/", "srv/")):
        candidate = "/" + candidate
    if candidate.startswith("~"):
        candidate = os.path.expanduser(candidate)
    if not _looks_like_path_command(candidate):
        return ""
    resolved = os.path.realpath(candidate)
    if os.path.exists(resolved):
        return resolved
    if os.path.exists(os.path.dirname(resolved)) and _looks_like_path_command(candidate):
        return resolved
    return ""


def extract_primary_path(query: str | None) -> str:
    if not query:
        return ""
    candidates: list[str] = []
    for match in ABSOLUTE_PATH_PATTERN.finditer(query):
        candidates.append(match.group(0))
    for match in LIKELY_UNROOTED_PATH_PATTERN.finditer(query):
        candidates.append(match.group(1))

    for candidate in candidates:
        normalized = _normalize_path_candidate(candidate)
        if normalized:
            return normalized
    if "office" in query.lower():
        return "office"
    return ""


def extract_session_memory_records(query: str | None, thread_id: str, *, build_memory_record_fn) -> list[dict]:
    if not query:
        return []

    normalized_query = query.strip()
    lowered = normalized_query.lower()
    records: list[dict] = []

    path = extract_primary_path(normalized_query)
    if path:
        records.append(build_memory_record_fn(
            memory_id=session_memory_id(thread_id, "project_path"),
            layer="session",
            scope=thread_id,
            type="project_path",
            content=path,
            source_kind="rule_extractor",
            source_ref=normalized_query[:200],
            confidence=1.0,
        ))

    if any(phrase in normalized_query for phrase in ("用中文", "中文回答", "中文输出", "中文回复")):
        records.append(build_memory_record_fn(
            memory_id=session_memory_id(thread_id, "response_language"),
            layer="session",
            scope=thread_id,
            type="response_language",
            content="请使用中文输出。",
            source_kind="rule_extractor",
            source_ref=normalized_query[:200],
            confidence=0.95,
        ))

    if any(phrase in normalized_query for phrase in ("不要修改代码", "不要改代码", "别修改代码", "别改代码", "只分析", "不要修改任何代码")):
        records.append(build_memory_record_fn(
            memory_id=session_memory_id(thread_id, "code_change_policy"),
            layer="session",
            scope=thread_id,
            type="code_change_policy",
            content="本轮只分析，不修改任何代码。",
            source_kind="rule_extractor",
            source_ref=normalized_query[:200],
            confidence=1.0,
        ))

    if "office" in lowered and any(marker in normalized_query for marker in ("路径", "目录", "放到", "放在", "下面")):
        records.append(build_memory_record_fn(
            memory_id=session_memory_id(thread_id, "operation_workspace"),
            layer="session",
            scope=thread_id,
            type="operation_workspace",
            content="office",
            source_kind="rule_extractor",
            source_ref=normalized_query[:200],
            confidence=0.9,
        ))

    if "确认" in normalized_query and "高风险" in normalized_query and any(marker in normalized_query for marker in ("才", "只", "仅", "不要", "中低风险")):
        records.append(build_memory_record_fn(
            memory_id=session_memory_id(thread_id, "approval_policy"),
            layer="session",
            scope=thread_id,
            type="approval_policy",
            content="仅高风险步骤需要确认，中低风险步骤可直接执行。",
            source_kind="rule_extractor",
            source_ref=normalized_query[:200],
            confidence=0.95,
        ))

    return records


def sync_session_memory_from_query(
    query: str | None,
    thread_id: str,
    *,
    get_memory_store_fn,
    build_memory_record_fn,
) -> dict:
    extracted_records = extract_session_memory_records(
        query,
        thread_id,
        build_memory_record_fn=build_memory_record_fn,
    )
    if not extracted_records:
        return {}

    store = get_memory_store_fn()
    state_updates: dict = {}
    for record in extracted_records:
        store.upsert_memory(record)
        if record["type"] == "project_path":
            state_updates["current_project_path"] = record["content"]
    return state_updates


def load_session_memory_records(thread_id: str, *, get_memory_store_fn, limit: int = SESSION_MEMORY_PROMPT_LIMIT) -> list[dict]:
    return get_memory_store_fn().list_memories(
        layer="session",
        scope=thread_id,
        limit=limit,
    )


def load_session_project_path(thread_id: str, *, get_memory_store_fn) -> str:
    records = get_memory_store_fn().list_memories(
        layer="session",
        scope=thread_id,
        memory_type="project_path",
        limit=1,
    )
    if not records:
        return ""
    return records[0].get("content", "")


def format_session_memory_for_prompt(session_records: list[dict]) -> str:
    if not session_records:
        return ""

    labels = {
        "project_path": "当前项目路径",
        "response_language": "输出偏好",
        "code_change_policy": "代码策略",
        "operation_workspace": "当前操作目录",
        "approval_policy": "审批偏好",
    }
    lines = []
    for record in session_records:
        label = labels.get(record.get("type", ""), record.get("type", "会话记忆"))
        lines.append(f"- {label}：{record.get('content', '')}")
    return "\n".join(lines)


def build_session_memory_prompt(
    thread_id: str,
    *,
    get_memory_store_fn,
    limit: int = SESSION_MEMORY_PROMPT_LIMIT,
    prompt_cache: MemoryPromptCache | None = None,
) -> str:
    store = get_memory_store_fn()
    cache_key = (
        "session_prompt",
        _store_cache_namespace(store),
        thread_id,
        limit,
    )

    def build_prompt() -> str:
        records = store.list_memories(
            layer="session",
            scope=thread_id,
            limit=limit,
        )
        return format_session_memory_for_prompt(records)

    if prompt_cache is None:
        return build_prompt()
    return prompt_cache.get_or_build(cache_key, build_prompt)


def should_recall_long_term_memory(query: str | None) -> bool:
    if not query:
        return False
    lowered = query.lower()
    return any(hint in query or hint in lowered for hint in LONG_TERM_RECALL_HINTS)


def long_term_note_memory_id(content: str) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    return f"long_term::{{scope}}::user_preference_note::{digest}"


def classify_long_term_memory_type(query: str) -> str:
    lowered = query.lower()
    if any(marker in query for marker in ("高风险", "确认", "审批", "权限", "安全", "沙盒", "越权")):
        return "safety_preference"
    if any(marker in query for marker in ("工作流", "流程", "步骤", "先", "再", "测试", "运行测试", "提交前", "实现前")):
        return "workflow_preference"
    if any(marker in query for marker in ("项目", "仓库", "代码库", "目录", "路径", "模块", "服务", "端口", "数据库")) or ABSOLUTE_PATH_PATTERN.search(query):
        return "project_fact"
    if any(marker in query for marker in ("喜欢", "不喜欢", "偏好", "风格", "习惯", "用中文", "英文", "简洁")) or "preference" in lowered:
        return "user_preference"
    return "user_preference"


def classify_long_term_memory_subject(memory_type: str, query: str) -> str:
    lowered = query.lower()
    if memory_type == "user_preference":
        if any(marker in query for marker in ("中文", "英文", "语言", "中文回答", "英文回答")):
            return "response_language"
        if any(marker in query for marker in ("简洁", "详细", "风格", "语气", "口吻", "啰嗦", "markdown", "列表")):
            return "answer_style"
        if any(marker in query for marker in ("叫我", "称呼", "名字")):
            return "addressing"
        if "preference" in lowered:
            return "general_preference"
        return "general_preference"

    if memory_type == "project_fact":
        if ABSOLUTE_PATH_PATTERN.search(query) or any(marker in query for marker in ("路径", "目录", "仓库", "项目位置")):
            return "project_path"
        return "project_context"

    if memory_type == "workflow_preference":
        if any(marker in query for marker in ("测试", "运行测试", "回归", "pytest", "unittest")):
            return "testing_workflow"
        if any(marker in query for marker in ("先", "再", "步骤", "流程", "提交前", "实现前")):
            return "implementation_order"
        return "workflow_policy"

    if memory_type == "safety_preference":
        if any(marker in query for marker in ("确认", "审批", "高风险", "中低风险")):
            return "approval_policy"
        if any(marker in query for marker in ("沙盒", "越权", "权限")):
            return "sandbox_policy"
        return "safety_policy"

    return ""


def extract_long_term_memory_records(
    query: str | None,
    *,
    build_memory_record_fn,
    default_long_term_scope: str,
) -> list[dict]:
    if not query:
        return []

    normalized_query = query.strip()
    lowered = normalized_query.lower()
    should_capture = (
        "记住" in normalized_query
        or "以后" in normalized_query
        or "一直" in normalized_query
        or "长期" in normalized_query
        or "我喜欢" in normalized_query
        or "我不喜欢" in normalized_query
        or "my preference" in lowered
    )
    if not should_capture:
        return []

    digest = hashlib.sha1(normalized_query.encode("utf-8")).hexdigest()[:16]
    memory_type = classify_long_term_memory_type(normalized_query)
    subject = classify_long_term_memory_subject(memory_type, normalized_query)
    memory_id = f"long_term::{default_long_term_scope}::{memory_type}::{digest}"
    return [build_memory_record_fn(
        memory_id=memory_id,
        layer="long_term",
        scope=default_long_term_scope,
        type=memory_type,
        subject=subject,
        content=normalized_query,
        source_kind="rule_extractor",
        source_ref=normalized_query[:200],
        confidence=0.8,
    )]


def schedule_long_term_memory_capture(
    query: str | None,
    *,
    get_async_memory_writer_fn,
    build_memory_record_fn,
    default_long_term_scope: str,
) -> None:
    records = extract_long_term_memory_records(
        query,
        build_memory_record_fn=build_memory_record_fn,
        default_long_term_scope=default_long_term_scope,
    )
    if not records:
        return
    writer = get_async_memory_writer_fn()
    for record in records:
        writer.submit(record)


def load_long_term_profile_content(
    *,
    get_memory_store_fn,
    memory_dir: str,
    default_long_term_scope: str,
    user_profile_memory_type: str,
) -> str:
    records = get_memory_store_fn().list_memories(
        layer="long_term",
        scope=default_long_term_scope,
        memory_type=user_profile_memory_type,
        limit=1,
    )
    if records:
        content = records[0].get("content", "").strip()
        if content:
            return content

    profile_path = os.path.join(memory_dir, "user_profile.md")
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if content:
                return content

    return "暂无记录"


def build_long_term_memory_prompt(
    query: str | None,
    *,
    get_memory_store_fn,
    memory_dir: str,
    default_long_term_scope: str,
    user_profile_memory_type: str,
    long_term_memory_prompt_limit: int = LONG_TERM_MEMORY_PROMPT_LIMIT,
    prompt_cache: MemoryPromptCache | None = None,
) -> str:
    if not should_recall_long_term_memory(query):
        return ""

    store = get_memory_store_fn()
    normalized_query = (query or "").strip()
    cache_key = (
        "long_term_prompt",
        _store_cache_namespace(store),
        normalized_query[:300],
        memory_dir,
        _profile_file_mtime(memory_dir),
        default_long_term_scope,
        user_profile_memory_type,
        long_term_memory_prompt_limit,
    )

    def build_prompt() -> str:
        sections = []
        profile_content = load_long_term_profile_content(
            get_memory_store_fn=lambda: store,
            memory_dir=memory_dir,
            default_long_term_scope=default_long_term_scope,
            user_profile_memory_type=user_profile_memory_type,
        )
        if profile_content and profile_content != "暂无记录":
            sections.append(
                "【用户长期画像 (静态偏好)】\n"
                f"{profile_content}"
            )

        note_records = store.search_memories(
            normalized_query,
            layer="long_term",
            scope=default_long_term_scope,
            memory_types=SEARCHABLE_LONG_TERM_MEMORY_TYPES,
            limit=long_term_memory_prompt_limit,
        )
        if not note_records:
            note_records = store.list_memories(
                layer="long_term",
                scope=default_long_term_scope,
                limit=long_term_memory_prompt_limit,
            )
        note_lines = []
        for record in note_records:
            if record.get("type") == user_profile_memory_type:
                continue
            label = LONG_TERM_MEMORY_TYPE_LABELS.get(record.get("type", ""), "长期事实")
            subject = record.get("subject", "")
            subject_label = LONG_TERM_MEMORY_SUBJECT_LABELS.get(subject, subject)
            prefix = f"{label}/{subject_label}" if subject_label else label
            note_lines.append(f"- {prefix}：{record.get('content', '')}")
        if note_lines:
            sections.append("【长期事实记忆】\n" + "\n".join(note_lines))

        return "\n\n".join(section for section in sections if section.strip())

    if prompt_cache is None:
        return build_prompt()
    return prompt_cache.get_or_build(cache_key, build_prompt)


def with_working_memory(state, updates: dict, *, build_working_memory_snapshot_fn) -> dict:
    projected_state = dict(state)
    for key, value in updates.items():
        if key == "messages":
            continue
        projected_state[key] = value
    return {
        **updates,
        "working_memory": build_working_memory_snapshot_fn(projected_state),
    }
