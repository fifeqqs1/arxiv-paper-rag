import json
import os
import re
from urllib import error, request

from .base import mortyclaw_tool


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_ARXIV_RAG_API_BASE = "http://127.0.0.1:8001"
DEFAULT_ARXIV_RAG_FEISHU_REPLY_PATH = "/api/v1/feishu/reply"
DEFAULT_ARXIV_RAG_SESSION_ID = "mortyclaw_default"
MORTYCLAW_PASSTHROUGH_FLAG = "_mortyclaw_passthrough"


def _compact_text(value: str, limit: int = 400) -> str:
    text = re.sub(r"\s+", " ", (value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 30) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )

    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _get_first_env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return default


@mortyclaw_tool
def tavily_web_search(
    query: str,
    topic: str = "general",
    search_depth: str = "basic",
    max_results: int = 5,
    include_answer: bool = True,
) -> str:
    """
    使用 Tavily 联网搜索最新网页信息。
    适合处理以下场景：
    1. 用户明确要求联网、搜索、查资料、找来源。
    2. 问题涉及新闻、实时信息、最新动态、当前版本、外部网页内容。
    3. 回答时需要附带来源链接。

    参数说明：
    - query: 搜索关键词或完整问题。
    - topic: 搜索主题，推荐使用 "general"；如果是新闻/时事，使用 "news"。
    - search_depth: "basic" 或 "advanced"。普通查询用 basic，需要更深入检索时用 advanced。
    - max_results: 返回结果数量，建议 3 到 8。
    - include_answer: 是否让 Tavily 返回一个搜索摘要。
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return "Tavily 搜索不可用：未配置 TAVILY_API_KEY 环境变量。"

    query = (query or "").strip()
    if not query:
        return "Tavily 搜索参数错误：query 不能为空。"

    topic = (topic or "general").strip().lower()
    if topic not in {"general", "news"}:
        return "Tavily 搜索参数错误：topic 只能是 'general' 或 'news'。"

    search_depth = (search_depth or "basic").strip().lower()
    if search_depth not in {"basic", "advanced"}:
        return "Tavily 搜索参数错误：search_depth 只能是 'basic' 或 'advanced'。"

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        return "Tavily 搜索参数错误：max_results 必须是整数。"

    max_results = max(1, min(max_results, 10))

    payload = {
        "query": query,
        "topic": topic,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": bool(include_answer),
        "include_raw_content": False,
        "include_images": False,
    }

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        TAVILY_SEARCH_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
            parsed = json.loads(detail) if detail else {}
            message = parsed.get("detail") or parsed.get("error") or detail
        except Exception:
            message = str(exc)
        return f"Tavily 搜索失败：HTTP {exc.code}，{_compact_text(message, 200)}"
    except error.URLError as exc:
        return f"Tavily 搜索失败：网络错误，{exc.reason}"
    except Exception as exc:
        return f"Tavily 搜索失败：{exc}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "Tavily 搜索失败：返回结果不是合法 JSON。"

    answer = _compact_text(data.get("answer", ""), 500)
    results = data.get("results") or []

    if not results and not answer:
        return "Tavily 未返回任何搜索结果。"

    lines = [
        f"Tavily 搜索完成：query={query}",
        f"topic={topic}, search_depth={search_depth}, max_results={max_results}",
    ]

    if answer:
        lines.append(f"搜索摘要：{answer}")

    if not results:
        lines.append("来源链接：无")
        return "\n".join(lines)

    lines.append("来源结果：")
    for index, item in enumerate(results, start=1):
        title = _compact_text(item.get("title", "(无标题)"), 160)
        url = item.get("url", "")
        content = _compact_text(item.get("content", ""), 300)
        score = item.get("score")

        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if content:
            lines.append(f"   摘要: {content}")
        if isinstance(score, (int, float)):
            lines.append(f"   相关度: {score:.3f}")

    return "\n".join(lines)


@mortyclaw_tool
def arxiv_rag_ask(
    query: str,
    session_id: str = "",
) -> str:
    """
    调用本地 arxiv_rag 的 Feishu 会话入口，直接返回最终回答。
    适合处理以下场景：
    1. 用户想查学术论文、arXiv 论文、研究方法、模型原理、实验结论。
    2. 用户希望复用 arxiv_rag 在飞书机器人里的论文推荐、跟进提问和会话记忆逻辑。
    3. MortyClaw 只做转发，不自行改写 query，也不参与二次总结。

    参数说明：
    - query: 用户的原始学术问题。
    - session_id: 会话 ID。MortyClaw 内部应传当前 thread_id，用于复用 Feishu 会话记忆。

    重要约束：
    - 本工具只把原始 query 和 session_id 发给 arxiv_rag。
    - 不再传检索控制参数，也不在 MortyClaw 侧选择 /ask 或 /ask-agentic。
    - 一旦本工具成功返回，MortyClaw 会直接把结果发给用户，不再二次改写。
    """
    if not isinstance(query, str) or not query.strip():
        return "arxiv_rag 问答参数错误：query 不能为空。"

    base_url = _get_first_env(
        "ARXIV_RAG_API_BASE",
        "FEISHU__API_BASE_URL",
        default=DEFAULT_ARXIV_RAG_API_BASE,
    ).rstrip("/")
    endpoint_path = _get_first_env(
        "ARXIV_RAG_FEISHU_REPLY_PATH",
        default=DEFAULT_ARXIV_RAG_FEISHU_REPLY_PATH,
    )

    try:
        timeout_seconds = int(
            _get_first_env(
                "ARXIV_RAG_TIMEOUT_SECONDS",
                "FEISHU__REQUEST_TIMEOUT_SECONDS",
                default="60",
            )
        )
    except ValueError:
        return "arxiv_rag 问答参数错误：ARXIV_RAG_TIMEOUT_SECONDS 必须是整数。"

    payload = {
        "query": query,
        "session_id": (session_id or "").strip() or DEFAULT_ARXIV_RAG_SESSION_ID,
    }

    try:
        data = _post_json(f"{base_url}{endpoint_path}", payload, timeout=timeout_seconds)
    except error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
            parsed = json.loads(detail) if detail else {}
            message = parsed.get("detail") or parsed.get("error") or detail
        except Exception:
            message = str(exc)
        return f"arxiv_rag 调用失败：HTTP {exc.code}，{_compact_text(message, 200)}"
    except error.URLError as exc:
        return f"arxiv_rag 调用失败：网络错误，{exc.reason}"
    except json.JSONDecodeError:
        return "arxiv_rag 调用失败：返回结果不是合法 JSON。"
    except Exception as exc:
        return f"arxiv_rag 调用失败：{exc}"

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return "arxiv_rag 调用失败：返回结果中缺少 answer 字段。"

    passthrough_payload = {
        MORTYCLAW_PASSTHROUGH_FLAG: True,
        "tool": "arxiv_rag_ask",
        "display_text": answer,
        "query": data.get("query", query),
        "answer": answer,
        "session_id": data.get("session_id", payload["session_id"]),
        "endpoint_path": endpoint_path,
    }
    return json.dumps(passthrough_payload, ensure_ascii=False)
