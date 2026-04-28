import json
import logging
import re
from typing import Any, Dict, List, Optional, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from ..models import ReasoningStep, SourceItem, ToolArtefact

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


def extract_sources_from_tool_messages(messages: List) -> List[SourceItem]:
    """Extract sources from tool messages in conversation.

    :param messages: List of messages from graph state
    :returns: List of SourceItem objects
    """
    sources = []

    for msg in messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
            if msg.name == "retrieve_papers":
                # Parse tool response for sources
                # This would need to parse the actual document metadata
                # For now, return empty list
                pass

    return sources


def extract_tool_artefacts(messages: List) -> List[ToolArtefact]:
    """Extract tool artifacts from messages.

    :param messages: List of messages from graph state
    :returns: List of ToolArtefact objects
    """
    artefacts = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            artefact = ToolArtefact(
                tool_name=getattr(msg, "name", "unknown"),
                tool_call_id=getattr(msg, "tool_call_id", ""),
                content=msg.content,
                metadata={},
            )
            artefacts.append(artefact)

    return artefacts


def create_reasoning_step(
    step_name: str,
    description: str,
    metadata: Optional[Dict] = None,
) -> ReasoningStep:
    """Create a reasoning step record.

    :param step_name: Name of the step/node
    :param description: Human-readable description
    :param metadata: Additional metadata
    :returns: ReasoningStep object
    """
    return ReasoningStep(
        step_name=step_name,
        description=description,
        metadata=metadata or {},
    )


def filter_messages(messages: List) -> List[AIMessage | HumanMessage]:
    """Filter messages to include only HumanMessage and AIMessage types.

    Excludes tool messages and other internal message types.

    :param messages: List of messages to filter
    :returns: Filtered list of messages
    """
    return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]


def get_latest_query(messages: List) -> str:
    """Get the latest user query from messages.

    :param messages: List of messages
    :returns: Latest query text
    :raises ValueError: If no user query found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

    raise ValueError("No user query found in messages")


def get_latest_context(messages: List) -> str:
    """Get the latest context from tool messages.

    :param messages: List of messages
    :returns: Latest context text or empty string
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content if hasattr(msg, "content") else ""

    return ""


def parse_retrieved_hits(messages: List) -> List[Dict[str, Any]]:
    """Extract structured retrieval hits from tool messages when available."""
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        raw_content = getattr(msg, "content", "")
        if not isinstance(raw_content, str):
            continue
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("hits"), list):
            return [hit for hit in payload["hits"] if isinstance(hit, dict)]
    return []


def _extract_json_payload(raw_response: str) -> Dict:
    """Extract a JSON object from raw model text."""
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(raw_response)
        if not match:
            raise
        return json.loads(match.group(0))


async def generate_structured_output(
    runtime,
    prompt: str,
    output_model: type[StructuredModelT],
    temperature: float = 0.0,
) -> StructuredModelT:
    """Generate structured JSON output using plain text generation plus local validation."""
    schema_json = json.dumps(output_model.model_json_schema(), ensure_ascii=False)
    structured_prompt = (
        f"{prompt}\n\n"
        "Return ONLY valid JSON. Do not include markdown fences, explanations, or extra text.\n"
        f"JSON schema:\n{schema_json}"
    )
    response = await runtime.context.ollama_client.generate(
        model=runtime.context.model_name,
        prompt=structured_prompt,
        stream=False,
        provider=runtime.context.llm_provider,
        temperature=temperature,
    )
    raw_response = (response or {}).get("response", "").strip()
    if not raw_response:
        raise ValueError("LLM returned an empty response")

    try:
        return output_model.model_validate_json(raw_response)
    except (ValidationError, json.JSONDecodeError, ValueError):
        payload = _extract_json_payload(raw_response)
        return output_model.model_validate(payload)


async def generate_text_output(runtime, prompt: str, temperature: float = 0.0) -> str:
    """Generate plain-text output using the stable Ollama /generate endpoint."""
    response = await runtime.context.ollama_client.generate(
        model=runtime.context.model_name,
        prompt=prompt,
        stream=False,
        provider=runtime.context.llm_provider,
        temperature=temperature,
    )
    raw_response = (response or {}).get("response", "").strip()
    if not raw_response:
        raise ValueError("LLM returned an empty response")
    return raw_response
