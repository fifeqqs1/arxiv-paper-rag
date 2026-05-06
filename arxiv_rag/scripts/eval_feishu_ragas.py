#!/usr/bin/env python
"""Lightweight Feishu RAG evaluation with optional Ragas scoring."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

PASS_THRESHOLDS = {
    "faithfulness": 0.80,
    "response_relevancy": 0.75,
    "context_precision": 0.65,
    "pass_rate": 0.80,
    "source_rate": 0.90,
    "memory_pass_rate": 0.80,
}


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overriding exported environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _contains_all(answer: str, expected: list[str]) -> tuple[bool, list[str]]:
    lowered = answer.lower()
    missing = [term for term in expected if term.lower() not in lowered]
    return not missing, missing


def _contains_none(answer: str, forbidden: list[str]) -> tuple[bool, list[str]]:
    lowered = answer.lower()
    hits = [term for term in forbidden if term.lower() in lowered]
    return not hits, hits


def _percent(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _safe_mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    return statistics.mean(clean) if clean else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((percentile / 100) * len(ordered)) - 1))
    return ordered[index]


def _call_feishu_reply(
    client: httpx.Client,
    base_url: str,
    case: dict[str, Any],
    session_prefix: str,
    *,
    retries: int,
    retry_backoff: float,
) -> dict[str, Any]:
    started = time.monotonic()
    payload = {
        "session_id": f"{session_prefix}{case['session_id']}",
        "query": case["query"],
        "eval_debug": True,
    }
    last_error: Exception | None = None
    response: httpx.Response | None = None
    for attempt in range(retries + 1):
        try:
            response = client.post(f"{base_url.rstrip('/')}/api/v1/feishu/reply", json=payload)
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status_code = exc.response.status_code
            if status_code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise
        except httpx.RequestError as exc:
            last_error = exc
            if attempt >= retries:
                raise
        sleep_seconds = retry_backoff * (2**attempt)
        print(
            f"Retrying transient Feishu eval error in {sleep_seconds:.1f}s "
            f"(attempt {attempt + 1}/{retries})",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(sleep_seconds)
    if response is None:
        raise RuntimeError("Feishu eval request did not return a response") from last_error
    latency_ms = round((time.monotonic() - started) * 1000, 2)
    data = response.json()
    data["latency_ms"] = latency_ms
    return data


def _evaluate_rules(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("answer") or "")
    expected_contains = _as_list(case.get("expected_contains"))
    expected_not_contains = _as_list(case.get("expected_not_contains"))
    contains_ok, missing_terms = _contains_all(answer, expected_contains)
    forbidden_ok, forbidden_hits = _contains_none(answer, expected_not_contains)
    sources = _as_list(response.get("sources"))
    case_type = str(case.get("type") or "")
    requires_source = bool(case.get("requires_source", case_type not in {"reset", "paper_search_no_result"}))
    source_ok = bool(sources) if requires_source else True
    passed = contains_ok and forbidden_ok and source_ok
    return {
        "rule_pass": passed,
        "contains_ok": contains_ok,
        "missing_terms": missing_terms,
        "forbidden_ok": forbidden_ok,
        "forbidden_hits": forbidden_hits,
        "requires_source": requires_source,
        "source_ok": source_ok,
        "memory_case": bool(case.get("memory_case", "memory" in case_type or "followup" in case_type)),
    }


def _init_ragas_models(args: argparse.Namespace):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    api_key = os.getenv(args.qwen_api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{args.qwen_api_key_env} is required unless --skip-ragas is used")

    llm = ChatOpenAI(
        model=args.judge_model,
        api_key=api_key,
        base_url=args.qwen_base_url,
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    embeddings = OpenAIEmbeddings(
        model=args.embedding_model,
        api_key=api_key,
        base_url=args.qwen_base_url,
        skip_empty=True,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def _run_ragas(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not records:
        return []

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

    evaluator_llm, evaluator_embeddings = _init_ragas_models(args)
    dataset = Dataset.from_list(
        [
            {
                "user_input": row["query"],
                "response": row["answer"],
                "retrieved_contexts": row["contexts"],
                "reference": row.get("reference_answer") or "",
            }
            for row in records
        ]
    )
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    return result.to_pandas().to_dict("records")


def _normalise_ragas_scores(raw_rows: list[dict[str, Any]], case_ids: list[str]) -> dict[str, dict[str, float]]:
    by_case: dict[str, dict[str, float]] = {}
    aliases = {
        "faithfulness": ("faithfulness",),
        "response_relevancy": ("answer_relevancy", "response_relevancy"),
        "context_precision": ("llm_context_precision_without_reference", "context_precision"),
    }
    for case_id, raw in zip(case_ids, raw_rows):
        scores: dict[str, float] = {}
        for normalized_name, possible_names in aliases.items():
            for name in possible_names:
                if name in raw:
                    try:
                        value = float(raw[name])
                    except (TypeError, ValueError):
                        value = float("nan")
                    scores[normalized_name] = value
                    break
        by_case[case_id] = scores
    return by_case


def _summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    rule_passes = sum(1 for item in results if item["rule_pass"])
    source_required = [item for item in results if item["requires_source"]]
    source_passes = sum(1 for item in source_required if item["source_ok"])
    memory_cases = [item for item in results if item["memory_case"]]
    memory_passes = sum(1 for item in memory_cases if item["rule_pass"])
    latencies = [float(item["latency_ms"]) for item in results]

    ragas_values: dict[str, list[float]] = {"faithfulness": [], "response_relevancy": [], "context_precision": []}
    for item in results:
        for key in ragas_values:
            value = item.get("ragas", {}).get(key)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                ragas_values[key].append(float(value))

    return {
        "total": total,
        "pass_rate": rule_passes / total if total else 0.0,
        "source_rate": source_passes / len(source_required) if source_required else None,
        "memory_pass_rate": memory_passes / len(memory_cases) if memory_cases else None,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "ragas": {key: _safe_mean(values) for key, values in ragas_values.items()},
        "thresholds": PASS_THRESHOLDS,
    }


def _write_reports(results: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"feishu_ragas_{timestamp}.json"
    latest_md_path = output_dir / "feishu_ragas_latest.md"

    payload = {"summary": summary, "results": results}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Feishu RAG Ragas Evaluation",
        "",
        f"- total: {summary['total']}",
        f"- rule pass rate: {_percent(summary['pass_rate'])}",
        f"- source rate: {_percent(summary['source_rate'])}",
        f"- memory pass rate: {_percent(summary['memory_pass_rate'])}",
        f"- latency p50/p95: {summary['latency_p50_ms']} ms / {summary['latency_p95_ms']} ms",
        f"- ragas faithfulness: {summary['ragas']['faithfulness']}",
        f"- ragas response relevancy: {summary['ragas']['response_relevancy']}",
        f"- ragas context precision: {summary['ragas']['context_precision']}",
        "",
        "| case | type | rules | faithfulness | relevancy | context precision | latency ms | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        ragas = item.get("ragas", {})
        notes = []
        if item.get("missing_terms"):
            notes.append(f"missing={','.join(item['missing_terms'])}")
        if item.get("forbidden_hits"):
            notes.append(f"forbidden={','.join(item['forbidden_hits'])}")
        if item.get("requires_source") and not item.get("source_ok"):
            notes.append("missing_source")
        lines.append(
            "| {case_id} | {type} | {rules} | {faithfulness} | {relevancy} | {context_precision} | {latency} | {notes} |".format(
                case_id=item["case_id"],
                type=item.get("type", ""),
                rules="pass" if item["rule_pass"] else "fail",
                faithfulness=ragas.get("faithfulness", "n/a"),
                relevancy=ragas.get("response_relevancy", "n/a"),
                context_precision=ragas.get("context_precision", "n/a"),
                latency=item["latency_ms"],
                notes="; ".join(notes),
            )
        )

    latest_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, latest_md_path


def main() -> int:
    _load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(description="Evaluate /api/v1/feishu/reply with lightweight rules and Ragas")
    parser.add_argument("--dataset", type=Path, default=Path("eval/feishu_ragas_smoke.jsonl"))
    parser.add_argument("--base-url", default=os.getenv("FEISHU_EVAL_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-ragas", action="store_true", help="Only run deterministic rule checks")
    parser.add_argument("--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", "qwen3.5-plus"))
    parser.add_argument("--embedding-model", default=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-v4"))
    parser.add_argument("--qwen-base-url", default=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--qwen-api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--session-prefix", default="", help="Prefix session IDs to avoid persisted conversation collisions")
    parser.add_argument("--request-retries", type=int, default=3, help="Retries for transient Feishu API 429/5xx errors")
    parser.add_argument("--retry-backoff", type=float, default=20.0, help="Initial retry backoff in seconds")
    args = parser.parse_args()
    session_prefix = args.session_prefix or f"ragas_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"

    cases = _load_jsonl(args.dataset)
    results: list[dict[str, Any]] = []
    ragas_inputs: list[dict[str, Any]] = []
    ragas_case_ids: list[str] = []

    with httpx.Client(timeout=args.timeout) as client:
        for index, case in enumerate(cases, 1):
            case_id = str(case.get("case_id") or f"case_{index:03d}")
            print(f"[{index}/{len(cases)}] Running {case_id}", file=sys.stderr, flush=True)
            response = _call_feishu_reply(
                client,
                args.base_url,
                case,
                session_prefix,
                retries=args.request_retries,
                retry_backoff=args.retry_backoff,
            )
            rules = _evaluate_rules(case, response)
            result = {
                "case_id": case_id,
                "type": case.get("type", ""),
                "query": case["query"],
                "answer": response.get("answer", ""),
                "contexts": response.get("contexts") or [],
                "sources": response.get("sources") or [],
                "intent": response.get("intent"),
                "rewritten_query": response.get("rewritten_query"),
                "route": response.get("route"),
                "latency_ms": response["latency_ms"],
                **rules,
                "ragas": {},
            }
            results.append(result)

            if not args.skip_ragas and result["contexts"]:
                ragas_case_ids.append(case_id)
                ragas_inputs.append(
                    {
                        "case_id": case_id,
                        "query": case["query"],
                        "answer": response.get("answer", ""),
                        "contexts": response.get("contexts") or [],
                        "reference_answer": case.get("reference_answer") or "",
                    }
                )

    if not args.skip_ragas and ragas_inputs:
        raw_ragas = _run_ragas(ragas_inputs, args)
        scores_by_case = _normalise_ragas_scores(raw_ragas, ragas_case_ids)
        for item in results:
            item["ragas"] = scores_by_case.get(item["case_id"], {})

    summary = _summarise(results)
    json_path, md_path = _write_reports(results, summary, args.output_dir)
    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {md_path}")
    print(f"Rule pass rate: {_percent(summary['pass_rate'])}")
    print(f"Ragas: {summary['ragas']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
