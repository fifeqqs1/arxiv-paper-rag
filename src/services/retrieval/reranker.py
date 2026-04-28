import math
import re
from dataclasses import dataclass
from typing import Any

TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


@dataclass(frozen=True)
class ScoredHit:
    score: float
    original_index: int
    hit: dict[str, Any]


class Reranker:
    """Lightweight local reranker with section-aware boosts.

    This is deliberately dependency-free. It reranks candidates already returned by
    OpenSearch, so it can be replaced by a cross-encoder later without changing API flow.
    """

    def rerank(
        self,
        *,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int,
        section_types: list[str] | None = None,
        subqueries: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not hits or top_k <= 0:
            return []

        section_types = section_types or []
        query_tokens = self._tokens(" ".join([query, *(subqueries or [])]))
        scored = [
            ScoredHit(
                score=self._score_hit(hit, query_tokens, section_types),
                original_index=index,
                hit=hit,
            )
            for index, hit in enumerate(self._dedupe_hits(hits))
        ]
        scored.sort(key=lambda item: (-item.score, item.original_index))
        return [item.hit for item in scored[:top_k]]

    def _score_hit(self, hit: dict[str, Any], query_tokens: set[str], section_types: list[str]) -> float:
        base_score = float(hit.get("score") or 0.0)
        text = " ".join(
            str(hit.get(field, "") or "")
            for field in ("chunk_text", "section_title", "section_type", "title", "abstract")
        )
        text_tokens = self._tokens(text)
        overlap = len(query_tokens & text_tokens)
        overlap_score = overlap / math.sqrt(max(len(text_tokens), 1))

        section_score = 0.0
        hit_section_type = str(hit.get("section_type", "") or "").lower()
        if hit_section_type and hit_section_type in section_types:
            section_index = section_types.index(hit_section_type)
            section_score += 7.0 if section_index == 0 else 3.0

        section_text = " ".join(
            [
                str(hit.get("section_title", "") or ""),
                " ".join(str(item) for item in hit.get("section_path", []) or []),
            ]
        ).lower()
        for section_type in section_types:
            if section_type in section_text:
                section_score += 1.0

        # Favor retrieval engine ordering, but let strong section/query evidence reorder candidates.
        return base_score + (overlap_score * 8.0) + section_score

    def _dedupe_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            key = str(hit.get("chunk_id") or "")
            if not key:
                key = f"{hit.get('arxiv_id')}:{hit.get('chunk_index')}:{hash(str(hit.get('chunk_text', '')))}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1}
