import re

ARXIV_CITATION_RE = re.compile(
    r"\s*\[(?:arxiv|arXiv|ARXIV)\s*:\s*([a-z-]+/\d{7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)\]"
)
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)]|[（(]?\d+[）)])\s+")


def compact_repeated_single_paper_citations(answer: str, min_repetitions: int = 2) -> str:
    """Collapse noisy repeated citations when an answer only cites one paper."""
    if not answer:
        return answer

    matches = list(ARXIV_CITATION_RE.finditer(answer))
    if len(matches) < min_repetitions:
        return answer

    normalized_ids = {match.group(1).strip().lower() for match in matches}
    if len(normalized_ids) != 1:
        return answer

    citation = f"[arXiv:{matches[0].group(1).strip()}]"
    first_line = _first_non_empty_line(answer)
    if first_line and citation in first_line and not LIST_ITEM_RE.match(first_line):
        return _cleanup_citation_spacing(_remove_repeated_citations_after_first(answer)).strip()

    stripped_answer = _cleanup_citation_spacing(ARXIV_CITATION_RE.sub("", answer)).strip()
    if not stripped_answer:
        return f"以下内容基于 {citation}。"

    return f"以下内容基于 {citation}。\n\n{stripped_answer}"


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _remove_repeated_citations_after_first(answer: str) -> str:
    pieces: list[str] = []
    last_index = 0
    kept_first = False

    for match in ARXIV_CITATION_RE.finditer(answer):
        pieces.append(answer[last_index : match.start()])
        if not kept_first:
            pieces.append(match.group(0))
            kept_first = True
        last_index = match.end()

    pieces.append(answer[last_index:])
    return "".join(pieces)


def _cleanup_citation_spacing(text: str) -> str:
    text = re.sub(r"[ \t]+([。！？；：，、,.!?;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines())
