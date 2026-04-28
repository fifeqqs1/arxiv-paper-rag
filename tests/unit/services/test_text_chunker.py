from src.services.indexing.text_chunker import TextChunker


def test_chunker_preserves_numbered_section_path_and_type():
    chunker = TextChunker(chunk_size=120, overlap_size=20, min_chunk_size=10)
    sections = [
        {"title": "1 Introduction", "content": "intro " * 120},
        {"title": "2 Method", "content": "method " * 120},
        {"title": "2.1 Policy Architecture", "content": "architecture " * 120},
        {"title": "3 Experiments", "content": "experiment baseline result " * 50},
    ]

    chunks = chunker.chunk_paper(
        title="Test Paper",
        abstract="abstract text",
        full_text="",
        arxiv_id="2601.00001v1",
        paper_id="paper-1",
        sections=sections,
    )

    policy_chunk = next(chunk for chunk in chunks if chunk.metadata.section_title == "2.1 Policy Architecture")
    experiment_chunk = next(chunk for chunk in chunks if chunk.metadata.section_title == "3 Experiments")

    assert policy_chunk.metadata.section_path == ["2 Method", "2.1 Policy Architecture"]
    assert policy_chunk.metadata.section_level == 2
    assert policy_chunk.metadata.section_type == "method"
    assert experiment_chunk.metadata.section_type == "experiment"
    assert "Section path: 2 Method > 2.1 Policy Architecture" in policy_chunk.text
