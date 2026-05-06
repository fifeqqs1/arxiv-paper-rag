from typing import Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""

    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    overlap_with_previous: int
    overlap_with_next: int
    section_title: Optional[str] = None
    section_path: list[str] = Field(default_factory=list)
    section_level: int = 1
    section_type: Optional[str] = None


class TextChunk(BaseModel):
    """A chunk of text with metadata."""

    text: str
    metadata: ChunkMetadata
    arxiv_id: str
    paper_id: str
