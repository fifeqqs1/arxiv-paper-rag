from pydantic import BaseModel, Field


class FeishuLocalReplyRequest(BaseModel):
    """Request model for the local Feishu conversation endpoint."""

    session_id: str = Field(..., description="Conversation key, typically MortyClaw thread_id", min_length=1, max_length=200)
    query: str = Field(..., description="Original user query", min_length=1, max_length=4000)
    eval_debug: bool = Field(False, description="Return retrieval contexts and routing metadata for offline evaluation")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "local_geek_master",
                "query": "给我推荐一篇UAV论文",
            }
        }


class FeishuLocalReplyResponse(BaseModel):
    """Response model for the local Feishu conversation endpoint."""

    session_id: str = Field(..., description="Conversation key used for memory")
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Final answer produced by the Feishu conversation logic")
    contexts: list[str] | None = Field(None, description="Retrieved contexts used for Ragas evaluation")
    sources: list[str] | None = Field(None, description="Normalized source URLs used by the response")
    intent: str | None = Field(None, description="Feishu intent selected for this turn")
    rewritten_query: str | None = Field(None, description="Query sent to the RAG layer after Feishu rewriting")
    route: str | None = Field(None, description="RAG route selected by Feishu")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "local_geek_master",
                "query": "给我推荐一篇UAV论文",
                "answer": "我在当前已索引的论文库里找到了 1 篇相关论文：...",
            }
        }
