"""FastAPI 请求与响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatAskRequest(BaseModel):
    question: str
    session_id: str
    category: str = "全部"
    use_history: bool = True
    persist_log: bool = True
    include_references: bool = True
    enable_query_rewrite: bool = True
    enable_rerank: bool = True
    reference_limit: int | None = None


class FeedbackRequest(BaseModel):
    qa_log_id: str
    rating: str
    comment: str = ""
    session_id: str = ""


class SeedDocumentsRequest(BaseModel):
    run_async: bool = True


class BenchmarkBuildCorpusRequest(BaseModel):
    subset: str
    splits: list[str] = Field(default_factory=lambda: ["test"])
    rebuild: bool = False


class BenchmarkBuildIndexRequest(BaseModel):
    subset: str
    rebuild: bool = False
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    max_split_char_number: int | None = None


class BenchmarkRunRequest(BaseModel):
    subset: str
    split: str = "test"
    retriever_strategy: str = "hybrid"
    top_k: int = 5
    question_limit: int = 50
    enable_query_rewrite: bool = True
    enable_ragas: bool = True
    enable_faithfulness: bool = True
    enable_rerank: bool = True
    rebuild_corpus: bool = False
    rebuild_index: bool = False
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    max_split_char_number: int | None = None


class LocalEvalKnowledgeBaseCreateRequest(BaseModel):
    knowledge_base_name: str
    rebuild: bool = False
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    max_split_char_number: int | None = None


class LocalEvalRunRequest(BaseModel):
    knowledge_base_id: str
    knowledge_base_name: str
    dataset_key: str
    dataset_label: str
    sample_path: str
    retriever_strategy: str = "hybrid"
    top_k: int = 5
    question_limit: int = 0
    selected_question_ids: list[int] = Field(default_factory=list)
    enable_query_rewrite: bool = True
    enable_ragas: bool = True
    enable_faithfulness: bool = True
    enable_rerank: bool = True
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    max_split_char_number: int | None = None


class APIMessage(BaseModel):
    message: str
    data: Any | None = None
