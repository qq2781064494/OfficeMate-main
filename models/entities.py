"""MySQL 持久化实体。

这些 ORM 模型对应 FastAPI 重构后的核心业务表。
首版先围绕“兼容现有业务字典结构”来设计字段，尽量减少上层服务改动。
"""

from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Integer, JSON, Text, String, func, Index
from sqlalchemy.orm import Mapped, mapped_column

from core.db import Base


class DocumentEntity(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(64))
    file_name: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(32))
    file_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    text_length: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="processing")
    source_label: Mapped[str] = mapped_column(String(64), default="")
    raw_path: Mapped[str] = mapped_column(String(1024), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    uploaded_at: Mapped[str] = mapped_column(String(32), index=True)
    updated_at: Mapped[str] = mapped_column(String(32), default="")


class QALogEntity(Base):
    __tablename__ = "qa_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), default="全部")
    question_type: Mapped[str] = mapped_column(String(64), default="")
    source_docs_json: Mapped[dict | list] = mapped_column(JSON, default=list)
    mode: Mapped[str] = mapped_column(String(64), default="chat")
    trace_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    decision_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), index=True)


class FeedbackLogEntity(Base):
    __tablename__ = "feedback_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    qa_log_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(128), default="")
    rating: Mapped[str] = mapped_column(String(32))
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(32), index=True)
    updated_at: Mapped[str] = mapped_column(String(32), index=True)


class TaskRunEntity(Base):
    __tablename__ = "task_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    progress_stage: Mapped[str] = mapped_column(String(128), default="")
    progress_message: Mapped[str] = mapped_column(Text, default="")
    payload_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    result_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(32), index=True)
    started_at: Mapped[str] = mapped_column(String(32), default="")
    finished_at: Mapped[str] = mapped_column(String(32), default="")


class BenchmarkRunEntity(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_type: Mapped[str] = mapped_column(String(64), index=True)
    subset: Mapped[str] = mapped_column(String(128), default="")
    split: Mapped[str] = mapped_column(String(64), default="")
    knowledge_base_id: Mapped[str] = mapped_column(String(128), default="")
    knowledge_base_name: Mapped[str] = mapped_column(String(255), default="")
    retriever_strategy: Mapped[str] = mapped_column(String(32), default="hybrid")
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    question_count: Mapped[int] = mapped_column(Integer, default=0)
    enable_query_rewrite: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_rerank: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_faithfulness: Mapped[bool] = mapped_column(Boolean, default=True)
    summary_json: Mapped[dict | list] = mapped_column(JSON, default=dict)
    created_at: Mapped[str] = mapped_column(String(32), index=True)


class BenchmarkRunDetailEntity(Base):
    __tablename__ = "benchmark_run_details"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    question_order: Mapped[int] = mapped_column(Integer, default=0)
    detail_json: Mapped[dict | list] = mapped_column(JSON, default=dict)

    __table_args__ = (
        Index("idx_benchmark_run_details_run_order", "run_id", "question_order"),
    )


class LocalEvalKnowledgeBaseEntity(Base):
    __tablename__ = "local_eval_knowledge_bases"

    knowledge_base_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    knowledge_base_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_config_json: Mapped[dict | list] = mapped_column(JSON, default=dict)
    created_at: Mapped[str] = mapped_column(String(32), index=True)
    updated_at: Mapped[str] = mapped_column(String(32), index=True)
    persist_directory: Mapped[str] = mapped_column(String(1024), default="")
    manifest_path: Mapped[str] = mapped_column(String(1024), default="")
    source_files_json: Mapped[list] = mapped_column(JSON, default=list)


class BenchmarkCorpusRegistryEntity(Base):
    __tablename__ = "benchmark_corpus_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subset: Mapped[str] = mapped_column(String(128), index=True)
    registry_type: Mapped[str] = mapped_column(String(32), index=True)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    manifest_path: Mapped[str] = mapped_column(String(1024), default="")
    chunk_config_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
