"""本地题库测评的知识库实例注册与向量库管理。"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import config_data as config
from core.bootstrap import bootstrap_runtime
from core.db import session_scope
from models.entities import LocalEvalKnowledgeBaseEntity
from services.benchmark_store import BenchmarkChunkConfig
from services.model_provider import ModelProviderFactory
from utils.log_tool import get_logger
from vectorstores.milvus_backend import MilvusVectorBackend


logger = get_logger("local_eval_store")


@dataclass
class LocalEvalDatasetInfo:
    dataset_key: str
    dataset_label: str
    sample_path: Path
    question_count: int


@dataclass
class LocalEvalKnowledgeBaseInfo:
    knowledge_base_id: str
    knowledge_base_name: str
    document_count: int
    chunk_count: int
    chunk_config: dict
    created_at: str
    updated_at: str
    persist_directory: str
    manifest_path: str
    source_files: list[str]


@dataclass
class _LocalEvalPageDocument:
    page_content: str
    metadata: dict


class LocalEvalVectorStore:
    def __init__(
        self,
        knowledge_base_id: str,
        persist_directory: Path,
        chunk_config: BenchmarkChunkConfig | None = None,
    ):
        self.knowledge_base_id = knowledge_base_id
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.chunk_config = chunk_config or BenchmarkChunkConfig()
        self.backend = MilvusVectorBackend(
            collection_name=config.milvus_local_eval_collection,
            knowledge_scope="local_eval",
            knowledge_key=knowledge_base_id,
            embedding_factory=self._build_embedding_client,
            chunk_size=self.chunk_config.chunk_size,
            chunk_overlap=self.chunk_config.chunk_overlap,
            max_split_char_number=self.chunk_config.max_split_char_number,
        )

    def add_document(self, document_id: str, text: str, metadata: dict) -> int:
        return self.backend.add_document(document_id, text, metadata)

    def split_text(self, text: str) -> list[str]:
        return self.backend.split_text(text)

    def search(self, query: str, limit: int, category: str = "全部") -> list[tuple[object, float]]:
        return self.backend.search(query, limit=limit, category=category)

    def _build_embedding_client(self):
        return ModelProviderFactory.create_embedding_provider().build_embedding_client(
            check_embedding_ctx_length=False,
        )


class LocalEvalCorpusStore:
    def __init__(self):
        bootstrap_runtime()

    def list_available_datasets(self) -> list[LocalEvalDatasetInfo]:
        return [
            LocalEvalDatasetInfo(
                dataset_key="local_sample_220",
                dataset_label="220 全量题库",
                sample_path=config.EVALUATION_SAMPLE_PATH,
                question_count=self._count_samples(config.EVALUATION_SAMPLE_PATH),
            ),
            LocalEvalDatasetInfo(
                dataset_key="local_sample_complex_20",
                dataset_label="复杂 20 题",
                sample_path=config.COMPLEX_EVALUATION_SAMPLE_PATH,
                question_count=self._count_samples(config.COMPLEX_EVALUATION_SAMPLE_PATH),
            ),
        ]

    def list_knowledge_bases(self) -> list[dict]:
        records = self._read_registry()
        return sorted(records, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get_knowledge_base(self, knowledge_base_id: str) -> dict | None:
        for record in self._read_registry():
            if record.get("knowledge_base_id") == knowledge_base_id:
                return record
        return None

    def build_knowledge_base(
        self,
        *,
        knowledge_base_name: str,
        chunk_config: BenchmarkChunkConfig,
        rebuild: bool = False,
    ) -> dict:
        registry = self._read_registry()
        existing = next((item for item in registry if item.get("knowledge_base_name") == knowledge_base_name), None)
        if existing and not rebuild:
            raise ValueError(f"知识库名称已存在：{knowledge_base_name}。如需覆盖，请勾选重建当前知识库。")

        knowledge_base_id = existing["knowledge_base_id"] if existing else self._build_kb_id(knowledge_base_name)
        kb_root = config.LOCAL_EVAL_KB_DIR / knowledge_base_id
        docs_dir = kb_root / "docs"
        chroma_dir = kb_root / "chroma"
        manifest_path = kb_root / "manifest.json"
        index_meta_path = kb_root / "index_meta.json"

        if rebuild and kb_root.exists():
            shutil.rmtree(kb_root)

        kb_root.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        manifest_records = []
        source_files = []
        for item in config.SAMPLE_DOCS:
            source_path = config.SAMPLE_DOC_DIR / item["file_name"]
            target_path = docs_dir / item["file_name"]
            target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            source_files.append(item["file_name"])
            manifest_records.append(
                {
                    "document_id": f"{knowledge_base_id}_{item['file_name'].replace('.txt', '')}",
                    "title": item["title"],
                    "category": item["category"],
                    "version": item["version"],
                    "file_name": item["file_name"],
                    "raw_path": str(target_path.relative_to(config.BASE_DIR)),
                }
            )
        manifest_path.write_text(json.dumps(manifest_records, ensure_ascii=False, indent=2), encoding="utf-8")

        vector_store = LocalEvalVectorStore(
            knowledge_base_id=knowledge_base_id,
            persist_directory=chroma_dir,
            chunk_config=chunk_config,
        )
        chunk_total = 0
        for record in manifest_records:
            raw_path = config.BASE_DIR / record["raw_path"]
            text = raw_path.read_text(encoding="utf-8")
            chunk_total += vector_store.add_document(
                document_id=record["document_id"],
                text=text,
                metadata={
                    "title": record["title"],
                    "file_name": record["file_name"],
                    "category": record["category"],
                    "version": record["version"],
                    "benchmark": "local_eval",
                    "knowledge_base_id": knowledge_base_id,
                },
            )

        now = self._now()
        kb_record = {
            "knowledge_base_id": knowledge_base_id,
            "knowledge_base_name": knowledge_base_name,
            "document_count": len(manifest_records),
            "chunk_count": chunk_total,
            "chunk_config": chunk_config.to_dict(),
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
            "persist_directory": str(chroma_dir.relative_to(config.BASE_DIR)),
            "manifest_path": str(manifest_path.relative_to(config.BASE_DIR)),
            "source_files": source_files,
        }
        index_meta_path.write_text(json.dumps(kb_record, ensure_ascii=False, indent=2), encoding="utf-8")
        self._upsert_registry_record(kb_record)
        logger.info(
            "local_eval_store build_knowledge_base | knowledge_base_id=%s | knowledge_base_name=%s | chunk_count=%s",
            knowledge_base_id,
            knowledge_base_name,
            chunk_total,
        )
        return kb_record

    def load_corpus_manifest(self, knowledge_base_id: str) -> list[dict]:
        record = self.get_knowledge_base(knowledge_base_id)
        if not record:
            return []
        manifest_path = config.BASE_DIR / record["manifest_path"]
        if not manifest_path.exists():
            return []
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def get_vector_store(self, knowledge_base_id: str, chunk_config: BenchmarkChunkConfig | None = None) -> LocalEvalVectorStore:
        record = self.get_knowledge_base(knowledge_base_id)
        if not record:
            raise ValueError(f"未找到知识库：{knowledge_base_id}")
        if chunk_config is None:
            chunk_config = BenchmarkChunkConfig.from_dict(record.get("chunk_config"))
        persist_directory = config.BASE_DIR / record["persist_directory"]
        return LocalEvalVectorStore(
            knowledge_base_id=knowledge_base_id,
            persist_directory=persist_directory,
            chunk_config=chunk_config,
        )

    def load_eval_samples(self, sample_path: Path) -> list[dict]:
        return json.loads(sample_path.read_text(encoding="utf-8"))

    def suggest_knowledge_base_name(self, preset_label: str) -> str:
        base = _slugify(preset_label) or "sampledocs"
        return f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _count_samples(self, path: Path) -> int:
        try:
            return len(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return 0

    def _read_registry(self) -> list[dict]:
        with session_scope() as session:
            rows = (
                session.query(LocalEvalKnowledgeBaseEntity)
                .order_by(LocalEvalKnowledgeBaseEntity.updated_at.desc())
                .all()
            )
            return [self._row_to_record(row) for row in rows]

    def _upsert_registry_record(self, record: dict) -> None:
        with session_scope() as session:
            row = session.get(LocalEvalKnowledgeBaseEntity, record["knowledge_base_id"])
            if not row:
                row = LocalEvalKnowledgeBaseEntity(knowledge_base_id=record["knowledge_base_id"])
                session.add(row)
            row.knowledge_base_name = record["knowledge_base_name"]
            row.document_count = int(record.get("document_count", 0) or 0)
            row.chunk_count = int(record.get("chunk_count", 0) or 0)
            row.chunk_config_json = record.get("chunk_config", {})
            row.created_at = record.get("created_at", self._now())
            row.updated_at = record.get("updated_at", self._now())
            row.persist_directory = record.get("persist_directory", "")
            row.manifest_path = record.get("manifest_path", "")
            row.source_files_json = record.get("source_files", [])

    def _build_kb_id(self, knowledge_base_name: str) -> str:
        slug = _slugify(knowledge_base_name) or "sampledocs"
        return f"{slug}_{uuid4().hex[:8]}"

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _row_to_record(self, row: LocalEvalKnowledgeBaseEntity) -> dict:
        return {
            "knowledge_base_id": row.knowledge_base_id,
            "knowledge_base_name": row.knowledge_base_name,
            "document_count": row.document_count,
            "chunk_count": row.chunk_count,
            "chunk_config": row.chunk_config_json or {},
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "persist_directory": row.persist_directory,
            "manifest_path": row.manifest_path,
            "source_files": row.source_files_json or [],
        }


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", text.strip()).strip("_")
    return normalized[:64]
