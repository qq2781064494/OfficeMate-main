"""本地题库测评的知识库实例注册与向量库管理。"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock, local
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
from services.benchmark_store import BenchmarkChunkConfig
from services.model_provider import ModelProviderFactory
from utils.log_tool import get_logger


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
    _write_lock = Lock()

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
        self._thread_local = local()
        self.embedding = self._build_embedding_client()
        self._client = self._build_client()
        self._collection = self._get_or_create_collection()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_config.chunk_size,
            chunk_overlap=self.chunk_config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def add_document(self, document_id: str, text: str, metadata: dict) -> int:
        chunks = self.split_text(text)
        embeddings = self._get_thread_embedding_client().embed_documents(chunks)
        metadatas = []
        ids = []
        for index, _ in enumerate(chunks):
            metadatas.append({**metadata, "document_id": document_id, "chunk_index": index})
            ids.append(f"{document_id}-{index}")
        with self._write_lock:
            self._collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        return len(chunks)

    def split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_config.max_split_char_number:
            return [text]
        return self.splitter.split_text(text)

    def search(self, query: str, limit: int, category: str = "全部") -> list[tuple[object, float]]:
        try:
            query_embedding = self._get_thread_embedding_client().embed_query(query)
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": limit,
                "include": ["documents", "metadatas", "distances"],
            }
            if category != "全部":
                query_kwargs["where"] = {"category": category}
            result = self._collection.query(**query_kwargs)
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            rows = []
            for document, metadata, distance in zip(documents, metadatas, distances):
                rows.append((_LocalEvalPageDocument(page_content=document, metadata=metadata or {}), float(distance)))
            return rows
        except Exception:
            logger.exception(
                "local_eval_vector_store search_failed | knowledge_base_id=%s | query=%s | category=%s",
                self.knowledge_base_id,
                query,
                category,
            )
            return []

    def _build_client(self):
        from chromadb import PersistentClient

        return PersistentClient(path=str(self.persist_directory))

    def _get_or_create_collection(self):
        return self._client.get_or_create_collection(
            name=f"{config.local_eval_collection_prefix}_{self.knowledge_base_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def _get_thread_embedding_client(self):
        if not hasattr(self._thread_local, "embedding"):
            self._thread_local.embedding = self._build_embedding_client()
        return self._thread_local.embedding

    def _build_embedding_client(self):
        return ModelProviderFactory.create_embedding_provider().build_embedding_client(
            check_embedding_ctx_length=False,
        )


class LocalEvalCorpusStore:
    def __init__(self):
        config.ensure_runtime_dirs()

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
        try:
            return json.loads(config.LOCAL_EVAL_KB_INDEX_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _upsert_registry_record(self, record: dict) -> None:
        records = [item for item in self._read_registry() if item.get("knowledge_base_id") != record["knowledge_base_id"]]
        records.append(record)
        config.LOCAL_EVAL_KB_INDEX_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_kb_id(self, knowledge_base_name: str) -> str:
        slug = _slugify(knowledge_base_name) or "sampledocs"
        return f"{slug}_{uuid4().hex[:8]}"

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", text.strip()).strip("_")
    return normalized[:64]
