"""RAGBench 全局知识库构建与向量库管理。"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
from services.model_provider import ModelProviderFactory
from utils.log_tool import get_logger


logger = get_logger("benchmark_store")


@dataclass
class BenchmarkSubsetInfo:
    subset: str
    available_splits: list[str]
    question_count: int


@dataclass
class BenchmarkChunkConfig:
    chunk_size: int = config.benchmark_chunk_size
    chunk_overlap: int = config.benchmark_chunk_overlap
    max_split_char_number: int = config.benchmark_max_split_char_number

    def to_dict(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_split_char_number": self.max_split_char_number,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "BenchmarkChunkConfig":
        data = data or {}
        return cls(
            chunk_size=int(data.get("chunk_size", config.benchmark_chunk_size)),
            chunk_overlap=int(data.get("chunk_overlap", config.benchmark_chunk_overlap)),
            max_split_char_number=int(data.get("max_split_char_number", config.benchmark_max_split_char_number)),
        )


class BenchmarkVectorStore:
    """服务于单个 subset 的独立向量库。"""

    _write_lock = Lock()

    def __init__(self, subset: str, chunk_config: BenchmarkChunkConfig | None = None):
        self.subset = subset
        self.chunk_config = chunk_config or BenchmarkChunkConfig()
        self.persist_directory = config.BENCHMARK_CHROMA_DIR / subset
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding = self._build_embedding_client()
        self._thread_local = local()
        self._client = self._build_client()
        self._collection = self._get_or_create_collection()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_config.chunk_size,
            chunk_overlap=self.chunk_config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def reset(self) -> None:
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = self._build_client()
        self._collection = self._get_or_create_collection()

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

    def search(self, query: str, limit: int) -> list[tuple[object, float]]:
        try:
            query_embedding = self._get_thread_embedding_client().embed_query(query)
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            rows = []
            for document, metadata, distance in zip(documents, metadatas, distances):
                rows.append(
                    (
                        _BenchmarkPageDocument(page_content=document, metadata=metadata or {}),
                        float(distance),
                    )
                )
            return rows
        except Exception:
            logger.exception("benchmark_vector_store search_failed | subset=%s | query=%s", self.subset, query)
            return []

    def _build_client(self):
        from chromadb import PersistentClient

        return PersistentClient(path=str(self.persist_directory))

    def _get_or_create_collection(self):
        return self._client.get_or_create_collection(
            name=f"{config.benchmark_collection_prefix}_{self.subset}",
            metadata={"hnsw:space": "cosine"},
        )

    def _get_thread_embedding_client(self):
        if not hasattr(self._thread_local, "embedding"):
            self._thread_local.embedding = self._build_embedding_client()
        return self._thread_local.embedding

    def _build_embedding_client(self):
        return ModelProviderFactory.create_benchmark_embedding_provider().build_embedding_client(
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )


class BenchmarkCorpusStore:
    """负责 subset 语料构建、manifest 读取和独立向量库初始化。"""

    def __init__(self):
        config.ensure_runtime_dirs()

    def list_available_subsets(self) -> list[BenchmarkSubsetInfo]:
        infos: list[BenchmarkSubsetInfo] = []
        for subset_dir in sorted(config.RAGBENCH_DIR.iterdir()) if config.RAGBENCH_DIR.exists() else []:
            if not subset_dir.is_dir():
                continue
            eval_files = sorted(subset_dir.glob("*_officemate_eval.json"))
            available_splits = [path.name.replace("_officemate_eval.json", "") for path in eval_files]
            question_count = 0
            if eval_files:
                try:
                    question_count = len(json.loads(eval_files[0].read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    question_count = 0
            infos.append(BenchmarkSubsetInfo(subset=subset_dir.name, available_splits=available_splits, question_count=question_count))
        return infos

    def load_eval_samples(self, subset: str, split: str) -> list[dict]:
        path = config.RAGBENCH_DIR / subset / f"{split}_officemate_eval.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def load_raw_samples(self, subset: str, split: str) -> list[dict]:
        path = config.RAGBENCH_DIR / subset / f"{split}.jsonl"
        samples = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def build_subset_corpus(self, subset: str, splits: list[str], rebuild: bool = False) -> dict:
        subset_root = config.BENCHMARK_CORPUS_DIR / subset
        docs_dir = subset_root / "docs"
        manifest_path = subset_root / "manifest.json"

        if rebuild and subset_root.exists():
            shutil.rmtree(subset_root)

        if manifest_path.exists() and not rebuild:
            records = json.loads(manifest_path.read_text(encoding="utf-8"))
            return {
                "subset": subset,
                "document_count": len(records),
                "manifest_path": str(manifest_path.relative_to(config.BASE_DIR)),
                "status": "cached",
            }

        docs_dir.mkdir(parents=True, exist_ok=True)
        deduped: dict[str, dict] = {}

        for split in splits:
            for sample in self.load_raw_samples(subset, split):
                for document in sample.get("documents", []):
                    title = document.get("title", "untitled").strip() or "untitled"
                    passage = document.get("passage", "").strip()
                    content = f"Title: {title}\n\nPassage:\n{passage}".strip()
                    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
                    record = deduped.setdefault(
                        content_hash,
                        {
                            "document_id": f"{subset}_{content_hash[:12]}",
                            "title": title,
                            "content_hash": content_hash,
                            "source_splits": [],
                            "raw_path": "",
                            "file_name": f"{content_hash[:12]}.txt",
                        },
                    )
                    if split not in record["source_splits"]:
                        record["source_splits"].append(split)
                    if not record["raw_path"]:
                        file_path = docs_dir / record["file_name"]
                        file_path.write_text(content + "\n", encoding="utf-8")
                        record["raw_path"] = str(file_path.relative_to(config.BASE_DIR))

        records = sorted(deduped.values(), key=lambda item: item["document_id"])
        manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("benchmark_store build_subset_corpus | subset=%s | document_count=%s", subset, len(records))
        return {
            "subset": subset,
            "document_count": len(records),
            "manifest_path": str(manifest_path.relative_to(config.BASE_DIR)),
            "status": "built",
        }

    def load_corpus_manifest(self, subset: str) -> list[dict]:
        manifest_path = config.BENCHMARK_CORPUS_DIR / subset / "manifest.json"
        if not manifest_path.exists():
            return []
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def ensure_vector_index(
        self,
        subset: str,
        rebuild: bool = False,
        chunk_config: BenchmarkChunkConfig | None = None,
    ) -> dict:
        manifest = self.load_corpus_manifest(subset)
        if not manifest:
            raise ValueError(f"subset={subset} 还没有构建全局 benchmark 语料。")

        metadata_path = config.BENCHMARK_CHROMA_DIR / subset / "index_meta.json"
        if metadata_path.exists() and not rebuild:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            return {"subset": subset, "status": "cached", **meta}

        try:
            return self._build_vector_index_once(
                subset,
                manifest,
                metadata_path,
                rebuild=rebuild,
                chunk_config=chunk_config or BenchmarkChunkConfig(),
            )
        except Exception as exc:
            if "readonly database" not in str(exc).lower():
                raise
            logger.warning(
                "benchmark_store readonly_database_retry | subset=%s | error=%s",
                subset,
                exc,
            )
            return self._build_vector_index_once(
                subset,
                manifest,
                metadata_path,
                rebuild=True,
                chunk_config=chunk_config or BenchmarkChunkConfig(),
            )

    def _build_vector_index_once(
        self,
        subset: str,
        manifest: list[dict],
        metadata_path: Path,
        rebuild: bool,
        chunk_config: BenchmarkChunkConfig,
    ) -> dict:
        persist_directory = config.BENCHMARK_CHROMA_DIR / subset
        if rebuild and persist_directory.exists():
            shutil.rmtree(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

        vector_store = BenchmarkVectorStore(subset, chunk_config=chunk_config)

        chunk_total = 0
        for record in manifest:
            raw_path = config.BASE_DIR / record["raw_path"]
            text = raw_path.read_text(encoding="utf-8")
            chunk_total += vector_store.add_document(
                document_id=record["document_id"],
                text=text,
                metadata={
                    "title": record["title"],
                    "file_name": record["file_name"],
                    "category": subset,
                    "version": "ragbench",
                    "benchmark": "ragbench",
                    "subset": subset,
                },
            )

        meta = {
            "document_count": len(manifest),
            "chunk_count": chunk_total,
            "chunk_config": chunk_config.to_dict(),
        }
        metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("benchmark_store ensure_vector_index | subset=%s | chunk_count=%s", subset, chunk_total)
        return {"subset": subset, "status": "built", **meta}

    def get_vector_store(self, subset: str, chunk_config: BenchmarkChunkConfig | None = None) -> BenchmarkVectorStore:
        if chunk_config is None:
            metadata_path = config.BENCHMARK_CHROMA_DIR / subset / "index_meta.json"
            if metadata_path.exists():
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                chunk_config = BenchmarkChunkConfig.from_dict(meta.get("chunk_config"))
        return BenchmarkVectorStore(subset, chunk_config=chunk_config)


@dataclass
class _BenchmarkPageDocument:
    page_content: str
    metadata: dict
