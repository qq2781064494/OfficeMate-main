"""Milvus 向量库适配层。

目标：
1. 替换主知识库、benchmark、本地题库原来的 Chroma 存储
2. 对上层继续暴露 split / embed / search / delete 这些高层接口
3. 搜索结果保持 `(document_like, score)` 结构，尽量不影响现有 RAG 代码
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, local

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
from services.model_provider import ModelProviderFactory
from utils.log_tool import get_logger


logger = get_logger("milvus_backend")


@dataclass
class MilvusPageDocument:
    page_content: str
    metadata: dict


class MilvusVectorBackend:
    _write_lock = Lock()

    def __init__(
        self,
        *,
        collection_name: str,
        knowledge_scope: str,
        knowledge_key: str = "",
        embedding_factory=None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_split_char_number: int | None = None,
    ):
        self.collection_name = collection_name
        self.knowledge_scope = knowledge_scope
        self.knowledge_key = knowledge_key
        self._embedding_factory = embedding_factory or self._default_embedding_factory
        self.embedding = self._embedding_factory()
        self._thread_local = local()
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.max_split_char_number = max_split_char_number or config.max_split_char_number
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )
        self._client = self._build_client()
        self._ensure_collection()

    def split_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text) if len(text) > self.max_split_char_number else [text]

    def build_chunk_payload(self, document_id: str, chunks: list[str], metadata: dict) -> tuple[list[dict], list[str]]:
        rows = []
        ids = []
        for index, chunk in enumerate(chunks):
            chunk_id = f"{document_id}-{index}"
            ids.append(chunk_id)
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "knowledge_scope": self.knowledge_scope,
                    "knowledge_key": self.knowledge_key,
                    "title": str(metadata.get("title", "")),
                    "category": str(metadata.get("category", "")),
                    "version": str(metadata.get("version", "")),
                    "file_name": str(metadata.get("file_name", "")),
                    "chunk_index": index,
                    "content": chunk,
                    "source_label": str(metadata.get("source_label", "")),
                    "file_hash": str(metadata.get("file_hash", "")),
                }
            )
        return rows, ids

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []
        embedding_client = self._get_thread_embedding_client()
        embeddings: list[list[float]] = []
        batch_size = max(1, config.embedding_batch_size)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            embeddings.extend(embedding_client.embed_documents(batch))
        return embeddings

    def add_document(self, document_id: str, text: str, metadata: dict) -> int:
        chunks = self.split_text(text)
        embeddings = self.embed_chunks(chunks)
        return self.add_embeddings(document_id, chunks, embeddings, metadata)

    def add_embeddings(self, document_id: str, chunks: list[str], embeddings: list[list[float]], metadata: dict) -> int:
        rows, _ = self.build_chunk_payload(document_id, chunks, metadata)
        for row, vector in zip(rows, embeddings):
            row["embedding"] = vector
        with self._write_lock:
            self._client.upsert(collection_name=self.collection_name, data=rows)
            self._ensure_collection_loaded()
        logger.info(
            "milvus_backend add_embeddings | collection=%s | document_id=%s | chunk_count=%s",
            self.collection_name,
            document_id,
            len(rows),
        )
        return len(rows)

    def delete_document(self, document_id: str, chunk_count: int = 0) -> None:
        self._ensure_collection_loaded()
        expr = f'document_id == "{document_id}"'
        self._client.delete(collection_name=self.collection_name, filter=expr)
        logger.info(
            "milvus_backend delete_document | collection=%s | document_id=%s | chunk_count=%s",
            self.collection_name,
            document_id,
            chunk_count,
        )

    def search(self, query: str, limit: int, category: str = "全部") -> list[tuple[MilvusPageDocument, float]]:
        try:
            self._ensure_collection_loaded()
            query_embedding = self._get_thread_embedding_client().embed_query(query)
            filter_parts = [f'knowledge_scope == "{self.knowledge_scope}"']
            if self.knowledge_key:
                filter_parts.append(f'knowledge_key == "{self.knowledge_key}"')
            if category != "全部":
                filter_parts.append(f'category == "{category}"')
            results = self._client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=limit,
                output_fields=[
                    "document_id",
                    "knowledge_scope",
                    "knowledge_key",
                    "title",
                    "category",
                    "version",
                    "file_name",
                    "chunk_index",
                    "content",
                    "source_label",
                    "file_hash",
                ],
                filter=" and ".join(filter_parts),
            )
            rows: list[tuple[MilvusPageDocument, float]] = []
            for hit in results[0] if results else []:
                entity = hit.get("entity", {})
                rows.append(
                    (
                        MilvusPageDocument(
                            page_content=entity.get("content", ""),
                            metadata={
                                "document_id": entity.get("document_id", ""),
                                "knowledge_scope": entity.get("knowledge_scope", ""),
                                "knowledge_key": entity.get("knowledge_key", ""),
                                "title": entity.get("title", ""),
                                "category": entity.get("category", ""),
                                "version": entity.get("version", ""),
                                "file_name": entity.get("file_name", ""),
                                "chunk_index": entity.get("chunk_index", 0),
                                "source_label": entity.get("source_label", ""),
                                "file_hash": entity.get("file_hash", ""),
                            },
                        ),
                        float(hit.get("distance", hit.get("score", 0.0))),
                    )
                )
            return rows
        except Exception:
            logger.exception(
                "milvus_backend search_failed | collection=%s | query=%s | category=%s",
                self.collection_name,
                query,
                category,
            )
            return []

    def reset_scope(self) -> None:
        filter_parts = [f'knowledge_scope == "{self.knowledge_scope}"']
        if self.knowledge_key:
            filter_parts.append(f'knowledge_key == "{self.knowledge_key}"')
        self._client.delete(collection_name=self.collection_name, filter=" and ".join(filter_parts))

    def _build_client(self):
        from pymilvus import MilvusClient

        uri = f"http://{config.milvus_host}:{config.milvus_port}"
        token = None
        if config.milvus_user:
            token = f"{config.milvus_user}:{config.milvus_password}"
        kwargs = {"uri": uri, "db_name": config.milvus_db_name}
        if token:
            kwargs["token"] = token
        return MilvusClient(**kwargs)

    def _ensure_collection(self) -> None:
        if self._client.has_collection(collection_name=self.collection_name):
            self._ensure_vector_index()
            return
        from pymilvus import DataType

        dimension = len(self.embedding.embed_query("dimension probe"))
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="knowledge_scope", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="knowledge_key", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="version", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source_label", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="file_hash", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level="Strong",
            metric_type="COSINE",
            index_params=index_params,
        )
        self._ensure_collection_loaded()
        logger.info(
            "milvus_backend create_collection | collection=%s | dimension=%s",
            self.collection_name,
            dimension,
        )

    def _ensure_vector_index(self) -> None:
        try:
            indexes = self._client.list_indexes(collection_name=self.collection_name)
            if indexes:
                return
        except Exception:
            logger.exception("milvus_backend list_indexes_failed | collection=%s", self.collection_name)

        try:
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            self._client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
            )
            logger.info("milvus_backend create_index | collection=%s", self.collection_name)
        except Exception:
            logger.exception("milvus_backend create_index_failed | collection=%s", self.collection_name)

    def _ensure_collection_loaded(self) -> None:
        import time

        try:
            self._ensure_vector_index()
            self._client.load_collection(collection_name=self.collection_name)
            for _ in range(20):
                state = self._client.get_load_state(collection_name=self.collection_name)
                state_name = str(state.get("state", "")).lower() if isinstance(state, dict) else str(state).lower()
                if "loaded" in state_name:
                    return
                time.sleep(0.2)
        except Exception:
            logger.exception("milvus_backend load_collection_failed | collection=%s", self.collection_name)

    def _get_thread_embedding_client(self):
        if not hasattr(self._thread_local, "embedding"):
            self._thread_local.embedding = self._embedding_factory()
        return self._thread_local.embedding

    def _default_embedding_factory(self):
        return ModelProviderFactory.create_embedding_provider().build_embedding_client(
            check_embedding_ctx_length=False,
        )
