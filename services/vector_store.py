"""向量库封装层。

这个模块负责三件事：
1. 文本切片
2. 文本向量化
3. 相似度检索

它把 LangChain + Chroma 的细节藏起来，对上层只暴露 add/search/delete。
"""

from threading import Lock, local

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
from services.model_provider import ModelProviderFactory
from utils.log_tool import get_logger


logger = get_logger("vector_store")


class OfficeMateVectorStore:
    _write_lock = Lock()

    def __init__(self):
        # embedding 模型把文本转换成向量，供后续相似度检索使用。
        self.embedding = self._build_embedding_client()
        self._thread_local = local()
        # splitter 负责把长文档切成较小片段，避免一次入库文本过长。
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )
        # Chroma 是本地持久化向量数据库，数据会写到 storage/chroma_db。
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )
        logger.info(
            "vector_store initialized | embedding_model=%s | collection=%s | persist_directory=%s",
            config.embedding_model_name,
            config.collection_name,
            config.persist_directory,
        )

    def add_document(self, document_id, text, metadata):
        """把一篇文档切片后写入向量库，并返回片段数量。"""
        # 短文本直接入库，长文本才切片，避免不必要的分块。
        chunks = self.split_text(text)
        metadatas, ids = self.build_chunk_payload(document_id, chunks, metadata)
        with self._write_lock:
            self.vector_store.add_texts(chunks, metadatas=metadatas, ids=ids)
        logger.info(
            "vector_store add_document | document_id=%s | chunk_count=%s | title=%s | category=%s",
            document_id,
            len(chunks),
            metadata.get("title", ""),
            metadata.get("category", ""),
        )
        return len(chunks)

    def split_text(self, text):
        """把文本切成适合入库的片段列表。"""
        return self.splitter.split_text(text) if len(text) > config.max_split_char_number else [text]

    def build_chunk_payload(self, document_id, chunks, metadata):
        """构造 Chroma 入库所需的 metadata 和 ids。"""
        metadatas = []
        ids = []
        for index, _ in enumerate(chunks):
            metadatas.append({**metadata, "document_id": document_id, "chunk_index": index})
            ids.append(f"{document_id}-{index}")
        return metadatas, ids

    def embed_chunks(self, chunks):
        """把片段列表转成向量；支持受控并发调用。"""
        if not chunks:
            return []
        embeddings = []
        embedding_client = self._get_thread_embedding_client()
        batch_size = max(1, config.embedding_batch_size)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            embeddings.extend(embedding_client.embed_documents(batch))
        return embeddings

    def add_embeddings(self, document_id, chunks, embeddings, metadata):
        """使用预先计算好的 embeddings 串行写入 Chroma。"""
        metadatas, ids = self.build_chunk_payload(document_id, chunks, metadata)
        collection = getattr(self.vector_store, "_collection", None)
        with self._write_lock:
            if collection and embeddings:
                upsert = getattr(collection, "upsert", None)
                add = getattr(collection, "add", None)
                writer = upsert or add
                if writer:
                    writer(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=chunks,
                    )
                    logger.info(
                        "vector_store add_embeddings | document_id=%s | chunk_count=%s | title=%s | category=%s",
                        document_id,
                        len(chunks),
                        metadata.get("title", ""),
                        metadata.get("category", ""),
                    )
                    return len(chunks)
            self.vector_store.add_texts(chunks, metadatas=metadatas, ids=ids)
        logger.info(
            "vector_store add_embeddings_fallback | document_id=%s | chunk_count=%s | title=%s | category=%s",
            document_id,
            len(chunks),
            metadata.get("title", ""),
            metadata.get("category", ""),
        )
        return len(chunks)

    def _get_thread_embedding_client(self):
        """为每个工作线程缓存一份 embedding client，减少并发共享风险。"""
        if not hasattr(self._thread_local, "embedding"):
            self._thread_local.embedding = self._build_embedding_client()
        return self._thread_local.embedding

    def _build_embedding_client(self):
        """创建本地 oMLX 的 embedding client。"""
        return ModelProviderFactory.create_embedding_provider().build_embedding_client(
            check_embedding_ctx_length=False,
        )

    def delete_document(self, document_id, chunk_count=0):
        """按 document_id 删除整篇文档对应的所有向量片段。"""
        chunk_total = int(chunk_count or 0)
        if chunk_total > 0:
            # 如果已知片段数，就直接构造片段 ID 精确删除，效率更高。
            ids = [f"{document_id}-{index}" for index in range(chunk_total)]
            self.vector_store.delete(ids=ids)
            logger.info("vector_store delete_document_by_ids | document_id=%s | chunk_count=%s", document_id, chunk_total)
            return
        # 如果片段数未知，就退化成按 metadata 条件删除。
        self.vector_store.delete(where={"document_id": document_id})
        logger.info("vector_store delete_document_by_filter | document_id=%s", document_id)

    def search(self, query, category="全部", limit=None):
        """执行带可选分类过滤的相似度检索。"""
        filters = None if category == "全部" else {"category": category}
        limit = limit or config.similarity_threshold
        try:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=limit,
                filter=filters,
            )
            logger.info(
                "vector_store search | query=%s | category=%s | limit=%s | result_count=%s",
                query,
                category,
                limit,
                len(results),
            )
            return results
        except Exception:
            # 检索异常时返回空列表，让上层按“无依据”逻辑兜底。
            logger.exception("vector_store search_failed | query=%s | category=%s | limit=%s", query, category, limit)
            return []
