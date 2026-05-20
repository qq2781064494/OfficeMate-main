"""主知识库向量存储封装。

重构后底层实现从 Chroma 切换为 Milvus，
但接口继续保持不变，避免影响上层文档服务与 RAG 检索链路。
"""

import config_data as config
from utils.log_tool import get_logger
from vectorstores.milvus_backend import MilvusVectorBackend


logger = get_logger("vector_store")


class OfficeMateVectorStore:
    def __init__(self):
        self.backend = MilvusVectorBackend(
            collection_name=config.milvus_main_collection,
            knowledge_scope="main",
            knowledge_key="",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_split_char_number=config.max_split_char_number,
        )
        logger.info(
            "vector_store initialized | embedding_model=%s | collection=%s | backend=milvus",
            config.embedding_model_name,
            config.milvus_main_collection,
        )

    def add_document(self, document_id, text, metadata):
        """把一篇文档切片后写入向量库，并返回片段数量。"""
        return self.backend.add_document(document_id, text, metadata)

    def split_text(self, text):
        """把文本切成适合入库的片段列表。"""
        return self.backend.split_text(text)

    def build_chunk_payload(self, document_id, chunks, metadata):
        """保留兼容方法，供旧代码复用。"""
        return self.backend.build_chunk_payload(document_id, chunks, metadata)

    def embed_chunks(self, chunks):
        """把片段列表转成向量；支持受控并发调用。"""
        return self.backend.embed_chunks(chunks)

    def add_embeddings(self, document_id, chunks, embeddings, metadata):
        """使用预先计算好的 embeddings 写入 Milvus。"""
        return self.backend.add_embeddings(document_id, chunks, embeddings, metadata)

    def delete_document(self, document_id, chunk_count=0):
        """按 document_id 删除整篇文档对应的所有向量片段。"""
        self.backend.delete_document(document_id, chunk_count)

    def search(self, query, category="全部", limit=None):
        """执行带可选分类过滤的相似度检索。"""
        limit = limit or config.similarity_threshold
        return self.backend.search(query, category=category, limit=limit)
