"""retrieval 阶段实现。

这一层负责“把可能有用的证据找出来”。

项目里用了三种检索角色：
- `VectorRetriever`：偏语义相似，适合理解同义表达
- `BM25Retriever`：偏关键词匹配，适合制度术语、编号、材料名
- `HybridRetriever`：把两者合并，得到更稳的候选池
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import math
import re
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
from services.rag.planning import PlannedTask
from services.rag.query import QueryRewriteResult, QueryRewriter
from services.storage_service import JsonStorageService
from services.vector_store import OfficeMateVectorStore
from utils.log_tool import get_logger


vector_logger = get_logger("vector_retriever")
bm25_logger = get_logger("bm25_retriever")
hybrid_logger = get_logger("hybrid_retriever")
coordinator_logger = get_logger("retrieval_coordinator")


@dataclass
class RetrievalCandidate:
    """一条候选证据片段。"""

    document_id: str
    title: str
    category: str
    version: str
    file_name: str
    content: str
    metadata: Dict[str, object] = field(default_factory=dict)
    score: float = 0.0
    source: str = ""
    retrieval_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def unique_key(self) -> str:
        chunk_index = self.metadata.get("chunk_index", "")
        return f"{self.document_id}:{chunk_index}:{self.file_name}"


class BaseRetriever(ABC):
    """检索器统一接口。"""

    @abstractmethod
    def retrieve(
        self,
        rewrite_result: QueryRewriteResult,
        category: str = "全部",
        limit: int | None = None,
    ) -> List[RetrievalCandidate]:
        raise NotImplementedError


class VectorRetriever(BaseRetriever):
    """向量检索器。"""

    def __init__(self, vector_store: OfficeMateVectorStore | None = None):
        self.vector_store = vector_store or OfficeMateVectorStore()

    def retrieve(
        self,
        rewrite_result: QueryRewriteResult,
        category: str = "全部",
        limit: int | None = None,
    ) -> List[RetrievalCandidate]:
        """对每条 retrieval query 执行向量检索，再合并结果。"""
        fetch_k = limit or config.hybrid_fetch_k
        candidates: List[RetrievalCandidate] = []
        for query in rewrite_result.retrieval_queries:
            search_results = self.vector_store.search(query, category=category, limit=fetch_k)
            for document, score in search_results:
                metadata = dict(document.metadata)
                candidates.append(
                    RetrievalCandidate(
                        document_id=metadata.get("document_id", ""),
                        title=metadata.get("title", metadata.get("file_name", "未命名文档")),
                        category=metadata.get("category", "未分类"),
                        version=metadata.get("version", "未填写"),
                        file_name=metadata.get("file_name", ""),
                        content=document.page_content,
                        metadata=metadata,
                        score=0.0,
                        source="vector",
                        retrieval_scores={"vector_raw": float(score)},
                    )
                )

        vector_logger.info(
            "vector_retriever completed | category=%s | fetch_k=%s | query_count=%s | candidate_count=%s",
            category,
            fetch_k,
            len(rewrite_result.retrieval_queries),
            len(candidates),
        )
        return candidates


class BM25Retriever(BaseRetriever):
    """基于 BM25 的关键词检索器。"""

    def __init__(self, storage: JsonStorageService | None = None):
        self.storage = storage or JsonStorageService()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )
        self._indexed_chunks: List[RetrievalCandidate] | None = None
        self._tokenized_chunks: List[List[str]] | None = None
        self._document_frequencies: Dict[str, int] | None = None
        self._avg_doc_len: float = 0.0

    def retrieve(
        self,
        rewrite_result: QueryRewriteResult,
        category: str = "全部",
        limit: int | None = None,
    ) -> List[RetrievalCandidate]:
        self._ensure_index()
        fetch_k = limit or config.hybrid_fetch_k
        tokenized_chunks = self._tokenized_chunks or []
        candidates = self._indexed_chunks or []
        document_frequencies = self._document_frequencies or {}

        scored_candidates: List[tuple[float, RetrievalCandidate]] = []
        for chunk_tokens, candidate in zip(tokenized_chunks, candidates):
            if category != "全部" and candidate.category != category:
                continue
            score = self._score_queries(rewrite_result.retrieval_queries, chunk_tokens, document_frequencies)
            if score <= 0:
                continue
            cloned = RetrievalCandidate(
                document_id=candidate.document_id,
                title=candidate.title,
                category=candidate.category,
                version=candidate.version,
                file_name=candidate.file_name,
                content=candidate.content,
                metadata=dict(candidate.metadata),
                score=0.0,
                source="bm25",
                retrieval_scores={"bm25_raw": score},
            )
            scored_candidates.append((score, cloned))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        top_candidates = [candidate for _, candidate in scored_candidates[:fetch_k]]
        bm25_logger.info(
            "bm25_retriever completed | category=%s | fetch_k=%s | query_count=%s | candidate_count=%s",
            category,
            fetch_k,
            len(rewrite_result.retrieval_queries),
            len(top_candidates),
        )
        return top_candidates

    def _ensure_index(self) -> None:
        """延迟构建 BM25 索引。"""
        if self._indexed_chunks is not None:
            return

        indexed_chunks: List[RetrievalCandidate] = []
        tokenized_chunks: List[List[str]] = []
        document_frequencies: Dict[str, int] = defaultdict(int)

        # 只给成功入库的文档建索引，避免把失败/处理中记录混进来。
        for document in self.storage.list_documents():
            if document.get("status") != "success":
                continue
            content = self._read_raw_document(document)
            if not content:
                continue
            for chunk_index, chunk in enumerate(self._split_content(content)):
                metadata = {
                    "document_id": document.get("id", ""),
                    "chunk_index": chunk_index,
                }
                candidate = RetrievalCandidate(
                    document_id=document.get("id", ""),
                    title=document.get("title", document.get("file_name", "未命名文档")),
                    category=document.get("category", "未分类"),
                    version=document.get("version", "未填写"),
                    file_name=document.get("file_name", ""),
                    content=chunk,
                    metadata=metadata,
                    source="bm25",
                )
                tokens = self._tokenize(chunk)
                if not tokens:
                    continue
                indexed_chunks.append(candidate)
                tokenized_chunks.append(tokens)
                for token in set(tokens):
                    document_frequencies[token] += 1

        self._indexed_chunks = indexed_chunks
        self._tokenized_chunks = tokenized_chunks
        self._document_frequencies = dict(document_frequencies)
        if tokenized_chunks:
            self._avg_doc_len = sum(len(tokens) for tokens in tokenized_chunks) / len(tokenized_chunks)
        bm25_logger.info("bm25_index_ready | chunk_count=%s | avg_doc_len=%.2f", len(indexed_chunks), self._avg_doc_len)

    def _read_raw_document(self, document: Dict[str, object]) -> str:
        """从磁盘读取文档原文。"""
        raw_path = str(document.get("raw_path", "")).replace("\\", "/")
        if not raw_path:
            return ""
        file_path = config.BASE_DIR / raw_path
        if not file_path.exists():
            return ""
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _split_content(self, content: str) -> List[str]:
        return self.splitter.split_text(content) if len(content) > config.max_split_char_number else [content]

    def _score_queries(self, queries: List[str], chunk_tokens: List[str], document_frequencies: Dict[str, int]) -> float:
        """多条 query 中取最佳得分。"""
        best_score = 0.0
        for query in queries:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                continue
            score = self._bm25_score(query_tokens, chunk_tokens, document_frequencies)
            best_score = max(best_score, score)
        return best_score

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str], document_frequencies: Dict[str, int]) -> float:
        """BM25 核心打分公式。"""
        if not doc_tokens or not self._avg_doc_len:
            return 0.0
        k1 = 1.5
        b = 0.75
        term_counts = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        corpus_size = len(self._tokenized_chunks or [])
        score = 0.0
        for token in query_tokens:
            tf = term_counts.get(token, 0)
            if tf == 0:
                continue
            df = document_frequencies.get(token, 0)
            idf = math.log(1 + (corpus_size - df + 0.5) / (df + 0.5))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / self._avg_doc_len)
            score += idf * numerator / denominator
        return score

    def _tokenize(self, text: str) -> List[str]:
        """做一个适合中英混合文本的轻量分词。"""
        normalized = text.lower()
        base_tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized)
        chinese_chars = [token for token in base_tokens if re.fullmatch(r"[\u4e00-\u9fff]", token)]
        bigrams = [f"{chinese_chars[index]}{chinese_chars[index + 1]}" for index in range(len(chinese_chars) - 1)]
        return base_tokens + bigrams


class HybridRetriever(BaseRetriever):
    """混合检索器。"""

    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ):
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()

    def retrieve(
        self,
        rewrite_result: QueryRewriteResult,
        category: str = "全部",
        limit: int | None = None,
    ) -> List[RetrievalCandidate]:
        fetch_k = limit or config.hybrid_fetch_k
        vector_candidates = self.vector_retriever.retrieve(rewrite_result, category=category, limit=fetch_k)
        bm25_candidates = self.bm25_retriever.retrieve(rewrite_result, category=category, limit=fetch_k)
        vector_scores = self._normalize_vector_scores(vector_candidates)
        bm25_scores = self._normalize_positive_scores(bm25_candidates, raw_key="bm25_raw")

        merged: Dict[str, RetrievalCandidate] = {}
        for candidate in vector_candidates:
            key = candidate.unique_key
            merged_candidate = merged.setdefault(key, self._clone_candidate(candidate))
            merged_candidate.retrieval_scores["vector"] = vector_scores.get(key, 0.0)
        for candidate in bm25_candidates:
            key = candidate.unique_key
            merged_candidate = merged.setdefault(key, self._clone_candidate(candidate))
            merged_candidate.retrieval_scores["bm25"] = bm25_scores.get(key, 0.0)

        # 同一个 chunk 可能同时被两路检索命中，这里合并后再统一算分。
        for candidate in merged.values():
            vector_score = candidate.retrieval_scores.get("vector", 0.0)
            bm25_score = candidate.retrieval_scores.get("bm25", 0.0)
            candidate.score = config.hybrid_vector_weight * vector_score + config.hybrid_bm25_weight * bm25_score
            candidate.source = "hybrid"

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        hybrid_logger.info(
            "hybrid_retriever completed | category=%s | vector_candidates=%s | bm25_candidates=%s | merged_candidates=%s",
            category,
            len(vector_candidates),
            len(bm25_candidates),
            len(ranked),
        )
        hybrid_logger.debug(
            "hybrid_retriever top_candidates=%s",
            [
                {
                    "title": item.title,
                    "category": item.category,
                    "score": round(item.score, 4),
                    "scores": item.retrieval_scores,
                }
                for item in ranked[: min(5, len(ranked))]
            ],
        )
        return ranked[:fetch_k]

    def _normalize_vector_scores(self, candidates: List[RetrievalCandidate]) -> Dict[str, float]:
        """把向量距离转成“越大越好”的归一化分数。"""
        if not candidates:
            return {}
        reciprocal_scores = {
            candidate.unique_key: 1 / (1 + candidate.retrieval_scores.get("vector_raw", 0.0))
            for candidate in candidates
        }
        max_score = max(reciprocal_scores.values()) or 1.0
        return {key: score / max_score for key, score in reciprocal_scores.items()}

    def _normalize_positive_scores(self, candidates: List[RetrievalCandidate], raw_key: str) -> Dict[str, float]:
        """把正向分数归一化到 0~1。"""
        if not candidates:
            return {}
        raw_scores = {
            candidate.unique_key: candidate.retrieval_scores.get(raw_key, 0.0)
            for candidate in candidates
        }
        max_score = max(raw_scores.values()) or 1.0
        return {key: score / max_score for key, score in raw_scores.items()}

    def _clone_candidate(self, candidate: RetrievalCandidate) -> RetrievalCandidate:
        return RetrievalCandidate(
            document_id=candidate.document_id,
            title=candidate.title,
            category=candidate.category,
            version=candidate.version,
            file_name=candidate.file_name,
            content=candidate.content,
            metadata=dict(candidate.metadata),
            score=candidate.score,
            source=candidate.source,
            retrieval_scores=dict(candidate.retrieval_scores),
        )


class RetrieverFactory:
    """检索器工厂。"""

    @staticmethod
    def create(strategy: str = "hybrid") -> BaseRetriever:
        if strategy == "vector":
            return VectorRetriever()
        if strategy == "bm25":
            return BM25Retriever()
        return HybridRetriever()


class RetrievalCoordinator:
    """多任务检索协调器。"""

    def __init__(self, retriever: BaseRetriever, query_rewriter: QueryRewriter):
        self.retriever = retriever
        self.query_rewriter = query_rewriter

    def collect_shared_candidates(
        self,
        rewrite_result: QueryRewriteResult,
        planned_tasks: List[PlannedTask],
        selected_category: str = "全部",
        limit: int | None = None,
    ) -> tuple[Dict[str, List[RetrievalCandidate]], Dict[str, QueryRewriteResult]]:
        """按分类为一组子任务准备共享候选池。"""
        grouped_tasks = self._group_tasks_by_category(planned_tasks, selected_category)
        category_candidates_map: Dict[str, List[RetrievalCandidate]] = {}
        category_rewrite_map: Dict[str, QueryRewriteResult] = {}

        for category, tasks in grouped_tasks.items():
            merged_hints = self._merge_task_hints(tasks)
            scoped_rewrite = self.query_rewriter.narrow_to_hints(rewrite_result, merged_hints)
            category_candidates = self.retriever.retrieve(scoped_rewrite, category=category, limit=limit)
            # 某些问题天然跨部门，例如“入职时工位、门禁、权限何时开通”，
            # 这时只检索单一分类容易漏证据，所以会补一轮全局候选。
            if self._should_expand_with_global_candidates(
                category=category,
                selected_category=selected_category,
                rewrite_result=scoped_rewrite,
                hints=merged_hints,
            ):
                global_candidates = self.retriever.retrieve(scoped_rewrite, category="全部", limit=limit)
                category_candidates = self._merge_candidate_pools(
                    primary_candidates=category_candidates,
                    supplemental_candidates=global_candidates,
                )
            category_candidates_map[category] = category_candidates
            category_rewrite_map[category] = scoped_rewrite
            coordinator_logger.info(
                "retrieval_coordinator shared_retrieval | category=%s | task_ids=%s | hints=%s | query_count=%s | candidate_count=%s",
                category,
                [task.task_id for task in tasks],
                merged_hints,
                len(scoped_rewrite.retrieval_queries),
                len(category_candidates),
            )

        return category_candidates_map, category_rewrite_map

    def _should_expand_with_global_candidates(
        self,
        *,
        category: str,
        selected_category: str,
        rewrite_result: QueryRewriteResult,
        hints: List[str],
    ) -> bool:
        """判断当前分类检索是否还要补做全库检索。"""
        if selected_category != "全部" or category == "全部":
            return False
        text = " ".join([rewrite_result.original_query, rewrite_result.normalized_query, *rewrite_result.retrieval_queries, *hints]).lower()
        cross_category_keywords = (
            "入职",
            "新员工",
            "紧急入职",
            "权限开通",
            "账号开通",
            "门禁",
            "工位",
            "办公设备",
            "跨部门",
        )
        return any(keyword in text for keyword in cross_category_keywords)

    def _merge_candidate_pools(
        self,
        *,
        primary_candidates: List[RetrievalCandidate],
        supplemental_candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """合并主候选池和补充候选池，重复时保留高分项。"""
        merged: Dict[str, RetrievalCandidate] = {}
        for candidate in [*primary_candidates, *supplemental_candidates]:
            key = candidate.unique_key
            existing = merged.get(key)
            if existing is None or candidate.score > existing.score:
                merged[key] = deepcopy(candidate)
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)

    def _group_tasks_by_category(self, planned_tasks: List[PlannedTask], selected_category: str) -> Dict[str, List[PlannedTask]]:
        """把多个子任务按最终生效分类聚合。"""
        groups: Dict[str, List[PlannedTask]] = {}
        for task in planned_tasks:
            category = selected_category if selected_category != "全部" else task.category
            groups.setdefault(category, []).append(task)
        return groups

    def _merge_task_hints(self, tasks: List[PlannedTask]) -> List[str]:
        """合并一个分类组内所有 hints，并做保序去重。"""
        merged_hints = []
        for task in tasks:
            merged_hints.extend(task.hints)
        return list(dict.fromkeys(item.strip() for item in merged_hints if item.strip()))


__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "RetrievalCandidate",
    "RetrievalCoordinator",
    "RetrieverFactory",
    "VectorRetriever",
]
