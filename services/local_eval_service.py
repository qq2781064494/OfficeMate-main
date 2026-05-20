"""本地题库主 RAG 测评服务。"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
from services.benchmark_eval_service import _SanitizedRagasLLM
from services.benchmark_results import BenchmarkResultStore
from services.benchmark_store import BenchmarkChunkConfig
from services.chat_service import OfficeMateChatService
from services.local_eval_store import LocalEvalCorpusStore
from services.model_provider import ModelProviderFactory
from services.rag.query import QueryRewriteResult
from services.rag.retrieval import BaseRetriever, RetrievalCandidate
from utils.log_tool import get_logger


logger = get_logger("local_eval")


@dataclass
class LocalEvalConfig:
    knowledge_base_id: str
    knowledge_base_name: str
    dataset_key: str
    dataset_label: str
    sample_path: Path
    retriever_strategy: str = "hybrid"
    top_k: int = config.benchmark_default_top_k
    question_limit: int = 0
    selected_question_ids: list[int] = field(default_factory=list)
    enable_query_rewrite: bool = True
    enable_ragas: bool = True
    enable_faithfulness: bool = True
    enable_rerank: bool = True
    chunk_config: BenchmarkChunkConfig = field(default_factory=BenchmarkChunkConfig)


class LocalEvalService:
    """使用本地题库和主问答链路运行评测。"""

    def __init__(
        self,
        corpus_store: LocalEvalCorpusStore | None = None,
        result_store: BenchmarkResultStore | None = None,
    ):
        self.corpus_store = corpus_store or LocalEvalCorpusStore()
        self.result_store = result_store or BenchmarkResultStore()

    def run_evaluation(
        self,
        eval_config: LocalEvalConfig,
        status_callback: Callable[[str], None] | None = None,
    ) -> dict:
        all_samples = self.corpus_store.load_eval_samples(eval_config.sample_path)
        samples, selected_question_ids = self._resolve_samples(
            all_samples,
            selected_question_ids=eval_config.selected_question_ids,
            question_limit=eval_config.question_limit,
        )
        kb_info = self.corpus_store.get_knowledge_base(eval_config.knowledge_base_id)
        if not kb_info:
            raise ValueError(f"未找到知识库：{eval_config.knowledge_base_id}")
        retriever = _LocalEvalRetriever(
            knowledge_base_id=eval_config.knowledge_base_id,
            strategy=eval_config.retriever_strategy,
            corpus_store=self.corpus_store,
            manifest=self.corpus_store.load_corpus_manifest(eval_config.knowledge_base_id),
            chunk_config=eval_config.chunk_config,
        )
        chat_service = OfficeMateChatService(retriever=retriever)

        self._emit_status(status_callback, "正在批量运行本地题库检索与问答...")
        details = []
        for index, sample in enumerate(samples, start=1):
            self._emit_status(status_callback, f"正在处理第 {index}/{len(samples)} 题...")
            detail = self._run_single_sample(
                sample=sample,
                question_id=selected_question_ids[index - 1],
                eval_config=eval_config,
                chat_service=chat_service,
            )
            details.append(detail)

        retrieval_metrics = self._compute_retrieval_metrics(details, top_k=eval_config.top_k)
        rerank_metrics = (
            self._compute_rerank_metrics(details, top_k=eval_config.top_k)
            if eval_config.enable_rerank
            else {"status": "disabled"}
        )
        ragas_metrics = (
            self._compute_ragas_metrics(details, enable_faithfulness=eval_config.enable_faithfulness)
            if eval_config.enable_ragas
            else {"status": "disabled", "faithfulness": "已跳过"}
        )

        summary = {
            "subset": "local_eval_kb",
            "split": "sample_docs",
            "mode": "local_eval_rag",
            "knowledge_base_id": eval_config.knowledge_base_id,
            "knowledge_base_name": eval_config.knowledge_base_name,
            "dataset_label": eval_config.dataset_label,
            "retriever_strategy": eval_config.retriever_strategy,
            "top_k": eval_config.top_k,
            "question_limit": len(details),
            "selected_question_ids": selected_question_ids,
            "enable_query_rewrite": eval_config.enable_query_rewrite,
            "enable_rerank": eval_config.enable_rerank,
            "enable_faithfulness": eval_config.enable_faithfulness,
            "question_count": len(details),
            "document_count": kb_info["document_count"],
            "chunk_count": kb_info.get("chunk_count", 0),
            "chunk_config": kb_info.get("chunk_config", {}),
            "retrieval_metrics": retrieval_metrics,
            "rerank_metrics": rerank_metrics,
            "ragas_metrics": ragas_metrics,
        }
        run_record = self.result_store.save_run(summary, details)
        summary["run_id"] = run_record["run_id"]
        summary["detail_path"] = run_record["detail_path"]
        self._emit_status(status_callback, "评测完成。")
        return summary

    def _resolve_samples(self, all_samples: list[dict], selected_question_ids: list[int], question_limit: int) -> tuple[list[dict], list[int]]:
        indexed_samples = list(enumerate(all_samples, start=1))
        if selected_question_ids:
            selected_set = {int(item) for item in selected_question_ids}
            indexed_samples = [item for item in indexed_samples if item[0] in selected_set]
        if question_limit:
            indexed_samples = indexed_samples[: min(question_limit, len(indexed_samples))]
        return [sample for _, sample in indexed_samples], [index for index, _ in indexed_samples]

    def _run_single_sample(
        self,
        sample: dict,
        question_id: int,
        eval_config: LocalEvalConfig,
        chat_service: OfficeMateChatService,
    ) -> dict:
        question = sample.get("query", "")
        expected_titles = sample.get("expected_titles", [])
        effective_category = "全部" if eval_config.dataset_key == "local_sample_complex_20" else str(sample.get("category", "全部"))
        result = chat_service.answer_question(
            question=question,
            session_id=f"{eval_config.knowledge_base_id}_{question_id}",
            category=effective_category,
            use_history=False,
            persist_log=False,
            include_references=False,
            enable_query_rewrite=eval_config.enable_query_rewrite,
            enable_rerank=eval_config.enable_rerank,
            reference_limit=eval_config.top_k,
        )
        retrieved_titles = result.get("retrieved_titles", [])
        pre_rerank_titles = result.get("pre_rerank_titles", [])

        first_hit_rank = None
        for rank, title in enumerate(retrieved_titles[: eval_config.top_k], start=1):
            if title in expected_titles:
                first_hit_rank = rank
                break
        pre_rerank_first_hit_rank = None
        for rank, title in enumerate(pre_rerank_titles[: eval_config.top_k], start=1):
            if title in expected_titles:
                pre_rerank_first_hit_rank = rank
                break

        return {
            "question_id": question_id,
            "knowledge_base_id": eval_config.knowledge_base_id,
            "question": question,
            "effective_category": effective_category,
            "normalized_query": result.get("normalized_query", question),
            "retrieval_queries": result.get("retrieval_queries", [question]),
            "matched_terms": result.get("matched_terms", []),
            "expected_titles": expected_titles,
            "pre_rerank_titles": pre_rerank_titles,
            "retrieved_titles": retrieved_titles,
            "retrieved_contexts": result.get("retrieved_contexts", []),
            "gold_answer": sample.get("gold_answer", ""),
            "predicted_answer": result.get("answer", ""),
            "question_type": result.get("question_type", ""),
            "pre_rerank_hit": pre_rerank_first_hit_rank is not None,
            "pre_rerank_first_hit_rank": pre_rerank_first_hit_rank,
            "retrieval_hit": first_hit_rank is not None,
            "first_hit_rank": first_hit_rank,
            "top_k": eval_config.top_k,
        }

    def _compute_retrieval_metrics(self, details: list[dict], top_k: int) -> dict:
        if not details:
            return {
                "recall_at_1": 0.0,
                "recall_at_3": 0.0,
                "recall_at_5": 0.0,
                "hit_rate_at_1": 0.0,
                "hit_rate_at_3": 0.0,
                "hit_rate_at_5": 0.0,
                "mrr": 0.0,
            }

        metrics = {}
        for k in [1, 3, 5]:
            capped_k = min(k, top_k)
            hit_count = 0
            recall_sum = 0.0
            for detail in details:
                expected_titles = set(detail.get("expected_titles", []))
                retrieved_titles = detail.get("retrieved_titles", [])[:capped_k]
                matched_titles = {title for title in retrieved_titles if title in expected_titles}
                if matched_titles:
                    hit_count += 1
                if expected_titles:
                    recall_sum += len(matched_titles) / len(expected_titles)
            metrics[f"recall_at_{k}"] = round(recall_sum / len(details), 4)
            metrics[f"hit_rate_at_{k}"] = round(hit_count / len(details), 4)

        reciprocal_rank_sum = 0.0
        for detail in details:
            rank = detail.get("first_hit_rank")
            if rank:
                reciprocal_rank_sum += 1 / rank
        metrics["mrr"] = round(reciprocal_rank_sum / len(details), 4)
        return metrics

    def _compute_rerank_metrics(self, details: list[dict], top_k: int) -> dict:
        if not details:
            return {"status": "empty"}

        pre_hit_at_1 = 0
        post_hit_at_1 = 0
        pre_hit_at_3 = 0
        post_hit_at_3 = 0
        pre_mrr_sum = 0.0
        post_mrr_sum = 0.0
        win = 0
        tie = 0
        lose = 0
        valid_rank_rows = 0
        rank_shift_sum = 0.0

        for detail in details:
            pre_rank = detail.get("pre_rerank_first_hit_rank")
            post_rank = detail.get("first_hit_rank")
            if pre_rank == 1:
                pre_hit_at_1 += 1
            if post_rank == 1:
                post_hit_at_1 += 1
            if pre_rank and pre_rank <= min(3, top_k):
                pre_hit_at_3 += 1
            if post_rank and post_rank <= min(3, top_k):
                post_hit_at_3 += 1
            if pre_rank:
                pre_mrr_sum += 1 / pre_rank
            if post_rank:
                post_mrr_sum += 1 / post_rank
            if pre_rank and post_rank:
                valid_rank_rows += 1
                rank_shift_sum += pre_rank - post_rank
                if post_rank < pre_rank:
                    win += 1
                elif post_rank == pre_rank:
                    tie += 1
                else:
                    lose += 1

        total = len(details)
        return {
            "status": "success",
            "pre_hit_rate_at_1": round(pre_hit_at_1 / total, 4),
            "post_hit_rate_at_1": round(post_hit_at_1 / total, 4),
            "pre_hit_rate_at_3": round(pre_hit_at_3 / total, 4),
            "post_hit_rate_at_3": round(post_hit_at_3 / total, 4),
            "pre_mrr": round(pre_mrr_sum / total, 4),
            "post_mrr": round(post_mrr_sum / total, 4),
            "delta_mrr": round((post_mrr_sum - pre_mrr_sum) / total, 4),
            "win_rate": round(win / total, 4),
            "tie_rate": round(tie / total, 4),
            "lose_rate": round(lose / total, 4),
            "avg_rank_improvement": round(rank_shift_sum / valid_rank_rows, 4) if valid_rank_rows else 0.0,
        }

    def _compute_ragas_metrics(self, details: list[dict], *, enable_faithfulness: bool) -> dict:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
            from ragas.run_config import RunConfig
        except ImportError as exc:
            return {"status": "missing_dependency", "error": str(exc)}
        except Exception as exc:
            return {"status": "failed_to_import", "error": str(exc)}

        dataset = Dataset.from_dict(
            {
                "question": [item["question"] for item in details],
                "answer": [item["predicted_answer"] for item in details],
                "contexts": [item["retrieved_contexts"] for item in details],
                "ground_truth": [item["gold_answer"] for item in details],
            }
        )
        metric_definitions = []
        if enable_faithfulness:
            metric_definitions.append(("faithfulness", deepcopy(faithfulness)))
        metric_definitions.extend(
            [
                ("answer_relevancy", deepcopy(answer_relevancy)),
                ("context_precision", deepcopy(context_precision)),
                ("context_recall", deepcopy(context_recall)),
            ]
        )
        metric_names = [name for name, _ in metric_definitions]
        job_error_collector = _RagasJobErrorCollector()
        ragas_executor_logger = logging.getLogger("ragas.executor")
        ragas_executor_logger.addHandler(job_error_collector)
        try:
            eval_llm = _SanitizedRagasLLM(
                ModelProviderFactory.create_benchmark_chat_provider().build_chat_model(temperature=0)
            )
            eval_embeddings = ModelProviderFactory.create_benchmark_embedding_provider().build_embedding_client(
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
            )
            metric_objects = [metric for _, metric in metric_definitions]
            answer_relevancy_metric = next(metric for name, metric in metric_definitions if name == "answer_relevancy")
            answer_relevancy_metric.strictness = 1
            run_config = RunConfig(timeout=300, max_retries=3, max_workers=1)
            result = evaluate(
                dataset,
                metrics=metric_objects,
                llm=eval_llm,
                embeddings=eval_embeddings,
                run_config=run_config,
                batch_size=1,
            )
            job_errors = self._map_ragas_job_errors(
                raw_job_errors=job_error_collector.errors,
                details=details,
                metric_names=metric_names,
            )
            if hasattr(result, "to_pandas"):
                frame = result.to_pandas()
                missing_counts = {}
                for metric_name in metric_names:
                    if metric_name in frame.columns:
                        missing_counts[metric_name] = int(frame[metric_name].isna().sum())
                payload = {
                    "status": "success_with_warnings" if job_errors else "success",
                    "answer_relevancy": round(float(frame["answer_relevancy"].mean()), 4),
                    "context_precision": round(float(frame["context_precision"].mean()), 4),
                    "context_recall": round(float(frame["context_recall"].mean()), 4),
                    "job_errors": job_errors,
                    "missing_counts": missing_counts,
                }
                payload["faithfulness"] = (
                    round(float(frame["faithfulness"].mean()), 4)
                    if enable_faithfulness and "faithfulness" in frame.columns
                    else "已跳过"
                )
                return payload
            return {
                "status": "success_with_warnings" if job_errors else "success",
                **dict(result),
                "job_errors": job_errors,
                "faithfulness": None if enable_faithfulness else "已跳过",
            }
        except Exception as exc:
            logger.exception("local_eval ragas_failed | error=%s", exc)
            job_errors = self._map_ragas_job_errors(
                raw_job_errors=job_error_collector.errors,
                details=details,
                metric_names=metric_names,
            )
            return {"status": "failed", "error": str(exc), "job_errors": job_errors}
        finally:
            ragas_executor_logger.removeHandler(job_error_collector)

    def _emit_status(self, callback: Callable[[str], None] | None, message: str) -> None:
        if callback:
            callback(message)

    def _map_ragas_job_errors(self, raw_job_errors: list[dict], details: list[dict], metric_names: list[str]) -> list[dict]:
        mapped_errors = []
        metric_count = len(metric_names)
        for item in raw_job_errors:
            job_index = int(item.get("job_index", -1))
            if job_index < 0 or metric_count == 0:
                continue
            row_index = job_index // metric_count
            metric_index = job_index % metric_count
            detail = details[row_index] if 0 <= row_index < len(details) else {}
            metric_name = metric_names[metric_index] if 0 <= metric_index < metric_count else "unknown"
            mapped_errors.append(
                {
                    "job_index": job_index,
                    "question_row_index": row_index + 1,
                    "question_id": detail.get("question_id"),
                    "question": detail.get("question", ""),
                    "metric": metric_name,
                    "exception_type": item.get("exception_type", ""),
                    "exception_message": item.get("exception_message", ""),
                    "is_timeout": item.get("exception_type") == "TimeoutError",
                }
            )
        return mapped_errors


class _RagasJobErrorCollector(logging.Handler):
    _PATTERN = re.compile(r"Exception raised in Job\[(?P<job_index>\d+)\]:\s*(?P<exception_type>[A-Za-z_][A-Za-z0-9_]*)\((?P<exception_message>.*)\)")

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.errors: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        match = self._PATTERN.search(message)
        if not match:
            return
        self.errors.append(
            {
                "job_index": int(match.group("job_index")),
                "exception_type": match.group("exception_type"),
                "exception_message": match.group("exception_message"),
            }
        )


class _LocalEvalRetriever(BaseRetriever):
    """在本地题库独立 corpus 上复用多策略检索。"""

    def __init__(
        self,
        knowledge_base_id: str,
        strategy: str,
        corpus_store: LocalEvalCorpusStore,
        manifest: list[dict],
        chunk_config: BenchmarkChunkConfig,
    ):
        self.knowledge_base_id = knowledge_base_id
        self.strategy = strategy
        self.corpus_store = corpus_store
        self.vector_store = corpus_store.get_vector_store(knowledge_base_id, chunk_config=chunk_config)
        self.manifest = manifest
        self.chunk_config = chunk_config
        self.splitter = _LocalEvalChunker(chunk_config)
        self._bm25_candidates, self._bm25_tokens, self._document_frequencies, self._avg_doc_len = self._build_bm25_index()

    def retrieve(self, rewrite_result: QueryRewriteResult, category: str = "全部", limit: int | None = None) -> list[RetrievalCandidate]:
        limit = limit or config.hybrid_fetch_k
        if self.strategy == "vector":
            return self._vector_retrieve(rewrite_result, category, limit)
        if self.strategy == "bm25":
            return self._bm25_retrieve(rewrite_result, category, limit)
        return self._hybrid_retrieve(rewrite_result, category, limit)

    def _vector_retrieve(self, rewrite_result: QueryRewriteResult, category: str, limit: int) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        for query in rewrite_result.retrieval_queries:
            for document, score in self.vector_store.search(query, limit, category=category):
                metadata = dict(document.metadata)
                candidates.append(
                    RetrievalCandidate(
                        document_id=metadata.get("document_id", ""),
                        title=metadata.get("title", metadata.get("file_name", "")),
                        category=metadata.get("category", "未分类"),
                        version=metadata.get("version", "未填写"),
                        file_name=metadata.get("file_name", ""),
                        content=document.page_content,
                        metadata=metadata,
                        source="vector",
                        retrieval_scores={"vector_raw": float(score)},
                    )
                )
        ranked = self._dedupe_keep_best(candidates, score_key="vector_raw", reverse=False)
        return ranked[:limit]

    def _bm25_retrieve(self, rewrite_result: QueryRewriteResult, category: str, limit: int) -> list[RetrievalCandidate]:
        scored_candidates: list[tuple[float, RetrievalCandidate]] = []
        for tokens, candidate in zip(self._bm25_tokens, self._bm25_candidates):
            if category != "全部" and candidate.category != category:
                continue
            score = self._score_queries(rewrite_result.retrieval_queries, tokens)
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
                source="bm25",
                retrieval_scores={"bm25_raw": score},
            )
            scored_candidates.append((score, cloned))
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in scored_candidates[:limit]]

    def _hybrid_retrieve(self, rewrite_result: QueryRewriteResult, category: str, limit: int) -> list[RetrievalCandidate]:
        vector_candidates = self._vector_retrieve(rewrite_result, category, limit)
        bm25_candidates = self._bm25_retrieve(rewrite_result, category, limit)
        vector_scores = self._normalize_vector_scores(vector_candidates)
        bm25_scores = self._normalize_positive_scores(bm25_candidates, raw_key="bm25_raw")

        merged: dict[str, RetrievalCandidate] = {}
        for candidate in vector_candidates:
            key = candidate.unique_key
            merged_candidate = merged.setdefault(key, self._clone_candidate(candidate))
            merged_candidate.retrieval_scores["vector"] = vector_scores.get(key, 0.0)
        for candidate in bm25_candidates:
            key = candidate.unique_key
            merged_candidate = merged.setdefault(key, self._clone_candidate(candidate))
            merged_candidate.retrieval_scores["bm25"] = bm25_scores.get(key, 0.0)
        for candidate in merged.values():
            candidate.score = (
                config.hybrid_vector_weight * candidate.retrieval_scores.get("vector", 0.0)
                + config.hybrid_bm25_weight * candidate.retrieval_scores.get("bm25", 0.0)
            )
            candidate.source = "hybrid"
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]

    def _build_bm25_index(self):
        indexed_chunks: list[RetrievalCandidate] = []
        tokenized_chunks: list[list[str]] = []
        document_frequencies: dict[str, int] = defaultdict(int)

        for document in self.manifest:
            raw_path = config.BASE_DIR / document["raw_path"]
            content = raw_path.read_text(encoding="utf-8")
            for chunk_index, chunk in enumerate(self.splitter.split(content)):
                candidate = RetrievalCandidate(
                    document_id=document["document_id"],
                    title=document["title"],
                    category=document["category"],
                    version=document["version"],
                    file_name=document["file_name"],
                    content=chunk,
                    metadata={"document_id": document["document_id"], "chunk_index": chunk_index},
                    source="bm25",
                )
                tokens = self._tokenize(chunk)
                if not tokens:
                    continue
                indexed_chunks.append(candidate)
                tokenized_chunks.append(tokens)
                for token in set(tokens):
                    document_frequencies[token] += 1

        avg_doc_len = 0.0
        if tokenized_chunks:
            avg_doc_len = sum(len(tokens) for tokens in tokenized_chunks) / len(tokenized_chunks)
        return indexed_chunks, tokenized_chunks, dict(document_frequencies), avg_doc_len

    def _score_queries(self, queries: list[str], doc_tokens: list[str]) -> float:
        best_score = 0.0
        for query in queries:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                continue
            best_score = max(best_score, self._bm25_score(query_tokens, doc_tokens))
        return best_score

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens or not self._avg_doc_len:
            return 0.0
        k1 = 1.5
        b = 0.75
        term_counts = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        corpus_size = len(self._bm25_tokens or [])
        score = 0.0
        for token in query_tokens:
            tf = term_counts.get(token, 0)
            if tf == 0:
                continue
            df = self._document_frequencies.get(token, 0)
            idf = math.log(1 + (corpus_size - df + 0.5) / (df + 0.5))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / self._avg_doc_len)
            score += idf * numerator / denominator
        return score

    def _tokenize(self, text: str) -> list[str]:
        normalized = text.lower()
        base_tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized)
        chinese_chars = [token for token in base_tokens if re.fullmatch(r"[\u4e00-\u9fff]", token)]
        bigrams = [f"{chinese_chars[i]}{chinese_chars[i + 1]}" for i in range(len(chinese_chars) - 1)]
        return base_tokens + bigrams

    def _normalize_vector_scores(self, candidates: list[RetrievalCandidate]) -> dict[str, float]:
        if not candidates:
            return {}
        reciprocal_scores = {
            candidate.unique_key: 1 / (1 + candidate.retrieval_scores.get("vector_raw", 0.0))
            for candidate in candidates
        }
        max_score = max(reciprocal_scores.values()) or 1.0
        return {key: score / max_score for key, score in reciprocal_scores.items()}

    def _normalize_positive_scores(self, candidates: list[RetrievalCandidate], raw_key: str) -> dict[str, float]:
        if not candidates:
            return {}
        raw_scores = {
            candidate.unique_key: candidate.retrieval_scores.get(raw_key, 0.0)
            for candidate in candidates
        }
        max_score = max(raw_scores.values()) or 1.0
        return {key: score / max_score for key, score in raw_scores.items()}

    def _dedupe_keep_best(self, candidates: list[RetrievalCandidate], score_key: str, reverse: bool) -> list[RetrievalCandidate]:
        best_by_key: dict[str, RetrievalCandidate] = {}
        for candidate in candidates:
            key = candidate.unique_key
            score = candidate.retrieval_scores.get(score_key, 0.0)
            existing = best_by_key.get(key)
            if existing is None:
                best_by_key[key] = candidate
                continue
            existing_score = existing.retrieval_scores.get(score_key, 0.0)
            should_replace = score > existing_score if reverse else score < existing_score
            if should_replace:
                best_by_key[key] = candidate
        return sorted(
            best_by_key.values(),
            key=lambda item: item.retrieval_scores.get(score_key, 0.0),
            reverse=reverse,
        )

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


class _LocalEvalChunker:
    def __init__(self, chunk_config: BenchmarkChunkConfig):
        self.chunk_config = chunk_config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_config.chunk_size,
            chunk_overlap=self.chunk_config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def split(self, text: str) -> list[str]:
        if len(text) <= self.chunk_config.max_split_char_number:
            return [text]
        return self.splitter.split_text(text)
