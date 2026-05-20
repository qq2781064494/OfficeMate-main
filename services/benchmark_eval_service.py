"""RAGBench 全局知识库模式评测服务。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config_data as config
from services.benchmark_results import BenchmarkResultStore
from services.benchmark_store import BenchmarkChunkConfig, BenchmarkCorpusStore
from services.model_provider import ModelProviderFactory
from services.rag.query import QueryRewriteResult, QueryRewriter
from services.rag.retrieval import RetrievalCandidate
from utils.log_tool import get_logger


logger = get_logger("benchmark_eval")


class _SanitizedRagasLLM:
    """给 Ragas 用的轻量包装，兼容只返回单候选且会混入 `<think>` 的模型。"""

    def __init__(self, base_llm):
        from ragas.llms.base import LangchainLLMWrapper

        self._wrapper = LangchainLLMWrapper(base_llm, bypass_n=True)

    def __getattr__(self, name):
        return getattr(self._wrapper, name)

    async def generate(self, *args, **kwargs):
        result = await self._wrapper.generate(*args, **kwargs)
        return self._sanitize_result(result)

    def generate_text(self, *args, **kwargs):
        result = self._wrapper.generate_text(*args, **kwargs)
        return self._sanitize_result(result)

    async def agenerate_text(self, *args, **kwargs):
        result = await self._wrapper.agenerate_text(*args, **kwargs)
        return self._sanitize_result(result)

    def _sanitize_result(self, result):
        for generation_group in result.generations:
            for generation in generation_group:
                if hasattr(generation, "text") and isinstance(generation.text, str):
                    generation.text = _strip_thinking_blocks(generation.text)
                if hasattr(generation, "message") and getattr(generation, "message", None) is not None:
                    content = getattr(generation.message, "content", None)
                    if isinstance(content, str):
                        generation.message.content = _strip_thinking_blocks(content)
        return result


def _strip_thinking_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


@dataclass
class BenchmarkEvalConfig:
    subset: str
    split: str = "test"
    retriever_strategy: str = "hybrid"
    top_k: int = config.benchmark_default_top_k
    question_limit: int = config.benchmark_default_question_limit
    enable_query_rewrite: bool = True
    enable_ragas: bool = True
    enable_faithfulness: bool = True
    enable_rerank: bool = True
    rebuild_corpus: bool = False
    rebuild_index: bool = False
    chunk_config: BenchmarkChunkConfig = field(default_factory=BenchmarkChunkConfig)


class BenchmarkEvalService:
    """以全局知识库模式运行 subset 级 benchmark。"""

    def __init__(
        self,
        corpus_store: BenchmarkCorpusStore | None = None,
        result_store: BenchmarkResultStore | None = None,
    ):
        self.corpus_store = corpus_store or BenchmarkCorpusStore()
        self.result_store = result_store or BenchmarkResultStore()
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 OfficeMate 的 benchmark 助手。"
                    "你只能依据提供的上下文回答问题。"
                    "如果上下文不足，请明确说明未找到明确依据。"
                    "请直接输出简洁答案，答案语言应该与问题语言一致，不要输出额外解释。",
                ),
                (
                    "human",
                    "问题：{question}\n\n"
                    "上下文：\n{context}\n\n"
                    "请基于上下文作答,答案语言要和问题语言一样，如问题是英文，就用英文作答，问题是中文，就用中文作答。",
                ),
            ]
        )
        self.chat_model = ModelProviderFactory.create_benchmark_chat_provider().build_chat_model(
            temperature=0,
            extra_body={"reasoning_split": True},
        )
        self.rewrite_model = None
        self.query_rewriter = QueryRewriter(chat_model_factory=self._get_rewrite_model)

    def run_evaluation(
        self,
        eval_config: BenchmarkEvalConfig,
        status_callback: Callable[[str], None] | None = None,
    ) -> dict:
        self._emit_status(status_callback, "正在构建 benchmark 全局知识库...")
        corpus_summary = self.corpus_store.build_subset_corpus(
            subset=eval_config.subset,
            splits=[eval_config.split],
            rebuild=eval_config.rebuild_corpus,
        )
        index_summary = self.corpus_store.ensure_vector_index(
            subset=eval_config.subset,
            rebuild=eval_config.rebuild_index,
            chunk_config=eval_config.chunk_config,
        )

        samples = self.corpus_store.load_eval_samples(eval_config.subset, eval_config.split)
        if eval_config.question_limit:
            samples = samples[: min(eval_config.question_limit, len(samples))]

        manifest = self.corpus_store.load_corpus_manifest(eval_config.subset)
        retriever = _BenchmarkRetriever(
            subset=eval_config.subset,
            strategy=eval_config.retriever_strategy,
            corpus_store=self.corpus_store,
            manifest=manifest,
            chunk_config=eval_config.chunk_config,
            enable_rerank=eval_config.enable_rerank,
        )

        self._emit_status(status_callback, "正在批量运行检索与问答...")
        details = []
        for index, sample in enumerate(samples, start=1):
            self._emit_status(status_callback, f"正在处理第 {index}/{len(samples)} 题...")
            detail = self._run_single_sample(sample, eval_config, retriever)
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
            "subset": eval_config.subset,
            "split": eval_config.split,
            "mode": "global_knowledge_base",
            "retriever_strategy": eval_config.retriever_strategy,
            "top_k": eval_config.top_k,
            "question_limit": eval_config.question_limit,
            "enable_query_rewrite": eval_config.enable_query_rewrite,
            "enable_rerank": eval_config.enable_rerank,
            "enable_faithfulness": eval_config.enable_faithfulness,
            "question_count": len(details),
            "document_count": corpus_summary["document_count"],
            "chunk_count": index_summary.get("chunk_count", 0),
            "chunk_config": eval_config.chunk_config.to_dict(),
            "retrieval_metrics": retrieval_metrics,
            "rerank_metrics": rerank_metrics,
            "ragas_metrics": ragas_metrics,
        }
        run_record = self.result_store.save_run(summary, details)
        summary["run_id"] = run_record["run_id"]
        summary["detail_path"] = run_record["detail_path"]
        self._emit_status(status_callback, "评测完成。")
        return summary

    def _run_single_sample(
        self,
        sample: dict,
        eval_config: BenchmarkEvalConfig,
        retriever: "_BenchmarkRetriever",
    ) -> dict:
        question = sample.get("query", "")
        expected_titles = sample.get("expected_titles", [])
        rewrite_result = self._build_rewrite_result(question, enable_query_rewrite=eval_config.enable_query_rewrite)
        retrieval_debug = retriever.retrieve_with_debug(rewrite_result, limit=eval_config.top_k)
        pre_rerank_candidates = retrieval_debug["pre_rerank_candidates"]
        candidates = retrieval_debug["final_candidates"]
        retrieved_titles = [candidate.title for candidate in candidates]
        pre_rerank_titles = [candidate.title for candidate in pre_rerank_candidates]
        first_hit_rank = None
        for rank, title in enumerate(retrieved_titles, start=1):
            if title in expected_titles:
                first_hit_rank = rank
                break
        pre_rerank_first_hit_rank = None
        for rank, title in enumerate(pre_rerank_titles, start=1):
            if title in expected_titles:
                pre_rerank_first_hit_rank = rank
                break

        contexts = [self._format_candidate_context(candidate) for candidate in candidates]
        answer = self._generate_answer(question, contexts)
        return {
            "question_id": sample.get("metadata", {}).get("ragbench_id", ""),
            "subset": eval_config.subset,
            "split": eval_config.split,
            "question": question,
            "normalized_query": rewrite_result.normalized_query,
            "retrieval_queries": rewrite_result.retrieval_queries,
            "matched_terms": rewrite_result.matched_terms,
            "expected_titles": expected_titles,
            "pre_rerank_titles": pre_rerank_titles,
            "retrieved_titles": retrieved_titles,
            "retrieved_contexts": contexts,
            "gold_answer": sample.get("metadata", {}).get("gold_response", ""),
            "predicted_answer": answer,
            "pre_rerank_hit": pre_rerank_first_hit_rank is not None,
            "pre_rerank_first_hit_rank": pre_rerank_first_hit_rank,
            "retrieval_hit": first_hit_rank is not None,
            "first_hit_rank": first_hit_rank,
            "top_k": eval_config.top_k,
        }

    def _generate_answer(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "未找到明确依据。"
        chain = self.answer_prompt | self.chat_model | StrOutputParser()
        raw_answer = chain.invoke(
            {
                "question": question,
                "context": "\n\n".join(contexts[: config.max_reference_documents]),
            }
        ).strip()
        return _strip_thinking_blocks(raw_answer)

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

    def _compute_ragas_metrics(self, details: list[dict], *, enable_faithfulness: bool) -> dict:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
            from ragas.run_config import RunConfig
        except ImportError as exc:
            return {"status": "missing_dependency", "error": str(exc)}
        except Exception as exc:  # pragma: no cover
            return {"status": "failed_to_import", "error": str(exc)}

        dataset = Dataset.from_dict(
            {
                "question": [item["question"] for item in details],
                "answer": [item["predicted_answer"] for item in details],
                "contexts": [item["retrieved_contexts"] for item in details],
                "ground_truth": [item["gold_answer"] for item in details],
            }
        )
        try:
            eval_llm = _SanitizedRagasLLM(
                ModelProviderFactory.create_benchmark_chat_provider().build_chat_model(temperature=0)
            )
            eval_embeddings = ModelProviderFactory.create_benchmark_embedding_provider().build_embedding_client(
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
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
            answer_relevancy_metric = next(metric for name, metric in metric_definitions if name == "answer_relevancy")
            answer_relevancy_metric.strictness = 1
            run_config = RunConfig(
                timeout=300,
                max_retries=3,
                max_workers=4,
            )
            result = evaluate(
                dataset,
                metrics=[metric for _, metric in metric_definitions],
                llm=eval_llm,
                embeddings=eval_embeddings,
                run_config=run_config,
                batch_size=20,
            )
            if hasattr(result, "to_pandas"):
                frame = result.to_pandas()
                payload = {
                    "status": "success",
                    "answer_relevancy": round(float(frame["answer_relevancy"].mean()), 4),
                    "context_precision": round(float(frame["context_precision"].mean()), 4),
                    "context_recall": round(float(frame["context_recall"].mean()), 4),
                }
                payload["faithfulness"] = (
                    round(float(frame["faithfulness"].mean()), 4)
                    if enable_faithfulness and "faithfulness" in frame.columns
                    else "已跳过"
                )
                return payload
            result_dict = dict(result)
            return {
                "status": "success",
                **result_dict,
                "faithfulness": None if enable_faithfulness else "已跳过",
            }
        except Exception as exc:  # pragma: no cover
            logger.exception("benchmark_eval ragas_failed | error=%s", exc)
            return {"status": "failed", "error": str(exc)}

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

            if pre_rank is not None and post_rank is not None:
                valid_rank_rows += 1
                rank_shift_sum += pre_rank - post_rank
                if post_rank < pre_rank:
                    win += 1
                elif post_rank == pre_rank:
                    tie += 1
                else:
                    lose += 1
            elif pre_rank is None and post_rank is not None:
                win += 1
            elif pre_rank is not None and post_rank is None:
                lose += 1
            else:
                tie += 1

        total = len(details)
        return {
            "status": "success",
            "pre_mrr": round(pre_mrr_sum / total, 4),
            "post_mrr": round(post_mrr_sum / total, 4),
            "delta_mrr": round((post_mrr_sum - pre_mrr_sum) / total, 4),
            "pre_hit_rate_at_1": round(pre_hit_at_1 / total, 4),
            "post_hit_rate_at_1": round(post_hit_at_1 / total, 4),
            "delta_hit_rate_at_1": round((post_hit_at_1 - pre_hit_at_1) / total, 4),
            "pre_hit_rate_at_3": round(pre_hit_at_3 / total, 4),
            "post_hit_rate_at_3": round(post_hit_at_3 / total, 4),
            "delta_hit_rate_at_3": round((post_hit_at_3 - pre_hit_at_3) / total, 4),
            "avg_rank_improvement": round(rank_shift_sum / valid_rank_rows, 4) if valid_rank_rows else 0.0,
            "win_rate": round(win / total, 4),
            "tie_rate": round(tie / total, 4),
            "lose_rate": round(lose / total, 4),
        }

    def _format_candidate_context(self, candidate: RetrievalCandidate) -> str:
        return f"Title: {candidate.title}\n\nPassage:\n{candidate.content}"

    def _emit_status(self, callback: Callable[[str], None] | None, message: str) -> None:
        if callback:
            callback(message)

    def _build_rewrite_result(self, question: str, *, enable_query_rewrite: bool) -> QueryRewriteResult:
        if enable_query_rewrite:
            return self.query_rewriter.rewrite(question)
        return QueryRewriteResult(
            original_query=question,
            normalized_query=question,
            retrieval_queries=[question],
        )

    def _get_rewrite_model(self):
        if self.rewrite_model is None:
            self.rewrite_model = ModelProviderFactory.create_rewrite_provider().build_chat_model(
                temperature=0.1,
                extra_body={"reasoning_split": True},
            )
            logger.info(
                "benchmark_eval rewrite_model_initialized | provider=%s | model=%s | base_url=%s | reasoning_split=%s",
                config.rewrite_provider,
                config.rewrite_model_name,
                config.rewrite_base_url,
                True,
            )
        return self.rewrite_model


class _BenchmarkRetriever:
    """在 benchmark corpus 上复用 query rewrite 和多策略检索。"""

    def __init__(
        self,
        subset: str,
        strategy: str,
        corpus_store: BenchmarkCorpusStore,
        manifest: list[dict],
        chunk_config: BenchmarkChunkConfig,
        enable_rerank: bool,
    ):
        self.subset = subset
        self.strategy = strategy
        self.corpus_store = corpus_store
        self.vector_store = corpus_store.get_vector_store(subset, chunk_config=chunk_config)
        self.manifest = manifest
        self.chunk_config = chunk_config
        self.enable_rerank = enable_rerank
        self.rerank_provider = ModelProviderFactory.create_benchmark_rerank_provider() if enable_rerank else None
        self.splitter = _BenchmarkChunker(chunk_config)
        self._bm25_candidates, self._bm25_tokens, self._document_frequencies, self._avg_doc_len = self._build_bm25_index()

    def retrieve(self, rewrite_result, limit: int) -> list[RetrievalCandidate]:
        return self.retrieve_with_debug(rewrite_result, limit)["final_candidates"]

    def retrieve_with_debug(self, rewrite_result, limit: int) -> dict:
        candidates: list[RetrievalCandidate]
        if self.strategy == "vector":
            candidates = self._vector_retrieve(rewrite_result, limit)
        elif self.strategy == "bm25":
            candidates = self._bm25_retrieve(rewrite_result, limit)
        else:
            candidates = self._hybrid_retrieve(rewrite_result, limit)
        pre_rerank_candidates = [self._clone_candidate(candidate) for candidate in candidates[:limit]]
        final_candidates = self._apply_rerank(rewrite_result, candidates, limit)
        return {
            "pre_rerank_candidates": pre_rerank_candidates,
            "final_candidates": final_candidates,
        }

    def _vector_retrieve(self, rewrite_result, limit: int) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        for query in rewrite_result.retrieval_queries:
            for document, score in self.vector_store.search(query, limit):
                metadata = dict(document.metadata)
                candidates.append(
                    RetrievalCandidate(
                        document_id=metadata.get("document_id", ""),
                        title=metadata.get("title", metadata.get("file_name", "")),
                        category=metadata.get("category", self.subset),
                        version=metadata.get("version", "ragbench"),
                        file_name=metadata.get("file_name", ""),
                        content=document.page_content,
                        metadata=metadata,
                        source="vector",
                        retrieval_scores={"vector_raw": float(score)},
                    )
                )
        ranked = self._dedupe_keep_best(candidates, score_key="vector_raw", reverse=False)
        return ranked[:limit]

    def _bm25_retrieve(self, rewrite_result, limit: int) -> list[RetrievalCandidate]:
        scored_candidates: list[tuple[float, RetrievalCandidate]] = []
        for tokens, candidate in zip(self._bm25_tokens, self._bm25_candidates):
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

    def _hybrid_retrieve(self, rewrite_result, limit: int) -> list[RetrievalCandidate]:
        vector_candidates = self._vector_retrieve(rewrite_result, limit)
        bm25_candidates = self._bm25_retrieve(rewrite_result, limit)
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
        for candidate in merged.values():
            candidate.score = (
                config.hybrid_vector_weight * candidate.retrieval_scores.get("vector", 0.0)
                + config.hybrid_bm25_weight * candidate.retrieval_scores.get("bm25", 0.0)
            )
            candidate.source = "hybrid"
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]

    def _apply_rerank(self, rewrite_result, candidates: list[RetrievalCandidate], limit: int) -> list[RetrievalCandidate]:
        if not self.enable_rerank or not self.rerank_provider or not candidates:
            return candidates[:limit]
        try:
            query_text = rewrite_result.normalized_query or rewrite_result.original_query
            results = self.rerank_provider.rerank(
                query=query_text,
                documents=[candidate.content for candidate in candidates],
                top_n=min(limit, len(candidates)),
            )
            if not results:
                return candidates[:limit]
            reranked: list[RetrievalCandidate] = []
            for item in results:
                index = item.get("index")
                if not isinstance(index, int) or index < 0 or index >= len(candidates):
                    continue
                candidate = self._clone_candidate(candidates[index])
                semantic_score = float(item.get("relevance_score", item.get("score", item.get("relevance", 0.0))))
                candidate.retrieval_scores["benchmark_rerank"] = semantic_score
                candidate.score = semantic_score + candidate.score * 0.1
                reranked.append(candidate)
            reranked.sort(key=lambda item: item.score, reverse=True)
            return reranked[:limit] or candidates[:limit]
        except Exception as exc:
            logger.exception("benchmark_eval rerank_failed | subset=%s | error=%s", self.subset, exc)
            return candidates[:limit]

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
                    category=self.subset,
                    version="ragbench",
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
        sorted_candidates = sorted(
            best_by_key.values(),
            key=lambda item: item.retrieval_scores.get(score_key, 0.0),
            reverse=reverse,
        )
        return sorted_candidates

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


class _BenchmarkChunker:
    """benchmark 语料的轻量切片器。"""

    def __init__(self, chunk_config: BenchmarkChunkConfig):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

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
