"""离线评测服务。

用于评估混合检索在一组标准问题上的命中情况，支持简历中常提到的：
- Recall@K
- MRR
- Hit Rate
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import config_data as config
from services.benchmark_results import BenchmarkResultStore
from services.chat_service import OfficeMateChatService
from services.rag.query import QueryRewriteResult
from services.rag.retrieval import HybridRetriever


@dataclass
class LocalEvalConfig:
    sample_label: str
    sample_path: Path
    question_limit: int = 0
    enable_query_rewrite: bool = True
    enable_rerank: bool = True
    enable_ragas: bool = True


class EvaluationService:
    """对检索链路做轻量离线评测。

    这是把 Demo 升级成“可运营 AI 应用”的关键一步。
    没有评测，你只能凭感觉说效果变好了；
    有了评测，你就能用指标说明混合检索的收益。
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        sample_path: Path | None = None,
        result_store: BenchmarkResultStore | None = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.sample_path = sample_path or config.EVALUATION_SAMPLE_PATH
        self.chat_service = OfficeMateChatService()
        self.result_store = result_store or BenchmarkResultStore()

    def evaluate_recall(self, k: int = 5) -> Dict[str, float]:
        """按评测集统计 Recall@K、MRR 和 Hit Rate。

        指标解释：
        - Recall@K: 相关文档有没有被召回到前 K 条
        - MRR: 第一条命中文档排得有多靠前
        - Hit Rate: 每个问题是否至少命中一条正确结果
        """
        samples = self._load_samples()
        if not samples:
            return {"recall_at_k": 0.0, "mrr": 0.0, "hit_rate": 0.0, "sample_count": 0}

        hit_count = 0
        reciprocal_rank_sum = 0.0
        recall_sum = 0.0

        for sample in samples:
            query = str(sample["query"]).strip()
            rewrite_result = QueryRewriteResult(
                original_query=query,
                normalized_query=query,
                retrieval_queries=[query],
            )
            candidates = self.retriever.retrieve(
                rewrite_result,
                category=sample.get("category", "全部"),
                limit=k,
            )
            candidate_titles = [candidate.title for candidate in candidates]
            expected_titles = set(sample.get("expected_titles", []))

            first_hit_rank = None
            matched_titles = set()
            for index, title in enumerate(candidate_titles, start=1):
                if title in expected_titles:
                    matched_titles.add(title)
                    if first_hit_rank is None:
                        first_hit_rank = index

            if matched_titles:
                hit_count += 1
            if first_hit_rank:
                reciprocal_rank_sum += 1 / first_hit_rank
            if expected_titles:
                recall_sum += len(matched_titles) / len(expected_titles)

        sample_count = len(samples)
        return {
            "recall_at_k": round(recall_sum / sample_count, 4),
            "mrr": round(reciprocal_rank_sum / sample_count, 4),
            "hit_rate": round(hit_count / sample_count, 4),
            "sample_count": sample_count,
        }

    def _load_samples(self) -> List[Dict[str, object]]:
        """读取评测样本。

        样本格式非常简单，便于你后面自己继续扩充：
        - query
        - category
        - expected_titles
        """
        if not self.sample_path.exists():
            return []
        return json.loads(self.sample_path.read_text(encoding="utf-8"))

    def run_local_evaluation(
        self,
        eval_config: LocalEvalConfig,
        status_callback: Callable[[str], None] | None = None,
    ) -> Dict[str, object]:
        result = self.evaluate_with_chat_rag(eval_config, status_callback=status_callback)
        summary = dict(result["summary"])
        summary["retrieval_metrics"] = result["retrieval_metrics"]
        summary["rerank_metrics"] = result["rerank_metrics"]
        summary["ragas_metrics"] = result["ragas_metrics"]
        run_record = self.result_store.save_run(summary, result["details"])
        summary["run_id"] = run_record["run_id"]
        summary["detail_path"] = run_record["detail_path"]
        return summary

    def evaluate_with_chat_rag(
        self,
        eval_config: LocalEvalConfig,
        status_callback: Callable[[str], None] | None = None,
    ) -> Dict[str, object]:
        """使用主问答链路批量评测本地样本。

        这里不会走 benchmark 专用 answer prompt，
        而是直接调用 ChatService.answer_question，
        因此生成答案与聊天页保持一致。
        """
        self.sample_path = eval_config.sample_path
        samples = self._load_samples()
        if eval_config.question_limit:
            samples = samples[: min(eval_config.question_limit, len(samples))]
        if not samples:
            return {
                "sample_count": 0,
                "retrieval_metrics": {
                    "recall_at_1": 0.0,
                    "recall_at_3": 0.0,
                    "recall_at_5": 0.0,
                    "hit_rate_at_1": 0.0,
                    "hit_rate_at_3": 0.0,
                    "hit_rate_at_5": 0.0,
                    "mrr": 0.0,
                },
                "rerank_metrics": {"status": "disabled" if not eval_config.enable_rerank else "empty"},
                "ragas_metrics": {"status": "empty"},
                "details": [],
            }

        details: List[Dict[str, object]] = []
        for index, sample in enumerate(samples, start=1):
            if status_callback:
                status_callback(f"正在处理第 {index}/{len(samples)} 题...")
            result = self.chat_service.answer_question(
                question=str(sample["query"]).strip(),
                session_id=f"offline_eval_{index}",
                category=str(sample.get("category", "全部")),
                use_history=False,
                persist_log=False,
                include_references=False,
                enable_query_rewrite=eval_config.enable_query_rewrite,
                enable_rerank=eval_config.enable_rerank,
            )
            source_docs = result.get("source_docs", [])
            retrieved_titles = [item.get("title", "") for item in source_docs]
            pre_rerank_titles = list(result.get("pre_rerank_titles", []))
            expected_titles = sample.get("expected_titles", [])

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

            details.append(
                {
                    "question_id": index,
                    "question": sample["query"],
                    "category": sample.get("category", "全部"),
                    "normalized_query": result.get("normalized_query", ""),
                    "retrieval_queries": result.get("retrieval_queries", []),
                    "matched_terms": result.get("matched_terms", []),
                    "expected_titles": expected_titles,
                    "pre_rerank_titles": pre_rerank_titles,
                    "retrieved_titles": retrieved_titles,
                    "gold_answer": sample.get("gold_answer", ""),
                    "predicted_answer": result.get("answer", ""),
                    "question_type": result.get("question_type", ""),
                    "source_docs": source_docs,
                    "retrieved_contexts": result.get("retrieved_contexts", []),
                    "pre_rerank_hit": pre_rerank_first_hit_rank is not None,
                    "pre_rerank_first_hit_rank": pre_rerank_first_hit_rank,
                    "retrieval_hit": first_hit_rank is not None,
                    "first_hit_rank": first_hit_rank,
                }
            )

        retrieval_metrics = self._compute_chat_rag_retrieval_metrics(details)
        rerank_metrics = (
            self._compute_chat_rag_rerank_metrics(details) if eval_config.enable_rerank else {"status": "disabled"}
        )
        ragas_metrics = self._compute_chat_rag_ragas_metrics(details) if eval_config.enable_ragas else {"status": "disabled"}
        return {
            "sample_count": len(details),
            "retrieval_metrics": retrieval_metrics,
            "rerank_metrics": rerank_metrics,
            "ragas_metrics": ragas_metrics,
            "details": details,
            "summary": {
                "subset": eval_config.sample_label,
                "split": "sample_docs",
                "mode": "local_chat_rag",
                "retriever_strategy": "main_chat_rag",
                "top_k": config.max_reference_documents,
                "question_limit": eval_config.question_limit or len(details),
                "enable_query_rewrite": eval_config.enable_query_rewrite,
                "enable_rerank": eval_config.enable_rerank,
                "question_count": len(details),
                "document_count": len(config.SAMPLE_DOCS),
            },
        }

    def _compute_chat_rag_retrieval_metrics(self, details: List[Dict[str, object]]) -> Dict[str, float]:
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
            hit_count = 0
            recall_sum = 0.0
            for detail in details:
                expected_titles = set(detail.get("expected_titles", []))
                retrieved_titles = detail.get("retrieved_titles", [])[:k]
                matched_titles = {title for title in retrieved_titles if title in expected_titles}
                if matched_titles:
                    hit_count += 1
                if expected_titles:
                    recall_sum += len(matched_titles) / len(expected_titles)
            metrics[f"recall_at_{k}"] = round(recall_sum / len(details), 4)
            metrics[f"hit_rate_at_{k}"] = round(hit_count / len(details), 4)

        reciprocal_rank_sum = 0.0
        for detail in details:
            first_hit_rank = detail.get("first_hit_rank")
            if first_hit_rank:
                reciprocal_rank_sum += 1 / first_hit_rank
        metrics["mrr"] = round(reciprocal_rank_sum / len(details), 4)
        return metrics

    def _compute_chat_rag_rerank_metrics(self, details: List[Dict[str, object]]) -> Dict[str, object]:
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
            if pre_rank and pre_rank <= 3:
                pre_hit_at_3 += 1
            if post_rank and post_rank <= 3:
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

    def _compute_chat_rag_ragas_metrics(self, details: List[Dict[str, object]]) -> Dict[str, object]:
        try:
            from copy import deepcopy

            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
            from ragas.run_config import RunConfig

            from services.benchmark_eval_service import _SanitizedRagasLLM
            from services.model_provider import ModelProviderFactory
        except ImportError as exc:
            return {"status": "missing_dependency", "error": str(exc)}
        except Exception as exc:
            return {"status": "failed_to_import", "error": str(exc)}

        dataset = Dataset.from_dict(
            {
                "question": [str(item["question"]) for item in details],
                "answer": [str(item["predicted_answer"]) for item in details],
                "contexts": [list(item.get("retrieved_contexts", [])) for item in details],
                "ground_truth": [str(item["gold_answer"]) for item in details],
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
            faithfulness_metric = deepcopy(faithfulness)
            answer_relevancy_metric = deepcopy(answer_relevancy)
            context_precision_metric = deepcopy(context_precision)
            context_recall_metric = deepcopy(context_recall)
            answer_relevancy_metric.strictness = 1
            run_config = RunConfig(
                timeout=300,
                max_retries=3,
                max_workers=1,
            )
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness_metric,
                    answer_relevancy_metric,
                    context_precision_metric,
                    context_recall_metric,
                ],
                llm=eval_llm,
                embeddings=eval_embeddings,
                run_config=run_config,
                batch_size=1,
            )
            if hasattr(result, "to_pandas"):
                frame = result.to_pandas()
                return {
                    "status": "success",
                    "faithfulness": round(float(frame["faithfulness"].mean()), 4),
                    "answer_relevancy": round(float(frame["answer_relevancy"].mean()), 4),
                    "context_precision": round(float(frame["context_precision"].mean()), 4),
                    "context_recall": round(float(frame["context_recall"].mean()), 4),
                }
            return {"status": "success", **dict(result)}
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}
