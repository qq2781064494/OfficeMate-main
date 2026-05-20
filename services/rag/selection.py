"""selection 阶段的共享 helper。

如果说 retrieval 是“尽量多找一些可能相关的材料”，
那么 selection 就是在这些候选里继续做筛选、聚合和预算分配，
最终把有限的引用名额留给更重要的证据。
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import config_data as config
from services.model_provider import ModelProviderFactory
from services.rag.answering import build_references
from services.rag.contracts import TaskInput, TaskPlan
from services.rag.planning import PlannedTask
from services.rag.query import QueryRewriteResult, QueryRewriter
from services.rag.retrieval import RetrievalCandidate, RetrievalCoordinator
from utils.log_tool import get_logger


logger = get_logger("rag_selection")


class SimpleReranker:
    """结合语义重排和规则兜底的轻量重排器。"""

    def __init__(self):
        self.model_name = config.rerank_model_name
        self.provider = ModelProviderFactory.create_rerank_provider()

    def rerank(
        self,
        rewrite_result: QueryRewriteResult,
        task: PlannedTask,
        candidates: list[RetrievalCandidate],
        limit: int,
    ) -> list[RetrievalCandidate]:
        """优先尝试模型重排，失败时退回规则重排。"""
        model_reranked = self._model_rerank(rewrite_result, task, candidates, limit)
        if model_reranked is not None:
            logger.info(
                "reranker model_completed | task_id=%s | task_category=%s | input_candidates=%s | output_candidates=%s | model=%s",
                task.task_id,
                task.category,
                len(candidates),
                len(model_reranked),
                self.model_name,
            )
            logger.debug(
                "reranker model_top_candidates=%s",
                [
                    {
                        "title": item.title,
                        "category": item.category,
                        "score": round(item.score, 4),
                    }
                    for item in model_reranked[: min(5, len(model_reranked))]
                ],
            )
            return model_reranked

        return self._rule_rerank(rewrite_result, task, candidates, limit)

    def _rule_rerank(
        self,
        rewrite_result: QueryRewriteResult,
        task: PlannedTask,
        candidates: list[RetrievalCandidate],
        limit: int,
    ) -> list[RetrievalCandidate]:
        """规则重排：在原始检索分数上叠加业务 bonus。"""
        reranked: list[RetrievalCandidate] = []
        for candidate in candidates:
            candidate = deepcopy(candidate)
            final_score = candidate.score + self._compute_candidate_bonus(candidate, rewrite_result, task)
            candidate.score = final_score
            candidate.retrieval_scores["rerank"] = final_score
            reranked.append(candidate)

        reranked.sort(key=lambda item: item.score, reverse=True)
        logger.info(
            "reranker completed | task_id=%s | task_category=%s | input_candidates=%s | output_candidates=%s",
            task.task_id,
            task.category,
            len(candidates),
            min(limit, len(reranked)),
        )
        logger.debug(
            "reranker top_candidates=%s",
            [
                {
                    "title": item.title,
                    "category": item.category,
                    "score": round(item.score, 4),
                }
                for item in reranked[: min(5, len(reranked))]
            ],
        )
        return reranked[:limit]

    def _model_rerank(
        self,
        rewrite_result: QueryRewriteResult,
        task: PlannedTask,
        candidates: list[RetrievalCandidate],
        limit: int,
    ) -> list[RetrievalCandidate] | None:
        """调用独立 rerank 模型做语义重排。"""
        if not candidates:
            return None

        query_text = rewrite_result.normalized_query or rewrite_result.original_query
        documents = [
            f"标题：{candidate.title}\n分类：{candidate.category}\n文件：{candidate.file_name}\n内容：{candidate.content}"
            for candidate in candidates
        ]
        try:
            parsed_results = self.provider.rerank(
                query=query_text,
                documents=documents,
                top_n=min(limit, len(documents)),
            )
            if not parsed_results:
                logger.warning(
                    "reranker model_empty_results_fallback | task_id=%s | provider=%s | model=%s",
                    task.task_id,
                    config.rerank_provider,
                    self.model_name,
                )
                return None

            reranked_candidates: list[RetrievalCandidate] = []
            for item in parsed_results:
                index = item.get("index")
                if not isinstance(index, int) or index < 0 or index >= len(candidates):
                    continue
                candidate = deepcopy(candidates[index])
                semantic_score = float(item.get("relevance_score", item.get("score", item.get("relevance", 0.0))))
                # 保留少量原始检索分数和规则 bonus，避免语义模型完全一票否决其他信号。
                candidate.score = semantic_score + candidate.score * 0.1 + self._compute_candidate_bonus(candidate, rewrite_result, task)
                candidate.retrieval_scores["model_rerank"] = semantic_score
                reranked_candidates.append(candidate)

            reranked_candidates.sort(key=lambda item: item.score, reverse=True)
            return reranked_candidates[:limit]
        except Exception as exc:
            logger.exception(
                "reranker model_failed_fallback_to_rule | task_id=%s | provider=%s | model=%s | error=%s",
                task.task_id,
                config.rerank_provider,
                self.model_name,
                exc,
            )
            return None

    def _compute_candidate_bonus(
        self,
        candidate: RetrievalCandidate,
        rewrite_result: QueryRewriteResult,
        task: PlannedTask,
    ) -> float:
        """给更贴近任务目标的候选增加附加分。"""
        bonus = 0.0
        title_text = f"{candidate.title} {candidate.file_name}".lower()
        content_text = candidate.content.lower()
        query_text = " ".join(rewrite_result.retrieval_queries).lower()

        if task.category != "全部" and candidate.category == task.category:
            bonus += 0.15
        for hint in task.hints:
            lowered_hint = hint.lower()
            if lowered_hint in title_text:
                bonus += 0.08
            if lowered_hint in content_text:
                bonus += 0.04
        for query in rewrite_result.retrieval_queries:
            lowered_query = query.lower()
            if lowered_query in title_text:
                bonus += 0.06

        onboarding_title_keywords = ("入职", "权限开通", "账号开通", "新员工", "门禁", "工位")
        if any(keyword in query_text for keyword in onboarding_title_keywords) and any(keyword in title_text for keyword in onboarding_title_keywords):
            bonus += 0.18

        timeline_keywords = ("最晚", "时间点", "时间节点", "何时", "截止", "时限")
        timeline_evidence_keywords = ("至少", "当天", "前开通", "工作日内", "12:00", "加急", "时限")
        if any(keyword in query_text for keyword in timeline_keywords) and any(keyword in content_text for keyword in timeline_evidence_keywords):
            bonus += 0.16

        if task.intent == "process_guide" and any(keyword in content_text for keyword in ("步骤", "流程", "审批")):
            bonus += 0.08
        if task.intent == "material_list" and any(keyword in content_text for keyword in ("材料", "附件", "提交")):
            bonus += 0.08
        return bonus


def aggregate_document_candidates(
    candidates: list[RetrievalCandidate],
    max_chunks_per_doc: int = 3,
) -> list[RetrievalCandidate]:
    """把同一文档的多 chunk 命中压成一个候选，避免重复占用多个 top_k 名额。"""
    if not candidates:
        return []

    # 先按文档聚合，避免同一篇文档的多个相邻 chunk 挤占多个 top_k 名额。
    candidates_by_doc: dict[str, list[RetrievalCandidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_doc[candidate.document_id].append(candidate)

    aggregated_candidates: list[RetrievalCandidate] = []
    for document_id, doc_candidates in candidates_by_doc.items():
        unique_candidates = dedupe_candidates_by_chunk(doc_candidates)
        if len(unique_candidates) <= 1:
            aggregated_candidates.extend(unique_candidates)
            continue

        top_candidates = sorted(
            unique_candidates,
            key=lambda item: item.score,
            reverse=True,
        )[: max(1, int(max_chunks_per_doc))]
        ordered_candidates = sorted(
            top_candidates,
            key=lambda item: normalize_chunk_index(item.metadata.get("chunk_index")),
        )
        best_candidate = max(unique_candidates, key=lambda item: item.score)
        aggregated_candidates.append(
            RetrievalCandidate(
                document_id=document_id,
                title=best_candidate.title,
                category=best_candidate.category,
                version=best_candidate.version,
                file_name=best_candidate.file_name,
                content=merge_candidate_contents(ordered_candidates),
                metadata={
                    **dict(best_candidate.metadata),
                    "chunk_index": build_aggregated_chunk_key(ordered_candidates),
                    "is_document_aggregated": True,
                    "aggregated_chunk_indices": [
                        candidate.metadata.get("chunk_index")
                        for candidate in ordered_candidates
                    ],
                    "aggregated_chunk_count": len(unique_candidates),
                },
                score=compute_aggregated_candidate_score(best_candidate, unique_candidates),
                source=best_candidate.source,
                retrieval_scores={
                    **best_candidate.retrieval_scores,
                    "document_aggregated": 1.0,
                    "aggregated_max_score": round(float(best_candidate.score), 4),
                    "aggregated_chunk_count": float(len(unique_candidates)),
                },
            )
        )

    aggregated_candidates.sort(key=lambda item: item.score, reverse=True)
    logger.info(
        "document_candidate aggregation_completed | raw_candidates=%s | aggregated_candidates=%s",
        len(candidates),
        len(aggregated_candidates),
    )
    return aggregated_candidates


def allocate_task_candidate_budgets(task_inputs: list[object], total_budget: int) -> dict[str, int]:
    """为多个子任务分配统一的候选预算，确保最终总量不超过全局 top_k。"""
    normalized_budget = max(0, int(total_budget))
    budgets = {
        item.planned_task.task_id: 0
        for item in task_inputs
    }
    if normalized_budget <= 0 or not task_inputs:
        return budgets

    prioritized_tasks = sorted(
        [item for item in task_inputs if item.candidates],
        key=lambda item: (
            -(max(candidate.score for candidate in item.candidates) if item.candidates else float("-inf")),
            -len(item.candidates),
            item.planned_task.task_id,
        ),
    )
    if not prioritized_tasks:
        return budgets

    for item in prioritized_tasks[:normalized_budget]:
        budgets[item.planned_task.task_id] = 1

    remaining_budget = normalized_budget - sum(budgets.values())
    while remaining_budget > 0:
        allocated_in_round = False
        for item in prioritized_tasks:
            task_id = item.planned_task.task_id
            if budgets[task_id] >= len(item.candidates):
                continue
            budgets[task_id] += 1
            remaining_budget -= 1
            allocated_in_round = True
            if remaining_budget <= 0:
                break
        if not allocated_in_round:
            break

    logger.info(
        "task_budget allocation_completed | total_budget=%s | task_budgets=%s",
        normalized_budget,
        budgets,
    )
    return budgets


def dedupe_candidates_by_chunk(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    """按 chunk 去重，重复时保留分数更高的一条。"""
    deduped: dict[str, RetrievalCandidate] = {}
    for candidate in candidates:
        chunk_key = candidate.unique_key
        existing = deduped.get(chunk_key)
        if existing is None or candidate.score > existing.score:
            deduped[chunk_key] = candidate
    return sorted(deduped.values(), key=lambda item: item.score, reverse=True)


def normalize_chunk_index(chunk_index) -> tuple[int, str]:
    """把 chunk 序号统一转成可排序形式。"""
    if isinstance(chunk_index, int):
        return (chunk_index, str(chunk_index))
    if isinstance(chunk_index, str) and chunk_index.isdigit():
        return (int(chunk_index), chunk_index)
    return (10**9, str(chunk_index or ""))


def compute_aggregated_candidate_score(
    best_candidate: RetrievalCandidate,
    unique_candidates: list[RetrievalCandidate],
) -> float:
    """为聚合后的“文档级候选”计算最终分数。"""
    bonus = 0.03 * min(len(unique_candidates) - 1, 3)
    normalized_indices = [
        normalize_chunk_index(candidate.metadata.get("chunk_index"))[0]
        for candidate in unique_candidates
        if normalize_chunk_index(candidate.metadata.get("chunk_index"))[0] < 10**9
    ]
    if len(normalized_indices) >= 2:
        sorted_indices = sorted(set(normalized_indices))
        if all((right - left) <= 1 for left, right in zip(sorted_indices, sorted_indices[1:])):
            bonus += 0.03
    return best_candidate.score + bonus


def build_aggregated_chunk_key(candidates: list[RetrievalCandidate]) -> str:
    """给聚合后的文档候选生成可追踪的 chunk key。"""
    indices = [str(candidate.metadata.get("chunk_index", "")) for candidate in candidates]
    return "agg:" + ",".join(indices)


def merge_candidate_contents(candidates: list[RetrievalCandidate]) -> str:
    """把多个 chunk 的正文拼成一个更完整的上下文块。"""
    merged_blocks = []
    for candidate in candidates:
        chunk_index = candidate.metadata.get("chunk_index", "unknown")
        merged_blocks.append(f"[片段 {chunk_index}]\n{candidate.content.strip()}")
    return "\n\n".join(merged_blocks)


def select_task_evidence(
    rewrite_result: QueryRewriteResult,
    planned_tasks: list[PlannedTask],
    selected_category: str,
    top_k: int,
    query_rewriter: QueryRewriter,
    retrieval_coordinator: RetrievalCoordinator,
    reranker: SimpleReranker,
    *,
    enable_rerank: bool = True,
) -> dict:
    """统一执行共享检索、task 级重排、证据预算和引用整理。"""
    normalized_top_k = max(1, int(top_k or config.max_reference_documents))
    shared_candidates_map, category_rewrite_map = retrieval_coordinator.collect_shared_candidates(
        rewrite_result=rewrite_result,
        planned_tasks=planned_tasks,
        selected_category=selected_category,
        limit=config.hybrid_fetch_k,
    )

    task_inputs: list[TaskInput] = []
    pre_rerank_references: list[dict] = []
    seen_pre_rerank_document_ids = set()
    # 先把每个任务需要的候选池准备好，但此时还不决定每个任务最终能拿到几个名额。
    for task in planned_tasks:
        task_category = selected_category if selected_category != "全部" else task.category
        task_rewrite = query_rewriter.narrow_to_hints(
            category_rewrite_map.get(task_category, rewrite_result),
            task.hints,
        )
        candidates = aggregate_document_candidates(shared_candidates_map.get(task_category, []))
        for item in build_references(candidates[:normalized_top_k], limit=normalized_top_k):
            if item["document_id"] in seen_pre_rerank_document_ids:
                continue
            seen_pre_rerank_document_ids.add(item["document_id"])
            pre_rerank_references.append(item)
        task_inputs.append(
            TaskInput(
                planned_task=task,
                task_category=task_category,
                task_rewrite_result=task_rewrite,
                candidates=candidates,
            )
        )

    task_budgets = allocate_task_candidate_budgets(task_inputs, total_budget=normalized_top_k)
    task_plans: list[TaskPlan] = []
    references: list[dict] = []
    task_reference_groups: list[dict] = []
    retrieved_contexts: list[str] = []
    seen_document_ids = set()
    seen_context_keys = set()

    # 再根据预算逐个任务做 rerank 和最终证据截断。
    for item in task_inputs:
        candidate_budget = task_budgets.get(item.planned_task.task_id, 0)
        if candidate_budget <= 0:
            reranked = []
        elif not enable_rerank:
            reranked = item.candidates[:candidate_budget]
        else:
            reranked = reranker.rerank(
                item.task_rewrite_result,
                item.planned_task,
                item.candidates,
                limit=candidate_budget,
            )

        task_references = build_references(reranked, limit=normalized_top_k)
        task_reference_groups.append(
            {
                "task_id": item.planned_task.task_id,
                "category": item.task_category,
                "document_ids": [ref["document_id"] for ref in task_references],
            }
        )
        for ref in task_references:
            if ref["document_id"] in seen_document_ids:
                continue
            seen_document_ids.add(ref["document_id"])
            references.append(ref)
        for candidate in reranked:
            context_key = candidate.unique_key
            if context_key in seen_context_keys:
                continue
            seen_context_keys.add(context_key)
            retrieved_contexts.append(
                f"标题：{candidate.title}\n分类：{candidate.category}\n版本：{candidate.version}\n内容：{candidate.content}"
            )

        task_plans.append(
            TaskPlan(
                planned_task=item.planned_task,
                task_category=item.task_category,
                candidates=reranked,
            )
        )

    return {
        "task_plans": task_plans,
        "references": references,
        "task_reference_groups": task_reference_groups,
        "pre_rerank_titles": [item["title"] for item in pre_rerank_references],
        "retrieved_titles": [item["title"] for item in references],
        "retrieved_contexts": retrieved_contexts,
    }
