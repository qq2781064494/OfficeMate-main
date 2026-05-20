"""Decision-ReAct 风格问答使用的粗粒度工具。"""

from __future__ import annotations

from typing import List

import config_data as config
from decision_react.agent import QueryDecisionAgent
from services.chat_service import OfficeMateChatService
from services.rag.answering import (
    TaskAnswer,
    finalize_task_answers,
    generate_task_answers_parallel,
    strip_think_blocks,
)
from services.rag.planning import PlannedTask
from services.rag.query import QueryRewriteResult, build_single_task, rewrite_with_model, rewrite_with_rules
from services.rag.selection import select_task_evidence
from utils.log_tool import get_logger


logger = get_logger("decision_react_tools")


class UnderstandQuestionTool:
    """负责“先理解问题，再决定后面怎么走”的粗粒度工具。"""

    def __init__(self, runtime: OfficeMateChatService):
        self.runtime = runtime
        self.decision_agent = QueryDecisionAgent(self.runtime.get_chat_model)
        self.planner = self.runtime.task_planner

    def run(self, question: str, question_type_key: str, selected_category: str) -> dict:
        decision = self.decision_agent.decide(question, selected_category)
        rewrite_result = (
            rewrite_with_model(question, self.runtime.query_rewriter)
            if decision.use_rewrite
            else rewrite_with_rules(question)
        )

        effective_category = selected_category
        if effective_category == "全部" and decision.suggested_category != "全部":
            # 这里仍然让 planner 保持“全部”视角，避免过早缩窄分类；
            # 但后续单任务直答时可以使用推荐分类。
            effective_category = "全部"

        if decision.use_planner:
            planned_tasks = self.planner.plan(
                rewrite_result.normalized_query,
                question_type_key,
                effective_category,
            )
        else:
            planned_tasks = [
                build_single_task(
                    question=rewrite_result.normalized_query,
                    question_type_key=question_type_key,
                    selected_category=selected_category,
                    suggested_category=decision.suggested_category,
                )
            ]

        logger.info(
            "react_tool understand_question_completed | complexity=%s | use_rewrite=%s | use_planner=%s | normalized=%s | task_ids=%s",
            decision.complexity,
            decision.use_rewrite,
            decision.use_planner,
            rewrite_result.normalized_query,
            [task.task_id for task in planned_tasks],
        )
        return {
            "decision": decision,
            "rewrite_result": rewrite_result,
            "planned_tasks": planned_tasks,
        }


class RetrieveEvidenceTool:
    """负责共享召回、按任务重排和引用整理。"""

    def __init__(self, runtime: OfficeMateChatService):
        self.runtime = runtime

    def run(self, rewrite_result: QueryRewriteResult, planned_tasks: List[PlannedTask], selected_category: str) -> dict:
        evidence = select_task_evidence(
            rewrite_result=rewrite_result,
            planned_tasks=planned_tasks,
            selected_category=selected_category,
            top_k=config.max_reference_documents,
            query_rewriter=self.runtime.query_rewriter,
            retrieval_coordinator=self.runtime.get_retrieval_coordinator(),
            reranker=self.runtime.reranker,
        )

        logger.info(
            "react_tool retrieve_evidence_completed | categories=%s | task_ids=%s | reference_count=%s",
            sorted({item.task_category for item in evidence["task_plans"]}),
            [item.planned_task.task_id for item in evidence["task_plans"]],
            len(evidence["references"]),
        )
        return evidence


class GenerateAnswersTool:
    """负责并行生成多个子任务答案。"""

    def __init__(self, runtime: OfficeMateChatService):
        self.runtime = runtime

    def run(self, question, question_type, history, task_plans, status_callback=None) -> List[TaskAnswer]:
        return generate_task_answers_parallel(
            question=question,
            question_type=question_type,
            history=history,
            task_plans=task_plans,
            session_id="decision_react",
            prompt_template=self.runtime.task_prompt_template,
            chat_model_factory=self.runtime.get_chat_model,
            question_type_labels=config.QUESTION_TYPE_LABELS,
            logger=logger,
            status_callback=status_callback,
            parallel_workers=config.parallel_subtask_workers,
            log_prefix="react_tool generate_answers",
        )


class FinalizeAnswerTool:
    """负责输出最终单份答案。

    这里把两种结束路径收敛到一个工具里：
    - 需要汇总模型时：调用 synthesizer
    - 不需要汇总模型时：把多个子答案按章节做结构化合并，而不是生硬拼接
    """

    def __init__(self, runtime: OfficeMateChatService):
        self.runtime = runtime

    def run(
        self,
        question: str,
        task_answers: List[TaskAnswer],
        use_synthesize: bool,
        task_reference_groups: List[dict],
    ) -> tuple[str, dict]:
        answer, metadata = finalize_task_answers(
            question=question,
            task_answers=task_answers,
            use_synthesize=use_synthesize,
            chat_model_factory=self.runtime.get_chat_model,
            logger=logger,
            task_reference_groups=task_reference_groups,
        )
        return strip_think_blocks(answer), metadata
