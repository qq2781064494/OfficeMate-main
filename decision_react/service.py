"""Decision-Guided ReAct 风格问答服务。"""

from __future__ import annotations

import json
from time import perf_counter
from typing import List

import config_data as config
from decision_react.schemas import DecisionReactResult, ToolTrace
from decision_react.tools import FinalizeAnswerTool, GenerateAnswersTool, RetrieveEvidenceTool, UnderstandQuestionTool
from services.rag.answering import build_final_answer
from services.chat_service import OfficeMateChatService
from utils.log_tool import get_logger


logger = get_logger("decision_react_service")


class DecisionReactService:
    """一条“先决策、再执行工具”的问答链路。

    和现有固定流水线的区别：
    - 现有链路默认每个问题都先走 rewrite / planner / synthesize
    - 这里先由决策 Agent 判断问题复杂度
    - 再决定应该调用哪些工具

    这样更适合处理：
    - 简单问题走轻路径
    - 复杂问题走重路径
    """

    def __init__(self):
        self.runtime = OfficeMateChatService()
        self.understand_question_tool = UnderstandQuestionTool(self.runtime)
        self.retrieve_evidence_tool = RetrieveEvidenceTool(self.runtime)
        self.generate_answers_tool = GenerateAnswersTool(self.runtime)
        self.finalize_answer_tool = FinalizeAnswerTool(self.runtime)

    def answer_question(self, question: str, session_id: str, category: str = "全部", status_callback=None) -> DecisionReactResult:
        """执行一条“先决策，再按工具步骤走”的问答流程。"""
        question_type_key = self.runtime.infer_question_type(question)
        question_type = config.QUESTION_TYPE_LABELS[question_type_key]
        history = self.runtime.build_history(session_id)
        trace: List[ToolTrace] = []

        understanding = self._run_step(
            trace,
            "understand_question_tool",
            lambda: self.understand_question_tool.run(question, question_type_key, category),
            lambda result: {
                "decision": result["decision"].model_dump(),
                "normalized_query": result["rewrite_result"].normalized_query,
                "retrieval_queries": result["rewrite_result"].retrieval_queries,
                "planned_tasks": [
                    {
                        "task_id": task.task_id,
                        "description": task.description,
                        "category": task.category,
                        "intent": task.intent,
                    }
                    for task in result["planned_tasks"]
                ],
            },
        )
        decision = understanding["decision"]
        rewrite_result = understanding["rewrite_result"]
        planned_tasks = understanding["planned_tasks"]

        selected_category = category
        # 当前实验链路里，Decision Agent 给出的建议分类更多用于 trace 观察。
        # 这里先保持页面显式选择的分类优先，避免实验模式过早替用户缩窄范围。
        if selected_category == "全部" and decision.suggested_category != "全部":
            selected_category = "全部"

        if status_callback:
            status_callback("正在检索证据并整理候选材料...")

        evidence = self._run_step(
            trace,
            "retrieve_evidence_tool",
            lambda: self.retrieve_evidence_tool.run(rewrite_result, planned_tasks, category),
            lambda result: {
                "task_ids": [item.planned_task.task_id for item in result["task_plans"]],
                "reference_count": len(result["references"]),
                "categories": sorted({item.task_category for item in result["task_plans"]}),
            },
        )
        task_plans = evidence["task_plans"]
        references = evidence["references"]
        task_reference_groups = evidence["task_reference_groups"]

        if not references:
            # 即使没有找到证据，也照样记录 qa_log，方便后续分析“没答出来”的问题。
            answer = config.NO_EVIDENCE_MESSAGE + "\n\n### 引用文档\n无"
            qa_log = self.runtime.storage.add_qa_log(
                {
                    "session_id": session_id,
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "question_type": question_type,
                    "source_docs": [],
                    "mode": "decision_react",
                    "decision": decision.model_dump(),
                    "trace": [item.__dict__ for item in trace],
                }
            )
            return DecisionReactResult(
                answer=answer,
                question_type=question_type,
                qa_log_id=qa_log["id"],
                source_docs=[],
                decision=decision.model_dump(),
                trace=[item.__dict__ for item in trace],
            )

        task_answers = self._run_step(
            trace,
            "generate_answers_tool",
            lambda: self.generate_answers_tool.run(
                question=question,
                question_type=question_type,
                history=history,
                task_plans=task_plans,
                status_callback=status_callback,
            ),
            lambda result: [
                {
                    "task_id": item.task_id,
                    "category": item.category,
                    "answer_length": len(item.answer),
                }
                for item in result
            ],
        )

        if not task_answers:
            answer_body = "已检索到相关材料，但本次未能成功生成最终答案，请稍后重试。"
        else:
            answer_body = self._run_step(
                trace,
                "finalize_answer_tool",
                lambda: self.finalize_answer_tool.run(
                    question=question,
                    task_answers=task_answers,
                    use_synthesize=decision.use_synthesize,
                    task_reference_groups=task_reference_groups,
                ),
                lambda result: result[1],
            )[0]

        full_answer = build_final_answer(answer_body, references, include_references=True)
        qa_log = self.runtime.storage.add_qa_log(
            {
                "session_id": session_id,
                "question": question,
                "answer": full_answer,
                "category": category,
                "question_type": question_type,
                "source_docs": references,
                "mode": "decision_react",
                "decision": decision.model_dump(),
                "trace": [item.__dict__ for item in trace],
            }
        )
        logger.info(
            "decision_react answer_completed | session_id=%s | qa_log_id=%s | decision=%s",
            session_id,
            qa_log["id"],
            decision.model_dump(),
        )

        return DecisionReactResult(
            answer=full_answer,
            question_type=question_type,
            qa_log_id=qa_log["id"],
            source_docs=references,
            decision=decision.model_dump(),
            trace=[item.__dict__ for item in trace],
        )

    def _run_step(self, trace, step_name, runner, serializer):
        """统一执行一个工具步骤，并记录耗时与摘要。"""
        start = perf_counter()
        result = runner()
        duration_ms = int((perf_counter() - start) * 1000)
        summary = serializer(result)
        summary_text = summary if isinstance(summary, str) else json.dumps(summary, ensure_ascii=False)
        trace.append(ToolTrace(step=step_name, summary=summary_text, duration_ms=duration_ms, metadata={"result": summary}))
        return result
