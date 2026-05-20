"""Agent-ReAct-RAG 的 4 个粗粒度工具。"""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

import config_data as config
from agent_react_rag.schemas import ToolTrace
from services.chat_service import OfficeMateChatService
from services.rag.answering import build_context, build_final_answer, strip_think_blocks
from services.rag.planning import PlannedTask
from services.rag.query import QueryRewriteResult, build_single_task, rewrite_with_model, rewrite_with_rules
from services.rag.selection import select_task_evidence
from utils.log_tool import get_logger


logger = get_logger("agent_react_rag_tools")


class AgentReactToolbox:
    """把现有 RAG 能力封装成 create_agent 可调用工具。

    设计原则：
    - 工具保持粗粒度，贴近业务阶段
    - 工具之间通过共享 state 传递中间结果，而不是把大段 JSON 反复塞回模型
    - 即使 Agent 跳过 rewrite 或 plan，后面的 retrieve / final answer 仍然能工作
    """

    def __init__(self, runtime: OfficeMateChatService, question: str, selected_category: str, question_type_key: str, question_type_label: str, history):
        self.runtime = runtime
        self.question = question
        self.selected_category = selected_category
        self.question_type_key = question_type_key
        self.question_type_label = question_type_label
        self.history = history
        self.trace: List[ToolTrace] = []
        self.state: Dict[str, Any] = {
            "rewrite_result": None,
            "planned_tasks": None,
            "task_plans": None,
            "references": [],
            "task_reference_groups": [],
            "evidence_summary": {},
            "final_answer": "",
            "effective_category": selected_category,
        }

        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 OfficeMate，负责基于企业内部制度材料回答问题。"
                    "你只能依据给定证据回答，不能编造规则。"
                    "请严格输出以下 Markdown 标题："
                    "### 最终回答\n### 操作步骤/材料清单\n### 风险提示\n"
                    "如果材料不足，请明确写“未找到明确依据”。",
                ),
                (
                    "human",
                    "用户问题：{question}\n"
                    "问题类型：{question_type}\n"
                    "任务拆解（如果有）：\n{task_summary}\n\n"
                    "证据材料如下：\n{evidence_context}\n\n"
                    "请整合以上内容，直接给出一份最终回答。",
                ),
            ]
        )

    def _append_trace(self, step: str, start: float, summary: str, metadata: Dict[str, Any] | None = None) -> None:
        """把一次工具调用记录成页面可展示的轨迹。"""
        duration_ms = int((perf_counter() - start) * 1000)
        self.trace.append(ToolTrace(step=step, summary=summary, duration_ms=duration_ms, metadata=metadata or {}))

    def _ensure_rewrite_result(self) -> QueryRewriteResult:
        """如果 Agent 没主动调用 rewrite_tool，就补一个规则改写结果。"""
        existing = self.state.get("rewrite_result")
        if existing is not None:
            return existing
        result = rewrite_with_rules(self.question)
        self.state["rewrite_result"] = result
        return result

    def _ensure_planned_tasks(self) -> List[PlannedTask]:
        """如果 Agent 没主动调用 plan_tool，就补一个默认单任务。"""
        existing = self.state.get("planned_tasks")
        if existing is not None:
            return existing
        task = build_single_task(
            question=self.question,
            question_type_key=self.question_type_key,
            selected_category=self.selected_category,
        )
        self.state["planned_tasks"] = [task]
        return [task]

    def get_tools(self):
        """动态创建并返回一组可供 `create_agent` 调用的工具。"""
        @tool("rewrite_tool")
        def rewrite_tool(question: str) -> str:
            """当用户问题口语化、模糊、信息省略较多时，先把问题改写得更适合制度检索和规划。"""
            start = perf_counter()
            result = rewrite_with_model(question or self.question, self.runtime.query_rewriter)
            self.state["rewrite_result"] = result
            summary = json.dumps(
                {
                    "normalized_query": result.normalized_query,
                    "retrieval_queries": result.retrieval_queries,
                    "matched_terms": result.matched_terms,
                },
                ensure_ascii=False,
            )
            self._append_trace("rewrite_tool", start, f"完成问题改写，生成 {len(result.retrieval_queries)} 条检索 query。", {"normalized_query": result.normalized_query})
            logger.info("agent_react_tool rewrite_completed | normalized=%s | query_count=%s", result.normalized_query, len(result.retrieval_queries))
            return summary

        @tool("plan_tool")
        def plan_tool(question: str) -> str:
            """当问题包含多个子问题、多个制度域或多个并列条件时，先拆解成子任务。"""
            start = perf_counter()
            rewrite_result = self._ensure_rewrite_result()
            tasks = self.runtime.task_planner.plan(
                rewrite_result.normalized_query if rewrite_result else (question or self.question),
                self.question_type_key,
                self.selected_category,
            )
            self.state["planned_tasks"] = tasks
            task_summary = [
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "category": task.category,
                    "intent": task.intent,
                }
                for task in tasks
            ]
            self._append_trace("plan_tool", start, f"完成任务拆解，得到 {len(tasks)} 个子任务。", {"task_ids": [task.task_id for task in tasks]})
            logger.info("agent_react_tool plan_completed | task_ids=%s", [task.task_id for task in tasks])
            return json.dumps(task_summary, ensure_ascii=False)

        @tool("retrieve_and_rerank_tool")
        def retrieve_and_rerank_tool(question: str, target_category: str = "全部") -> str:
            """执行向量检索、BM25 混合检索和 rerank，整理最终证据。

            如果你已经判断问题明显属于某个制度域，可以传入 target_category 缩窄检索范围。
            可选值：全部、行政流程、财务制度、HR制度、IT支持。
            在给用户最终回答前必须先调用这个工具。
            """
            start = perf_counter()
            rewrite_result = self._ensure_rewrite_result()
            planned_tasks = self._ensure_planned_tasks()
            # Agent 可以主动缩窄检索范围；如果传入分类不合法，就回退到页面当前分类。
            effective_category = target_category if target_category in config.CATEGORY_FILTER_OPTIONS else self.selected_category
            self.state["effective_category"] = effective_category
            evidence = select_task_evidence(
                rewrite_result=rewrite_result,
                planned_tasks=planned_tasks,
                selected_category=effective_category,
                top_k=config.max_reference_documents,
                query_rewriter=self.runtime.query_rewriter,
                retrieval_coordinator=self.runtime.get_retrieval_coordinator(),
                reranker=self.runtime.reranker,
            )
            task_plans = evidence["task_plans"]
            references = evidence["references"]
            task_reference_groups = evidence["task_reference_groups"]

            self.state["task_plans"] = task_plans
            self.state["references"] = references
            self.state["task_reference_groups"] = task_reference_groups

            summary_payload = {
                "task_count": len(task_plans),
                "categories": sorted({item.task_category for item in task_plans}),
                "effective_category": effective_category,
                "reference_count": len(references),
                "task_summaries": [
                    {
                        "task_id": item.planned_task.task_id,
                        "description": item.planned_task.description,
                        "category": item.task_category,
                        "candidate_count": len(item.candidates),
                    }
                    for item in task_plans
                ],
            }
            self.state["evidence_summary"] = summary_payload
            self._append_trace(
                "retrieve_and_rerank_tool",
                start,
                f"完成检索与重排，分类范围：{effective_category}，得到 {len(references)} 份引用文档、{len(task_plans)} 组任务证据。",
                summary_payload,
            )
            logger.info(
                "agent_react_tool retrieve_and_rerank_completed | task_ids=%s | effective_category=%s | reference_count=%s",
                [item.planned_task.task_id for item in task_plans],
                effective_category,
                len(references),
            )
            return json.dumps(summary_payload, ensure_ascii=False)

        @tool("generate_final_answer_tool")
        def generate_final_answer_tool(question: str) -> str:
            """基于已经检索和重排出的证据，生成唯一的最终回答。必须在 retrieve_and_rerank_tool 之后调用。"""
            start = perf_counter()
            task_plans = self.state.get("task_plans") or []
            references = self.state.get("references") or []
            if not task_plans or not references:
                self._append_trace("generate_final_answer_tool", start, "缺少证据，返回无依据提示。", {"reference_count": len(references)})
                answer = config.NO_EVIDENCE_MESSAGE + "\n\n### 引用文档\n无"
                self.state["final_answer"] = answer
                return answer

            # 这里会分别组织“任务摘要”和“证据上下文”：
            # - 任务摘要帮助模型理解整体结构
            # - 证据上下文提供真正可回答的依据
            task_summary_lines = []
            evidence_blocks = []
            for index, item in enumerate(task_plans, start=1):
                task = item.planned_task
                task_summary_lines.append(
                    f"{index}. {task.description} | 分类：{item.task_category} | intent：{task.intent}"
                )
                evidence_context = build_context(item.candidates, limit=len(item.candidates))
                evidence_blocks.append(
                    f"[任务 {index}] {task.description} | 分类：{item.task_category}\n{evidence_context}"
                )

            chain = self.answer_prompt | self.runtime.get_chat_model() | StrOutputParser()
            answer_body = chain.invoke(
                {
                    "question": question or self.question,
                    "question_type": self.question_type_label,
                    "task_summary": "\n".join(task_summary_lines) if task_summary_lines else "无",
                    "evidence_context": "\n\n".join(evidence_blocks),
                }
            )
            answer_body = strip_think_blocks(answer_body)
            full_answer = build_final_answer(answer_body, references, include_references=True)
            self.state["final_answer"] = full_answer
            self._append_trace(
                "generate_final_answer_tool",
                start,
                f"已基于 {len(task_plans)} 组证据生成最终答案。",
                {"task_count": len(task_plans), "reference_count": len(references), "answer_length": len(full_answer)},
            )
            logger.info(
                "agent_react_tool generate_final_answer_completed | task_count=%s | reference_count=%s | answer_length=%s",
                len(task_plans),
                len(references),
                len(full_answer),
            )
            return full_answer

        return [
            rewrite_tool,
            plan_tool,
            retrieve_and_rerank_tool,
            generate_final_answer_tool,
        ]
