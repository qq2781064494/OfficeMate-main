"""planning 阶段实现。

planning 可以理解成“回答前先列小提纲”：
- 简单问题：通常只保留 1 个任务
- 复杂问题：拆成多个可独立检索的子任务

这样后面的检索和回答会更稳，因为每个子任务只关注一小块目标。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, dataclass as dataclass_decorator
import json
import re
from typing import Callable, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import config_data as config
from utils.log_tool import get_logger


planner_logger = get_logger("task_planner")


@dataclass_decorator
class PlannedTask:
    """一个可独立执行的子任务。"""

    task_id: str
    description: str
    category: str
    intent: str
    hints: List[str] = field(default_factory=list)


class PlannerTaskSchema(BaseModel):
    """约束单个子任务的结构化输出格式。"""

    task_id: str = Field(description="子任务唯一标识，使用简洁英文标识")
    description: str = Field(description="子任务描述，应明确说明本任务要回答什么")
    category: str = Field(description="子任务所属分类")
    intent: str = Field(description="任务类型，只能是 policy_qa/process_guide/material_list/notice_summary")
    hints: list[str] = Field(default_factory=list, description="用于检索和重排的关键词提示")


class PlannerResultSchema(BaseModel):
    """约束整个任务规划结果的结构化输出格式。"""

    tasks: list[PlannerTaskSchema] = Field(..., min_length=1, description="拆解后的子任务列表，至少包含一个子任务")


class BaseTaskPlanner(ABC):
    """所有 planner 的统一接口。"""

    @abstractmethod
    def plan(self, query: str, question_type: str, selected_category: str = "全部") -> List[PlannedTask]:
        raise NotImplementedError


class RuleTaskPlanner(BaseTaskPlanner):
    """纯规则版本的任务规划器。"""

    def plan(self, query: str, question_type: str, selected_category: str = "全部") -> List[PlannedTask]:
        tasks: List[PlannedTask] = []
        lowered = query.lower()

        # 下面这些条件不是互斥关系。
        # 一个问题可以同时命中多个制度域，因此会被拆成多个任务。
        if any(keyword in lowered for keyword in ("审批", "申请", "先走流程", "发起")):
            tasks.append(
                PlannedTask(
                    task_id="approval",
                    description="确认是否需要先审批、由谁发起以及审批节点",
                    category="行政流程",
                    intent="process_guide",
                    hints=["审批", "流程", "发起"],
                )
            )

        if any(keyword in lowered for keyword in ("报销", "报账", "费用")):
            tasks.append(
                PlannedTask(
                    task_id="reimbursement",
                    description="确认报销流程、材料要求和时效限制",
                    category="财务制度",
                    intent="material_list" if "材料" in query or "附件" in query else "policy_qa",
                    hints=["报销", "发票", "费用"],
                )
            )

        if any(keyword in lowered for keyword in ("补贴", "津贴", "补助")):
            tasks.append(
                PlannedTask(
                    task_id="allowance",
                    description="确认补贴或津贴的适用条件、计算方式和标准",
                    category="财务制度",
                    intent="policy_qa",
                    hints=["补贴", "津贴", "标准"],
                )
            )

        if any(keyword in lowered for keyword in ("vpn", "账号", "权限", "密码", "电脑", "软件")):
            tasks.append(
                PlannedTask(
                    task_id="it_support",
                    description="确认 IT 支持处理步骤、权限要求和常见异常处理",
                    category="IT支持",
                    intent="process_guide",
                    hints=["IT", "账号", "权限"],
                )
            )

        if any(keyword in lowered for keyword in ("年假", "病假", "补卡", "调休", "考勤", "请假")):
            tasks.append(
                PlannedTask(
                    task_id="hr_policy",
                    description="确认 HR 制度要求、申请条件和审批规则",
                    category="HR制度",
                    intent=question_type,
                    hints=["HR", "请假", "考勤"],
                )
            )

        if not tasks:
            # 什么规则都没命中时，也要给后续阶段一个统一的兜底任务。
            result = [
                PlannedTask(
                    task_id="general",
                    description=f"直接回答用户问题：{query}",
                    category=selected_category if selected_category != "全部" else "全部",
                    intent=question_type,
                    hints=[],
                )
            ]
            planner_logger.info("rule_planner fallback_single_task | query=%s | tasks=%s", query, result)
            return result

        deduped = self._deduplicate(tasks, selected_category)
        planner_logger.info("rule_planner completed | query=%s | selected_category=%s | tasks=%s", query, selected_category, deduped)
        return deduped

    def _deduplicate(self, tasks: List[PlannedTask], selected_category: str) -> List[PlannedTask]:
        """去重并套用用户在页面上手动选择的分类。"""
        deduplicated = []
        seen = set()
        for task in tasks:
            if task.task_id in seen:
                continue
            seen.add(task.task_id)
            if selected_category != "全部":
                task.category = selected_category
            deduplicated.append(task)
            if len(deduplicated) >= config.max_subtasks:
                break
        return deduplicated


class LLMTaskPlanner(BaseTaskPlanner):
    """基于大模型的任务规划器。"""

    def __init__(self, chat_model_factory: Callable):
        self.chat_model_factory = chat_model_factory
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是企业知识问答系统的任务规划器，不负责直接回答问题。"
                    "你的职责是把用户问题拆成 1 到 4 个可独立检索和回答的子任务。"
                    "每个子任务都必须包含 task_id、description、category、intent、hints。"
                    "category 只能从以下列表中选择："
                    f"{'、'.join(config.CATEGORY_FILTER_OPTIONS)}。"
                    "intent 只能从以下列表中选择："
                    f"{'、'.join(config.QUESTION_TYPE_LABELS.keys())}。"
                    "如果问题是单一意图，只输出一个任务。"
                    "必须返回结构化结果。"
                    "禁止输出解释文字、Markdown 代码块、XML 标签、额外前后缀。",
                ),
                (
                    "human",
                    "用户问题：{query}\n"
                    "页面当前选择的分类：{selected_category}\n"
                    "推断的问题类型：{question_type}",
                ),
            ]
        )

    def plan(self, query: str, question_type: str, selected_category: str = "全部") -> List[PlannedTask]:
        """优先尝试结构化输出，最后才退回文本解析。"""
        planner_logger.info(
            "llm_planner start | query=%s | question_type=%s | selected_category=%s",
            query,
            question_type,
            selected_category,
        )
        chat_model = self.chat_model_factory()
        payload = {
            "query": query,
            "selected_category": selected_category,
            "question_type": question_type,
        }

        result = self._plan_with_function_calling(chat_model, payload)
        if result is None:
            result = self._plan_with_schema_output(chat_model, payload)
        if result is None:
            result = self._plan_with_text_fallback(chat_model, payload)

        tasks = [
            PlannedTask(
                task_id=str(task.task_id),
                description=task.description,
                category=task.category,
                intent=task.intent,
                hints=task.hints,
            )
            for task in result.tasks
        ]
        planner_logger.info("llm_planner completed | query=%s | tasks=%s", query, tasks)
        return tasks

    def _plan_with_schema_output(self, chat_model, payload: dict) -> PlannerResultSchema | None:
        """优先使用 `json_schema` 方式拿结构化结果。"""
        if not hasattr(chat_model, "with_structured_output"):
            planner_logger.info("llm_planner schema_output_unsupported")
            return None
        try:
            structured_model = chat_model.with_structured_output(
                PlannerResultSchema,
                method="json_schema",
                include_raw=True,
                strict=True,
            )
            chain = self.prompt | structured_model
            result = chain.invoke(payload)
            normalized = self._coerce_structured_response(result)
            planner_logger.info("llm_planner schema_output_completed")
            return normalized
        except Exception as exc:
            planner_logger.warning("llm_planner schema_output_failed | error=%s", exc)
            return None

    def _plan_with_function_calling(self, chat_model, payload: dict) -> PlannerResultSchema | None:
        """当 `json_schema` 不可用时，回退到 function calling。"""
        if not hasattr(chat_model, "with_structured_output"):
            planner_logger.info("llm_planner function_calling_unsupported")
            return None
        try:
            structured_model = chat_model.with_structured_output(
                PlannerResultSchema,
                method="function_calling",
                include_raw=True,
                strict=True,
            )
            chain = self.prompt | structured_model
            result = chain.invoke(payload)
            normalized = self._coerce_structured_response(result)
            planner_logger.info("llm_planner function_calling_completed")
            return normalized
        except Exception as exc:
            planner_logger.warning("llm_planner function_calling_failed | error=%s", exc)
            return None

    def _plan_with_text_fallback(self, chat_model, payload: dict) -> PlannerResultSchema:
        """最后兜底：把模型输出当普通文本，再手动抽 JSON。"""
        chain = self.prompt | chat_model | StrOutputParser()
        raw_output = chain.invoke(payload)
        planner_logger.debug("llm_planner raw_output=%s", raw_output)
        return self._parse_planner_output(raw_output)

    def _coerce_planner_result(self, result) -> PlannerResultSchema:
        """把不同形态的结果统一转成 `PlannerResultSchema`。"""
        if isinstance(result, PlannerResultSchema):
            if not result.tasks:
                raise ValueError("Planner structured result contains no tasks")
            return result
        if isinstance(result, BaseModel):
            validated = PlannerResultSchema.model_validate(result.model_dump())
            if not validated.tasks:
                raise ValueError("Planner structured result contains no tasks")
            return validated
        if isinstance(result, dict):
            validated = PlannerResultSchema.model_validate(self._normalize_parsed_payload(result))
            if not validated.tasks:
                raise ValueError("Planner structured result contains no tasks")
            return validated
        if isinstance(result, list):
            validated = PlannerResultSchema.model_validate({"tasks": self._normalize_parsed_payload({"tasks": result})["tasks"]})
            if not validated.tasks:
                raise ValueError("Planner structured result contains no tasks")
            return validated
        raise ValueError(f"Unsupported structured planner result: {type(result)!r}")

    def _coerce_structured_response(self, result) -> PlannerResultSchema:
        """兼容不同 provider 对 structured output 的返回差异。"""
        if isinstance(result, dict) and any(key in result for key in ("parsed", "raw", "parsing_error")):
            parsed = result.get("parsed")
            if parsed is not None:
                validated = self._coerce_planner_result(parsed)
                if not validated.tasks:
                    raise ValueError("Structured planner parsed empty tasks")
                return validated

            raw = result.get("raw")
            raw_text = self._extract_raw_message_text(raw)
            if raw_text:
                planner_logger.debug("llm_planner structured_raw_output=%s", raw_text)
                return self._parse_planner_output(raw_text)

            parsing_error = result.get("parsing_error")
            if parsing_error is not None:
                raise ValueError(f"Structured planner parsing failed: {parsing_error}")

        return self._coerce_planner_result(result)

    def _extract_raw_message_text(self, raw) -> str:
        """尽量从原始 message 结构里抽出纯文本。"""
        if raw is None:
            return ""
        content = getattr(raw, "content", raw)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, str):
                    fragments.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
            return "\n".join(fragment for fragment in fragments if fragment.strip())
        return str(content)

    def _parse_planner_output(self, raw_output: str) -> PlannerResultSchema:
        """从文本里提取 JSON，并校验成标准 schema。"""
        cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.S).strip()
        if not cleaned:
            raise ValueError("LLM planner returned empty content")
        json_text = self._extract_json_text(cleaned)
        parsed = json.loads(json_text)
        if isinstance(parsed, list):
            parsed = {"tasks": parsed}
        if not isinstance(parsed, dict):
            raise ValueError(f"Unsupported planner output type: {type(parsed)!r}")
        parsed = self._normalize_parsed_payload(parsed)
        return PlannerResultSchema.model_validate(parsed)

    def _normalize_parsed_payload(self, payload: dict) -> dict:
        """先做一轮脏数据清洗，再进入 Pydantic 校验。"""
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            return payload
        normalized_tasks = []
        for index, task in enumerate(tasks, start=1):
            if not isinstance(task, dict):
                continue
            normalized_task = dict(task)
            raw_task_id = normalized_task.get("task_id", f"task_{index}")
            normalized_task["task_id"] = str(raw_task_id).strip() or f"task_{index}"
            raw_hints = normalized_task.get("hints", [])
            if isinstance(raw_hints, str):
                normalized_task["hints"] = [raw_hints]
            elif isinstance(raw_hints, list):
                normalized_task["hints"] = [str(item).strip() for item in raw_hints if str(item).strip()]
            else:
                normalized_task["hints"] = []
            normalized_tasks.append(normalized_task)
        payload["tasks"] = normalized_tasks
        return payload

    def _extract_json_text(self, text: str) -> str:
        """找到模型回复里第一个 JSON 对象或数组。"""
        object_start = text.find("{")
        array_start = text.find("[")
        starts = [index for index in (object_start, array_start) if index != -1]
        if not starts:
            raise ValueError(f"Planner output does not contain JSON: {text}")
        return text[min(starts):]


class HybridTaskPlanner(BaseTaskPlanner):
    """混合式 planner：LLM 优先，规则兜底。"""

    def __init__(self, llm_planner: LLMTaskPlanner, rule_planner: RuleTaskPlanner | None = None):
        self.llm_planner = llm_planner
        self.rule_planner = rule_planner or RuleTaskPlanner()
        self.allowed_categories = set(config.CATEGORY_FILTER_OPTIONS)
        self.allowed_intents = set(config.QUESTION_TYPE_LABELS.keys())

    def plan(self, query: str, question_type: str, selected_category: str = "全部") -> List[PlannedTask]:
        fallback_tasks = self.rule_planner.plan(query, question_type, selected_category)
        try:
            llm_tasks = self.llm_planner.plan(query, question_type, selected_category)
            validated_tasks = self._validate_tasks(llm_tasks, selected_category, question_type)
            if validated_tasks:
                planner_logger.info(
                    "hybrid_planner using_llm_plan | query=%s | selected_category=%s | tasks=%s",
                    query,
                    selected_category,
                    validated_tasks,
                )
                return validated_tasks
        except Exception as exc:
            planner_logger.exception("hybrid_planner llm_failed_fallback_to_rule | query=%s | error=%s", query, exc)

        planner_logger.info("hybrid_planner using_rule_fallback | query=%s | tasks=%s", query, fallback_tasks)
        return fallback_tasks

    def _validate_tasks(self, tasks: List[PlannedTask], selected_category: str, question_type: str) -> List[PlannedTask]:
        """校验 LLM 规划结果，避免脏数据流入后续阶段。"""
        validated: List[PlannedTask] = []
        seen_keys = set()
        for task in tasks:
            if not task.description.strip():
                continue

            category = task.category if task.category in self.allowed_categories else "全部"
            intent = task.intent if task.intent in self.allowed_intents else question_type
            if selected_category != "全部":
                category = selected_category

            hints = [item.strip() for item in task.hints if item.strip()][:4]
            normalized_task = PlannedTask(
                task_id=task.task_id.strip() or f"task_{len(validated) + 1}",
                description=task.description.strip(),
                category=category,
                intent=intent,
                hints=hints,
            )

            dedupe_key = (
                normalized_task.category,
                normalized_task.intent,
                normalized_task.description[:24],
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            validated.append(normalized_task)
            if len(validated) >= config.max_subtasks:
                break

        if not validated:
            planner_logger.warning("hybrid_planner validation_empty | selected_category=%s | question_type=%s", selected_category, question_type)
            return []
        planner_logger.debug("hybrid_planner validated_tasks=%s", validated)
        return validated


class PlannerFactory:
    """按策略名创建 planner，减少业务层和具体实现的耦合。"""

    @staticmethod
    def create(strategy: str, chat_model_factory: Callable | None = None) -> BaseTaskPlanner:
        if strategy == "rule":
            return RuleTaskPlanner()
        if strategy == "llm":
            if chat_model_factory is None:
                raise ValueError("LLMTaskPlanner requires chat_model_factory")
            return LLMTaskPlanner(chat_model_factory)
        if chat_model_factory is None:
            raise ValueError("HybridTaskPlanner requires chat_model_factory")
        return HybridTaskPlanner(
            llm_planner=LLMTaskPlanner(chat_model_factory),
            rule_planner=RuleTaskPlanner(),
        )


__all__ = [
    "BaseTaskPlanner",
    "HybridTaskPlanner",
    "LLMTaskPlanner",
    "PlannedTask",
    "PlannerFactory",
    "PlannerResultSchema",
    "PlannerTaskSchema",
    "RuleTaskPlanner",
]
