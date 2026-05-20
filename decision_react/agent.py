"""Decision Agent：先决定走哪些步骤，再执行工具。"""

from __future__ import annotations

import json
import re
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from decision_react.schemas import DecisionSchema
import config_data as config
from utils.log_tool import get_logger


logger = get_logger("decision_react_agent")


class QueryDecisionAgent:
    """一个受控的“决策型 Agent”。

    它不是全自由 Agent，而是只负责一件事：
    判断当前问题应该走轻路径还是重路径。

    这样可以避免原固定流水线里“简单问题也走完整链路”的问题。
    """

    def __init__(self, chat_model_factory: Callable):
        self.chat_model_factory = chat_model_factory
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是企业知识问答系统的决策 Agent。"
                    "你的职责不是直接回答问题，而是判断应该走哪些步骤。"
                    "请只输出 JSON，不要输出解释、Markdown 或代码块。"
                    '输出格式必须是：{{"complexity":"simple|complex","use_rewrite":true,"use_planner":false,"use_synthesize":false,"suggested_category":"全部","reason":"..."}}'
                    "判断原则："
                    "1. 简单单一制度问题通常是 simple。"
                    "2. 涉及多个并列子问题、多个制度域、多个条件对比的问题通常是 complex。"
                    "3. simple 问题通常不需要 planner，也通常不需要 synthesize。"
                    "4. 口语化、模糊、省略较多的问题更需要 rewrite。"
                    "5. suggested_category 只能从这些分类中选择："
                    f"{'、'.join(config.CATEGORY_FILTER_OPTIONS)}。"
                ),
                (
                    "human",
                    "用户问题：{question}\n"
                    "当前页面分类：{selected_category}",
                ),
            ]
        )

    def decide(self, question: str, selected_category: str = "全部") -> DecisionSchema:
        """先尝试用 LLM 决策，失败时走规则兜底。"""
        try:
            chain = self.prompt | self.chat_model_factory() | StrOutputParser()
            raw_output = chain.invoke({"question": question, "selected_category": selected_category})
            decision = self._parse_output(raw_output)
            logger.info(
                "decision_agent llm_completed | question=%s | selected_category=%s | decision=%s",
                question,
                selected_category,
                decision.model_dump(),
            )
            return decision
        except Exception as exc:
            logger.exception(
                "decision_agent llm_failed_fallback_to_rule | question=%s | selected_category=%s | error=%s",
                question,
                selected_category,
                exc,
            )
            return self._rule_decide(question, selected_category)

    def _parse_output(self, raw_output: str) -> DecisionSchema:
        cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.S).strip()
        object_start = cleaned.find("{")
        if object_start == -1:
            raise ValueError(f"Decision output does not contain JSON: {cleaned}")
        parsed = json.loads(cleaned[object_start:])
        decision = DecisionSchema.model_validate(parsed)
        if decision.suggested_category not in config.CATEGORY_FILTER_OPTIONS:
            decision.suggested_category = "全部"
        if decision.complexity not in {"simple", "complex"}:
            decision.complexity = "simple"
        return decision

    def _rule_decide(self, question: str, selected_category: str) -> DecisionSchema:
        """规则兜底：即使 LLM 决策失败，也能给出一个可执行结果。"""
        lowered = question.lower()
        multi_intent_markers = ("和", "以及", "分别", "同时", "还需要", "并且", "另外")
        policy_markers = ("报销", "补贴", "审批", "请假", "调休", "vpn", "权限", "采购")
        hit_count = sum(1 for marker in policy_markers if marker in lowered)
        is_complex = any(marker in question for marker in multi_intent_markers) or hit_count >= 2

        suggested_category = selected_category
        if suggested_category == "全部":
            if any(item in lowered for item in ("年假", "病假", "请假", "调休", "考勤")):
                suggested_category = "HR制度"
            elif any(item in lowered for item in ("报销", "补贴", "发票", "借款")):
                suggested_category = "财务制度"
            elif any(item in lowered for item in ("vpn", "账号", "权限", "密码")):
                suggested_category = "IT支持"

        decision = DecisionSchema(
            complexity="complex" if is_complex else "simple",
            use_rewrite=is_complex or len(question) >= 18,
            use_planner=is_complex,
            use_synthesize=is_complex,
            suggested_category=suggested_category,
            reason="规则兜底：根据并列结构和多意图关键词判断问题复杂度。",
        )
        logger.info(
            "decision_agent rule_completed | question=%s | selected_category=%s | decision=%s",
            question,
            selected_category,
            decision.model_dump(),
        )
        return decision
