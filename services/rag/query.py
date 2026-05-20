"""query 阶段实现。

这一层的目标不是直接回答问题，而是先把问题变得更适合检索：
- 判断问题属于什么类型
- 把口语化问题改得更规范
- 用同义词扩展检索词
- 在不改动原意的前提下，让后续召回更多相关文档
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config_data as config
from services.rag.planning import PlannedTask


@dataclass
class QueryRewriteResult:
    """保存一次查询改写的结果。"""

    original_query: str
    normalized_query: str
    retrieval_queries: List[str]
    matched_terms: Dict[str, List[str]] = field(default_factory=dict)


class QueryRewriter:
    """基于轻量词典和可选 LLM 的企业术语改写器。"""

    def __init__(self, synonyms: Dict[str, List[str]] | None = None, chat_model_factory: Callable | None = None):
        self.synonyms = synonyms or config.QUERY_SYNONYMS
        self.chat_model_factory = chat_model_factory
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是企业知识问答系统的查询改写器，只负责输出最终改写后的问题。"
                    "你的输出必须严格满足以下规则："
                    "1. 只能输出一行纯文本问题，不能输出多行。"
                    "2. 只能输出改写后的问题本身，禁止输出任何解释、备注、前后缀、标题、标签、编号、引号、Markdown、XML、JSON。"
                    "3. 严禁输出“改写后：”“用户问题改写成：”“重写结果：”“优化后的问题：”这类前缀。"
                    "4. 保留原问题真实意图，不得改变事实，不得补造制度内容，不得擅自增加限定条件。"
                    "5. 如果原问题已经清晰，就尽量保持原意，只做最小必要规范化。"
                    "6. 可以补全口语省略、统一称谓、去掉明显重复片段，但不要扩写成说明文。"
                    "7. 如果你不确定怎么改写，就原样输出用户问题。"
                    "8. 原始问题与改写后的问题要使用同样的语言"
                    "正确示例：年假最晚需要提前几天申请？"
                    "错误示例：用户问题改写成：年假最晚需要提前几天申请？"
                    "错误示例：改写后问题如下：年假最晚需要提前几天申请？"
                    "错误示例：以 JSON 形式输出改写结果。"
                ),
                ("human", "用户问题：{query}"),
            ]
        )

    def rewrite(self, query: str) -> QueryRewriteResult:
        """执行一次查询改写。

        优先走 LLM 改写，因为它更擅长补全口语、省略和模糊表达；
        如果模型不可用，就退回规则改写，保证主链路不断。
        """
        from utils.log_tool import get_logger

        rewrite_logger = get_logger("query_rewriter")
        if self.chat_model_factory is not None:
            try:
                result = self._llm_rewrite(query)
                rewrite_logger.info(
                    "query_rewrite llm_completed | original=%s | normalized=%s | matched_terms=%s | retrieval_queries=%s",
                    query,
                    result.normalized_query,
                    result.matched_terms,
                    result.retrieval_queries,
                )
                return result
            except Exception as exc:
                rewrite_logger.exception("query_rewrite llm_failed_fallback_to_rule | query=%s | error=%s", query, exc)

        result = self._rule_rewrite(query)
        rewrite_logger.info(
            "query_rewrite completed | original=%s | normalized=%s | matched_terms=%s | retrieval_queries=%s",
            query,
            result.normalized_query,
            result.matched_terms,
            result.retrieval_queries,
        )
        return result

    def _rule_rewrite(self, query: str) -> QueryRewriteResult:
        """基于同义词词典做轻量扩展。"""
        retrieval_queries: List[str] = [query.strip()]
        matched_terms: Dict[str, List[str]] = {}

        for canonical, variants in self.synonyms.items():
            matched = [term for term in [canonical, *variants] if term.lower() in query.lower()]
            if not matched:
                continue

            matched_terms[canonical] = matched
            retrieval_queries.append(canonical)
            retrieval_queries.extend(variants)

        # `dict.fromkeys(...)` 是很常见的“保序去重”技巧。
        unique_queries = list(dict.fromkeys(item.strip() for item in retrieval_queries if item.strip()))
        normalized_query = unique_queries[0]
        if matched_terms:
            # `normalized_query` 给 planner 和日志阅读用，会带上术语扩展说明。
            expanded_terms = "、".join(matched_terms.keys())
            normalized_query = f"{query}\n术语扩展：{expanded_terms}"

        return QueryRewriteResult(
            original_query=query,
            normalized_query=normalized_query,
            retrieval_queries=unique_queries,
            matched_terms=matched_terms,
        )

    def _llm_rewrite(self, query: str) -> QueryRewriteResult:
        """先让模型改写，再叠加规则词典扩展。"""
        chain = self.prompt | self.chat_model_factory() | StrOutputParser()
        raw_output = chain.invoke({"query": query})
        rewritten_query = self._clean_rewritten_question(raw_output, query)
        rule_result = self._rule_rewrite(rewritten_query)
        return QueryRewriteResult(
            original_query=query,
            normalized_query=rewritten_query,
            retrieval_queries=rule_result.retrieval_queries,
            matched_terms=rule_result.matched_terms,
        )

    def _clean_rewritten_question(self, raw_output: str, original_query: str) -> str:
        """清洗模型输出，尽量只保留真正的问题文本。"""
        cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.S).strip()
        if not cleaned:
            raise ValueError("LLM rewrite returned empty content")

        first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
        rewritten = first_line or cleaned
        rewritten = re.sub(r"^(重写后问题|改写后问题|改写结果|问题)\s*[:：]\s*", "", rewritten).strip()
        rewritten = rewritten.strip("`\"' ")
        return rewritten or original_query.strip()

    def narrow_to_hints(self, rewrite_result: QueryRewriteResult, hints: List[str]) -> QueryRewriteResult:
        """根据任务 hints 缩窄检索词集合。"""
        from utils.log_tool import get_logger

        rewrite_logger = get_logger("query_rewriter")
        clean_hints = [hint.strip().lower() for hint in hints if hint.strip()]
        if not clean_hints:
            return rewrite_result

        scoped_queries = [rewrite_result.original_query.strip()]
        for item in rewrite_result.retrieval_queries:
            lowered = item.lower()
            if any(hint in lowered or lowered in hint for hint in clean_hints):
                scoped_queries.append(item)

        unique_queries = list(dict.fromkeys(item.strip() for item in scoped_queries if item.strip()))
        if len(unique_queries) <= 1:
            rewrite_logger.info(
                "query_rewrite narrow_to_hints no_match_fallback | hints=%s | fallback_queries=%s",
                hints,
                rewrite_result.retrieval_queries,
            )
            return rewrite_result

        scoped_result = QueryRewriteResult(
            original_query=rewrite_result.original_query,
            normalized_query=rewrite_result.normalized_query,
            retrieval_queries=unique_queries,
            matched_terms=rewrite_result.matched_terms,
        )
        rewrite_logger.info(
            "query_rewrite narrow_to_hints completed | hints=%s | scoped_queries=%s",
            hints,
            unique_queries,
        )
        return scoped_result


def infer_question_type(question: str) -> str:
    """用轻规则判断问题类型。"""
    lowered = question.lower()
    if any(keyword in lowered for keyword in ("材料", "附件", "提交什么", "要带什么", "需要什么")):
        return "material_list"
    if any(keyword in lowered for keyword in ("流程", "步骤", "怎么走", "怎么发起", "如何办理")):
        return "process_guide"
    if any(keyword in lowered for keyword in ("总结", "概括", "通知重点", "提炼")):
        return "notice_summary"
    return "policy_qa"


def resolve_question_type_label(question_type_key: str) -> str:
    """把内部类型 key 转成更友好的中文标签。"""
    return config.QUESTION_TYPE_LABELS[question_type_key]


def rewrite_with_model(query: str, rewriter: QueryRewriter) -> QueryRewriteResult:
    return rewriter.rewrite(query)


def rewrite_with_rules(query: str) -> QueryRewriteResult:
    return QueryRewriter(chat_model_factory=None).rewrite(query)


def infer_category_from_question(question: str) -> str:
    """在用户没有手动选择分类时，给出一个粗粒度建议。"""
    lowered = question.lower()
    if any(item in lowered for item in ("年假", "病假", "请假", "调休", "考勤")):
        return "HR制度"
    if any(item in lowered for item in ("报销", "补贴", "发票", "借款")):
        return "财务制度"
    if any(item in lowered for item in ("vpn", "账号", "权限", "密码")):
        return "IT支持"
    if any(item in lowered for item in ("审批", "采购", "流程")):
        return "行政流程"
    return "全部"


def build_single_task(
    question: str,
    question_type_key: str,
    selected_category: str,
    suggested_category: str = "全部",
) -> PlannedTask:
    """把简单问题包装成一个默认单任务。"""
    category = selected_category if selected_category != "全部" else suggested_category
    if category == "全部":
        category = infer_category_from_question(question)
    return PlannedTask(
        task_id="single_answer",
        description=f"直接回答用户问题：{question}",
        category=category,
        intent=question_type_key,
        hints=[],
    )


__all__ = [
    "QueryRewriteResult",
    "QueryRewriter",
    "build_single_task",
    "infer_question_type",
    "infer_category_from_question",
    "resolve_question_type_label",
    "rewrite_with_model",
    "rewrite_with_rules",
]
