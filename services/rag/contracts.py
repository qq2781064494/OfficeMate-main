"""主 RAG 链路的显式协议模型。

这些 dataclass 可以理解成“阶段之间传递的数据包”：
- `ChatRequest`：页面层把一次提问请求打包后交给服务层
- `PipelineState`：前半段流水线处理后的中间状态
- `ChatResponse`：最终返回给页面层或评测模块的结果

把协议对象显式写出来有两个好处：
1. 新手更容易看懂每个阶段到底输入什么、输出什么。
2. 后续重构时，不容易因为随手传 dict 而把字段传乱。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class ChatRequest:
    """一次问答请求的标准输入。"""

    question: str
    session_id: str
    category: str = "全部"
    use_history: bool = True
    persist_log: bool = True
    include_references: bool = True
    enable_query_rewrite: bool = True
    enable_rerank: bool = True
    reference_limit: int | None = None
    status_callback: Callable[[str], None] | None = None
    event_callback: Callable[[str, dict[str, Any]], None] | None = None


@dataclass
class ChatResponse:
    """一次问答完成后的标准输出。"""

    answer: str
    question_type: str
    qa_log_id: int | None
    source_docs: list[dict]
    retrieved_contexts: list[str]
    normalized_query: str
    retrieval_queries: list[str]
    matched_terms: Any
    pre_rerank_titles: list[str]
    retrieved_titles: list[str]
    planned_tasks: list[dict]

    def to_legacy_dict(self) -> dict[str, Any]:
        """兼容旧页面层仍然使用的 dict 返回格式。"""
        return {
            "answer": self.answer,
            "question_type": self.question_type,
            "qa_log_id": self.qa_log_id,
            "source_docs": self.source_docs,
            "retrieved_contexts": self.retrieved_contexts,
            "normalized_query": self.normalized_query,
            "retrieval_queries": self.retrieval_queries,
            "matched_terms": self.matched_terms,
            "pre_rerank_titles": self.pre_rerank_titles,
            "retrieved_titles": self.retrieved_titles,
            "planned_tasks": self.planned_tasks,
        }


@dataclass
class ChatResultHolder:
    """流式问答结束后承载元数据，再由旧页面层读取。"""

    response: ChatResponse | None = None
    legacy_result: dict[str, Any] = field(default_factory=dict)

    def set_response(self, response: ChatResponse) -> None:
        self.response = response
        self.legacy_result.clear()
        self.legacy_result.update(response.to_legacy_dict())

    def to_legacy_dict(self) -> dict[str, Any]:
        return self.legacy_result


@dataclass
class StreamingChatSession:
    """流式问答返回对象。"""

    stream: Iterable[str]
    result_holder: ChatResultHolder = field(default_factory=ChatResultHolder)


@dataclass
class TaskInput:
    """进入证据选择阶段前的任务输入包。"""

    planned_task: Any
    task_category: str
    task_rewrite_result: Any
    candidates: list[Any]


@dataclass
class TaskPlan:
    """证据分配完成后的子任务执行计划。"""

    planned_task: Any
    task_category: str
    candidates: list[Any]


@dataclass
class PipelineState:
    """问答主链路前半段的中间状态。"""

    question_type_key: str
    question_type: str
    rewrite_result: Any
    planned_tasks: list[Any]
    history: list[Any]
