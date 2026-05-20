"""Agent-ReAct-RAG 页面使用的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolTrace:
    """记录一次工具调用轨迹。"""

    step: str
    summary: str
    duration_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentReactResult:
    """一次 Agent-ReAct-RAG 问答的结果。"""

    answer: str
    question_type: str
    qa_log_id: str
    source_docs: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
