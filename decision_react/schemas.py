"""Decision-ReAct 模块用到的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class DecisionSchema(BaseModel):
    """决策 Agent 输出的控制信号。

    这里不是让 Agent 自由发挥，而是只让它决定几个关键开关：
    - 是否需要 rewrite
    - 是否需要 planner
    - 是否需要 synthesize
    - 当前问题更像 simple 还是 complex
    - 推荐优先落在哪个分类
    """

    complexity: str = Field(description="simple 或 complex")
    use_rewrite: bool = Field(description="是否需要先做 LLM rewrite")
    use_planner: bool = Field(description="是否需要做多任务拆解")
    use_synthesize: bool = Field(description="是否需要对子答案做汇总")
    suggested_category: str = Field(default="全部", description="推荐优先使用的知识分类")
    reason: str = Field(default="", description="简短解释为什么这样决策")


@dataclass
class ToolTrace:
    """保存一次工具执行的轨迹。

    新页面会把这些轨迹展示出来，方便你观察：
    - Agent 到底决定了什么
    - 每个工具什么时候被调用
    - 每一步产出了什么
    """

    step: str
    summary: str
    duration_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionReactResult:
    """Decision-ReAct 问答链路的最终结果。"""

    answer: str
    question_type: str
    qa_log_id: str
    source_docs: List[Dict[str, Any]]
    decision: Dict[str, Any]
    trace: List[Dict[str, Any]]

