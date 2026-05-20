"""Decision-Guided ReAct 风格问答模块。

这一套代码不会替换原有固定流水线，而是单独提供一条新的问答链路：
- 先由决策 Agent 判断问题是否复杂
- 再按需决定要不要 rewrite / plan / synthesize
- 最后调用现有的 rewrite、planner、retrieval、rerank、answer 等模块

这样做的目标是：
- 简单问题走轻路径，避免过度拆解
- 复杂问题走重路径，保留多任务能力
"""

