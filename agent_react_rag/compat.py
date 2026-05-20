"""create_agent 兼容层。

当前环境里的 langchain / langchain_core / langgraph 版本不完全匹配，
直接 `from langchain.agents import create_agent` 会因为缺少
`_DirectlyInjectedToolArg` 而报错。

这里做一个极小的运行时补丁，只修复这个导入阻塞点，让新页面可以
独立验证“真正的 create_agent 工具调用风格”。
"""

from __future__ import annotations


def get_create_agent():
    import langchain_core.tools.base as tool_base

    if not hasattr(tool_base, "_DirectlyInjectedToolArg") and hasattr(tool_base, "InjectedToolArg"):
        tool_base._DirectlyInjectedToolArg = tool_base.InjectedToolArg

    from langchain.agents import create_agent

    return create_agent
