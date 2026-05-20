"""LangChain 运行时兼容补丁。"""

from __future__ import annotations

from utils.log_tool import get_logger


logger = get_logger("langchain_compat")


def ensure_langchain_runtime_compatibility() -> None:
    """兼容部分环境中 langchain / langchain_core 版本不一致的问题。"""
    try:
        import langchain  # type: ignore
    except ImportError:
        logger.warning("langchain runtime compatibility skipped | package=missing")
        return

    if not hasattr(langchain, "verbose"):
        setattr(langchain, "verbose", False)
        logger.warning("langchain compatibility patch applied | attr=verbose")
    if not hasattr(langchain, "debug"):
        setattr(langchain, "debug", False)
        logger.warning("langchain compatibility patch applied | attr=debug")
    if not hasattr(langchain, "llm_cache"):
        setattr(langchain, "llm_cache", None)
        logger.warning("langchain compatibility patch applied | attr=llm_cache")
