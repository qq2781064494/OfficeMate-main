"""主 RAG 聊天服务兼容入口。

新实现已经迁移到 `services.rag.chat_service`。
这个文件只保留旧 import 路径兼容，避免页面、评测和 ReAct 侧一次性断裂。
"""

from services.rag.chat_service import OfficeMateChatService, _ThinkBlockStreamFilter, _strip_think_blocks

__all__ = ["OfficeMateChatService", "_ThinkBlockStreamFilter", "_strip_think_blocks"]
