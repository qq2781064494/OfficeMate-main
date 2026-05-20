"""项目级日志工具。

这个模块的目标是：
1. 给整个项目提供统一的 logging 配置
2. 让不同模块都能很方便地拿到自己的 logger
3. 同时输出到控制台和文件，便于开发调试与后续排错

如果你刚学 Python，可以把它理解为“日志工厂”：
- 业务模块不需要重复写 logging 配置
- 只需要 `get_logger("模块名")`
- 就能得到已经配置好的日志器
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from utils.path_tool import get_abs_path


LOG_ROOT = get_abs_path("logs")
# 这里会把日志统一存到项目根目录下的 logs 文件夹里。
os.makedirs(LOG_ROOT, exist_ok=True)

# 定义日志长什么样。
# 常见信息包括：
# - 时间
# - logger 名称
# - 日志级别
# - 文件名和行号
# - 日志正文
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def get_logger(
    name: str = "agent",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: str | None = None,
) -> logging.Logger:
    """创建并返回一个已经配置好的 logger。

    参数说明：
    - name: logger 名称，通常写模块名，例如 "chat_service"
    - console_level: 控制台输出的最低级别
    - file_level: 写入日志文件的最低级别
    - log_file: 可选，自定义日志文件路径

    返回：
    - 一个 logging.Logger 对象
    """
    logger = logging.getLogger(name)
    # 这里把 logger 的总级别设成 DEBUG，表示“先全部接收”，
    # 之后再由每个 handler 自己决定输出哪些级别。
    logger.setLevel(logging.DEBUG)
    # propagate=False 表示不要把日志继续向上级 logger 传递，
    # 否则有时会出现重复打印。
    logger.propagate = False

    # 如果这个 logger 之前已经配过 handler，就直接复用，
    # 否则会重复添加 handler，造成同一条日志输出多次。
    if logger.handlers:
        return logger

    # 控制台 handler：让我们在终端里实时看到日志。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    if not log_file:
        # 默认按“logger 名 + 日期”生成日志文件名，方便按模块和日期排查。
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    # 文件 handler：把日志写到磁盘，便于后续回看完整链路。
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)
    return logger


# 给不想自己取名字的模块一个默认 logger。
logger = get_logger()
