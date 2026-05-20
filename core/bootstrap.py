"""应用启动初始化。

这里统一做三件事：
1. 创建本地运行目录
2. 确保 MySQL 数据库存在
3. 自动建表
"""

from __future__ import annotations

import config_data as config
from core.db import Base, engine, ensure_database_exists


def bootstrap_runtime() -> None:
    """初始化项目运行时依赖。

    这个函数应该在 FastAPI 启动时优先执行，避免第一次请求才暴露环境问题。
    """
    config.ensure_runtime_dirs()
    ensure_database_exists()

    # 延迟导入模型，保证所有 ORM 实体都已经注册到 Base.metadata。
    import models.entities  # noqa: F401

    Base.metadata.create_all(bind=engine)
