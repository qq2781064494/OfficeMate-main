"""FastAPI 启动入口。

运行方式：
`uvicorn main:app --reload`
"""

from api.app import app

__all__ = ["app"]
