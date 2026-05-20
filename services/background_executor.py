"""后台任务执行器。

首版不引入 Celery / Redis，直接使用线程池跑长任务。
任务状态统一回写到 MySQL 的 task_runs 表。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Callable

from services.task_run_service import TaskRunService
from utils.log_tool import get_logger


logger = get_logger("background_executor")


class BackgroundExecutor:
    def __init__(self, max_workers: int = 4):
        self.task_service = TaskRunService()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="officemate-bg")

    def submit(self, task_type: str, payload: dict | list | None, runner: Callable[[], dict | list | None]) -> dict:
        task = self.task_service.create_task(task_type=task_type, payload=payload)
        task_id = task["id"]

        def wrapped():
            try:
                self.task_service.mark_running(task_id, progress_stage="running", progress_message="任务开始执行。")
                result = runner()
                self.task_service.mark_completed(task_id, result_json=result)
            except Exception as exc:
                logger.exception("background_task failed | task_id=%s | task_type=%s | error=%s", task_id, task_type, exc)
                self.task_service.mark_failed(task_id, str(exc))

        self.executor.submit(wrapped)
        return task


_executor = None
_lock = Lock()


def get_background_executor() -> BackgroundExecutor:
    global _executor
    if _executor is None:
        with _lock:
            if _executor is None:
                _executor = BackgroundExecutor()
    return _executor
