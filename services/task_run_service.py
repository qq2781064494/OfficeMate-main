"""长任务记录服务。

上传、索引构建、benchmark、本地题库评测都会落到这张表里，
客户端通过 task_id 轮询任务状态。
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from core.bootstrap import bootstrap_runtime
from core.db import session_scope
from models.entities import TaskRunEntity
from utils.log_tool import get_logger


logger = get_logger("task_run_service")


class TaskRunService:
    def __init__(self):
        bootstrap_runtime()

    def create_task(self, task_type: str, payload: dict | list | None = None) -> dict:
        now = self._now()
        with session_scope() as session:
            row = TaskRunEntity(
                id=uuid4().hex,
                task_type=task_type,
                status="queued",
                progress_stage="queued",
                progress_message="任务已创建，等待执行。",
                payload_json=payload,
                result_json=None,
                error_message="",
                created_at=now,
                started_at="",
                finished_at="",
            )
            session.add(row)
            session.flush()
            logger.info("task_run create | task_id=%s | task_type=%s", row.id, row.task_type)
            return self._to_dict(row)

    def get_task(self, task_id: str) -> dict | None:
        with session_scope() as session:
            row = session.get(TaskRunEntity, task_id)
            return self._to_dict(row) if row else None

    def list_tasks(self, limit: int = 20, task_type: str | None = None, status: str | None = None) -> list[dict]:
        with session_scope() as session:
            query = session.query(TaskRunEntity).order_by(TaskRunEntity.created_at.desc())
            if task_type:
                query = query.filter(TaskRunEntity.task_type == task_type)
            if status:
                query = query.filter(TaskRunEntity.status == status)
            query = query.limit(limit)
            return [self._to_dict(row) for row in query.all()]

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        progress_stage: str | None = None,
        progress_message: str | None = None,
        result_json: dict | list | None = None,
        error_message: str | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> dict | None:
        with session_scope() as session:
            row = session.get(TaskRunEntity, task_id)
            if not row:
                return None
            if status is not None:
                row.status = status
            if progress_stage is not None:
                row.progress_stage = progress_stage
            if progress_message is not None:
                row.progress_message = progress_message
            if result_json is not None:
                row.result_json = result_json
            if error_message is not None:
                row.error_message = error_message
            if started and not row.started_at:
                row.started_at = self._now()
            if finished:
                row.finished_at = self._now()
            session.flush()
            logger.info("task_run update | task_id=%s | status=%s | stage=%s", row.id, row.status, row.progress_stage)
            return self._to_dict(row)

    def mark_running(self, task_id: str, progress_stage: str, progress_message: str) -> dict | None:
        return self.update_task(
            task_id,
            status="running",
            progress_stage=progress_stage,
            progress_message=progress_message,
            started=True,
        )

    def mark_completed(self, task_id: str, result_json: dict | list | None = None, progress_message: str = "任务执行完成。") -> dict | None:
        return self.update_task(
            task_id,
            status="completed",
            progress_stage="completed",
            progress_message=progress_message,
            result_json=result_json,
            finished=True,
        )

    def mark_failed(self, task_id: str, error_message: str) -> dict | None:
        return self.update_task(
            task_id,
            status="failed",
            progress_stage="failed",
            progress_message="任务执行失败。",
            error_message=error_message,
            finished=True,
        )

    def _to_dict(self, row):
        if not row:
            return None
        return {
            "id": row.id,
            "task_type": row.task_type,
            "status": row.status,
            "progress_stage": row.progress_stage,
            "progress_message": row.progress_message,
            "payload_json": row.payload_json,
            "result_json": row.result_json,
            "error_message": row.error_message,
            "created_at": row.created_at,
            "started_at": row.started_at,
            "finished_at": row.finished_at,
        }

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
