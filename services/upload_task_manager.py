"""后台上传任务管理器。

这个模块解决的是“多文件上传时页面阻塞”的问题。

如果所有文件都在 Streamlit 页面请求里同步处理，用户会遇到：
- 页面长时间转圈
- 一次上传很多文件时体验很差
- 大文件或压缩包处理失败时不容易追踪

所以这里把上传流程拆成一个后台任务系统：
1. 前台只负责提交任务
2. 后台线程负责取任务执行
3. 任务状态存放在内存结构里，页面可以轮询查看进度
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from queue import Queue
from threading import Lock, Thread
from uuid import uuid4

import config_data as config
from services.document_service import DocumentService
from utils.log_tool import get_logger


logger = get_logger("upload_task_manager")


class UploadTaskManager:
    def __init__(self):
        # _lock 用来保护多线程下对任务字典的修改。
        self._lock = Lock()
        # _tasks 保存所有后台任务的状态。
        self._tasks = {}
        # _queue 是待执行任务队列。
        self._queue = Queue()
        # _dispatcher 是一个常驻后台线程，专门负责从队列里拿任务执行。
        self._dispatcher = Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def submit_task(self, uploaded_files, category, version, custom_title=""):
        """提交一个新的后台上传任务。

        调用方通常是页面层。
        这里不会立即处理文件，而是先生成任务记录，再把 task_id 放入队列。
        """
        task_id = uuid4().hex
        task = {
            "id": task_id,
            "status": "queued",
            "stage": "等待执行",
            "message": "任务已进入后台队列。",
            "category": category,
            "version": version.strip() or config.DEFAULT_VERSION,
            "custom_title": custom_title.strip(),
            "created_at": self._now(),
            "started_at": "",
            "finished_at": "",
            "submitted_file_count": len(uploaded_files),
            "total_documents": 0,
            "completed_documents": 0,
            "success_count": 0,
            "duplicate_count": 0,
            "failed_count": 0,
            "active_document": "",
            "results": [],
            # 注意：source_files 里会临时保存原始字节流，
            # 这样后台线程真正执行时还能拿到上传文件内容。
            "source_files": [
                {
                    "file_name": uploaded_file.name,
                    "file_bytes": uploaded_file.getvalue(),
                }
                for uploaded_file in uploaded_files
            ],
        }
        with self._lock:
            self._tasks[task_id] = task
        self._queue.put(task_id)
        logger.info("upload_task submitted | task_id=%s | file_count=%s", task_id, len(uploaded_files))
        return task_id

    def get_task(self, task_id):
        """给页面层查询当前任务快照。"""
        task = self._get_task_snapshot(task_id, include_source_files=False)
        return task

    def _get_task_snapshot(self, task_id, include_source_files):
        """读取任务当前状态，并按需移除大字段。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            task_copy = deepcopy(task)
            if not include_source_files:
                task_copy.pop("source_files", None)
            return task_copy

    def _dispatch_loop(self):
        """后台分发循环。

        这是一个永不结束的 while True：
        - 队列有任务就取出来
        - 调用 _run_task 真正执行
        - 如果中途抛异常，就把任务标记为失败
        """
        while True:
            task_id = self._queue.get()
            try:
                self._run_task(task_id)
            except Exception as exc:
                logger.exception("upload_task crashed | task_id=%s | error=%s", task_id, exc)
                self._update_task(
                    task_id,
                    status="failed",
                    stage="执行失败",
                    message=f"后台任务异常终止：{exc}",
                    finished_at=self._now(),
                )
            finally:
                self._queue.task_done()

    def _run_task(self, task_id):
        """真正执行一次后台上传任务。"""
        task = self._get_task_snapshot(task_id, include_source_files=True)
        if not task:
            return

        service = DocumentService()
        self._update_task(
            task_id,
            status="running",
            stage="收集文件",
            message="正在整理上传文件和压缩包内容。",
            started_at=self._now(),
        )

        # 第一步：把用户上传的文件展开成一组“待处理项目”。
        # 例如压缩包可能会被展开成多个文档。
        upload_items = service.expand_upload_items(
            source_files=task["source_files"],
            category=task["category"],
            version=task["version"],
            custom_title=task["custom_title"],
        )
        self._update_task(task_id, source_files=[])
        if not upload_items:
            self._update_task(
                task_id,
                status="failed",
                stage="收集文件",
                message="没有找到可导入的文件，请检查上传内容。",
                finished_at=self._now(),
            )
            return

        self._update_task(
            task_id,
            stage="并行预处理",
            message="正在并行解析文档并切片。",
            total_documents=len(upload_items),
        )

        # 第二步：并行做“预处理”。
        # 预处理包括：
        # - 解析文本
        # - 计算哈希
        # - 构造中间结构
        # 这些步骤比较适合并行。
        prepare_workers = max(1, min(config.upload_prepare_workers, len(upload_items)))
        prepared_items = []
        with ThreadPoolExecutor(max_workers=prepare_workers, thread_name_prefix="upload-prepare") as executor:
            future_map = {
                executor.submit(service.prepare_upload_item, item): item
                for item in upload_items
            }
            for future in as_completed(future_map):
                item = future_map[future]
                self._update_task(task_id, active_document=item["file_name"])
                try:
                    prepared = future.result()
                except Exception as exc:
                    self._append_result(task_id, service.build_failed_result(item["title"], item["file_name"], exc))
                    continue

                registration = service.register_prepared_document(prepared)
                if registration["status"] == "duplicate":
                    self._append_result(task_id, registration["result"])
                    continue
                prepared_items.append(registration["prepared"])

        current_task = self.get_task(task_id)
        if not prepared_items:
            if current_task and current_task["completed_documents"] > 0:
                message = "任务执行完成，没有新的成功入库文档。"
                if current_task["failed_count"] == 0:
                    message = "任务执行完成，没有需要新入库的文档。"
                self._update_task(
                    task_id,
                    status="completed",
                    stage="已完成",
                    message=message,
                    finished_at=self._now(),
                    active_document="",
                )
                return
            self._update_task(
                task_id,
                status="failed",
                stage="并行预处理",
                message="全部文档都处理失败了，请检查文件内容。",
                finished_at=self._now(),
                active_document="",
            )
            return

        self._update_task(
            task_id,
            stage="并行向量化",
            message="正在受控并发生成 embeddings。",
        )

        # 第三步：并行做 embedding。
        # 这里和真正写入知识库拆开，是因为“向量生成”和“最终落库”适合不同节奏。
        embed_workers = max(1, min(config.upload_embedding_workers, len(prepared_items)))
        with ThreadPoolExecutor(max_workers=embed_workers, thread_name_prefix="upload-embed") as executor:
            future_map = {
                executor.submit(service.embed_prepared_document, prepared): prepared
                for prepared in prepared_items
            }
            for future in as_completed(future_map):
                prepared = future_map[future]
                self._update_task(
                    task_id,
                    stage="串行写库",
                    message="正在把向量结果按顺序写入知识库。",
                    active_document=prepared["file_name"],
                )
                try:
                    embedded = future.result()
                    result = service.finalize_prepared_document(embedded)
                except Exception as exc:
                    result = service.mark_prepared_document_failed(prepared, exc)
                self._append_result(task_id, result)

        # 第四步：全部完成后，把任务整体状态改成 completed。
        self._update_task(
            task_id,
            status="completed",
            stage="已完成",
            message="后台导入任务已完成。",
            finished_at=self._now(),
            active_document="",
        )

    def _append_result(self, task_id, result):
        """把单个文件的处理结果追加进任务汇总。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task["results"].append(
                {
                    "title": result.get("document", {}).get("title", result.get("title", "")),
                    "file_name": result.get("document", {}).get("file_name", result.get("file_name", "")),
                    "status": result["status"],
                    "message": result["message"],
                }
            )
            task["completed_documents"] += 1
            if result["status"] == "success":
                task["success_count"] += 1
            elif result["status"] == "duplicate":
                task["duplicate_count"] += 1
            else:
                task["failed_count"] += 1
            completed = task["completed_documents"]
            total = max(task["total_documents"], completed)
            task["message"] = (
                f"已处理 {completed}/{total} 份文档，"
                f"成功 {task['success_count']}，重复 {task['duplicate_count']}，失败 {task['failed_count']}。"
            )

    def _update_task(self, task_id, **patch):
        """更新任务字典里的若干字段。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.update(patch)

    def _now(self):
        """统一生成任务状态里使用的时间字符串。"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 这里直接初始化一个全局单例，方便页面层或服务层直接复用。
upload_task_manager = UploadTaskManager()
