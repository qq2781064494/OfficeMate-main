"""本地 JSON 存储层。

这个项目没有接数据库，而是把文档元数据、问答日志、反馈记录
都存进本地 JSON 文件里。这个类就相当于一个轻量版 DAO / Repository。
"""

import json
from datetime import datetime
from threading import Lock
from uuid import uuid4

import config_data as config
from utils.log_tool import get_logger


logger = get_logger("storage_service")


class JsonStorageService:
    # 多线程情况下写文件时共用一把锁，降低并发写坏 JSON 的风险。
    _write_lock = Lock()

    def __init__(self):
        # 每次实例化时都确保目录和 JSON 文件存在，避免后续读写时报路径错误。
        config.ensure_runtime_dirs()

    def _read_records(self, path):
        """读取 JSON 数组；文件不存在或损坏时返回空列表。"""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_records(self, path, records):
        """把完整记录列表写回文件。当前实现是“整文件覆盖写入”。"""
        with self._write_lock:
            path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        logger.debug("storage_write | path=%s | record_count=%s", path, len(records))

    def _sort_desc(self, records, field):
        """按某个字段做倒序排序，常用于时间字段。"""
        return sorted(records, key=lambda item: item.get(field, ""), reverse=True)

    def _now(self):
        """统一生成项目中使用的时间字符串格式。"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def list_documents(self):
        """列出全部文档，并按上传时间倒序返回。"""
        return self._sort_desc(self._read_records(config.DOCUMENT_INDEX_PATH), "uploaded_at")

    def get_document_by_id(self, document_id):
        """按文档主键查询单条文档记录。"""
        for record in self._read_records(config.DOCUMENT_INDEX_PATH):
            if record.get("id") == document_id:
                return record
        return None

    def get_document_by_hash(self, file_hash):
        """按文件哈希查询，用于避免重复导入同一份文件。"""
        for record in self._read_records(config.DOCUMENT_INDEX_PATH):
            if record.get("file_hash") == file_hash:
                return record
        return None

    def add_document(self, record):
        """新增文档记录；如果缺少 id / uploaded_at，会在这里补齐。"""
        records = self._read_records(config.DOCUMENT_INDEX_PATH)
        if "id" not in record:
            record["id"] = uuid4().hex
        if "uploaded_at" not in record:
            record["uploaded_at"] = self._now()
        records.append(record)
        self._write_records(config.DOCUMENT_INDEX_PATH, records)
        logger.info("storage add_document | document_id=%s | title=%s", record["id"], record.get("title", ""))
        return record

    def update_document(self, document_id, patch):
        """更新已有文档记录，并返回更新后的对象。"""
        records = self._read_records(config.DOCUMENT_INDEX_PATH)
        updated = None
        for record in records:
            if record.get("id") == document_id:
                record.update(patch)
                updated = record
                break
        self._write_records(config.DOCUMENT_INDEX_PATH, records)
        logger.info("storage update_document | document_id=%s | patch_keys=%s", document_id, list(patch.keys()))
        return updated

    def delete_document(self, document_id):
        """删除文档索引中的一条记录。"""
        records = self._read_records(config.DOCUMENT_INDEX_PATH)
        remaining_records = [record for record in records if record.get("id") != document_id]
        if len(remaining_records) == len(records):
            return False
        self._write_records(config.DOCUMENT_INDEX_PATH, remaining_records)
        logger.info("storage delete_document | document_id=%s", document_id)
        return True

    def list_qa_logs(self, limit=None):
        """列出全部问答日志，可选限制返回数量。"""
        records = self._sort_desc(self._read_records(config.QA_LOG_PATH), "created_at")
        return records[:limit] if limit else records

    def list_session_logs(self, session_id, limit=None):
        """读取某个会话下的历史问答，用于构建多轮对话上下文。"""
        records = [
            record
            for record in self._read_records(config.QA_LOG_PATH)
            if record.get("session_id") == session_id
        ]
        records = sorted(records, key=lambda item: item.get("created_at", ""))
        return records[-limit:] if limit else records

    def add_qa_log(self, record):
        """新增一条问答记录。"""
        records = self._read_records(config.QA_LOG_PATH)
        if "id" not in record:
            record["id"] = uuid4().hex
        if "created_at" not in record:
            record["created_at"] = self._now()
        records.append(record)
        self._write_records(config.QA_LOG_PATH, records)
        logger.info(
            "storage add_qa_log | qa_log_id=%s | session_id=%s | question_type=%s",
            record["id"],
            record.get("session_id", ""),
            record.get("question_type", ""),
        )
        return record

    def list_feedback(self):
        """列出全部反馈记录。"""
        return self._sort_desc(self._read_records(config.FEEDBACK_PATH), "created_at")

    def get_feedback_by_qa_log_id(self, qa_log_id):
        """根据问答记录 ID 查询是否已经有反馈。"""
        for record in self._read_records(config.FEEDBACK_PATH):
            if record.get("qa_log_id") == qa_log_id:
                return record
        return None

    def upsert_feedback(self, qa_log_id, rating, comment, session_id):
        """更新已有反馈，或在不存在时插入一条新反馈。"""
        records = self._read_records(config.FEEDBACK_PATH)
        current_time = self._now()
        for record in records:
            if record.get("qa_log_id") == qa_log_id:
                record["rating"] = rating
                record["comment"] = comment
                record["updated_at"] = current_time
                self._write_records(config.FEEDBACK_PATH, records)
                logger.info("storage update_feedback | qa_log_id=%s | rating=%s", qa_log_id, rating)
                return record

        new_record = {
            "id": uuid4().hex,
            "qa_log_id": qa_log_id,
            "rating": rating,
            "comment": comment,
            "session_id": session_id,
            "created_at": current_time,
            "updated_at": current_time,
        }
        records.append(new_record)
        self._write_records(config.FEEDBACK_PATH, records)
        logger.info("storage add_feedback | qa_log_id=%s | rating=%s", qa_log_id, rating)
        return new_record

    def get_stats(self):
        """统计管理页展示用的核心指标。"""
        documents = self._read_records(config.DOCUMENT_INDEX_PATH)
        qa_logs = self._read_records(config.QA_LOG_PATH)
        feedback = self._read_records(config.FEEDBACK_PATH)
        categories = {record.get("category") for record in documents if record.get("category")}
        return {
            "document_count": len(documents),
            "category_count": len(categories),
            "qa_count": len(qa_logs),
            "feedback_count": len(feedback),
        }
