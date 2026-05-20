"""持久化存储层。

历史版本把文档、问答日志、反馈记录都写进本地 JSON。
FastAPI 重构后，这个类保留原有名字与方法签名，但底层实现改成 MySQL。

这样做的好处是：
1. RAG / Agent / 评测主流程几乎不用大改
2. 旧代码继续通过“字典接口”访问存储层
3. 可以逐步把项目从脚本式结构平滑迁移到服务化架构
"""

from datetime import datetime
from uuid import uuid4

import config_data as config
from core.bootstrap import bootstrap_runtime
from core.db import session_scope
from models.entities import DocumentEntity, FeedbackLogEntity, QALogEntity
from utils.log_tool import get_logger


logger = get_logger("storage_service")


class JsonStorageService:
    def __init__(self):
        # 启动时自动建库建表，保证旧业务类可以被直接实例化。
        bootstrap_runtime()

    def _sort_desc(self, records, field):
        """按某个字段做倒序排序，常用于时间字段。"""
        return sorted(records, key=lambda item: item.get(field, ""), reverse=True)

    def _now(self):
        """统一生成项目中使用的时间字符串格式。"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def list_documents(self):
        """列出全部文档，并按上传时间倒序返回。"""
        with session_scope() as session:
            rows = session.query(DocumentEntity).order_by(DocumentEntity.uploaded_at.desc()).all()
            return [self._document_to_dict(row) for row in rows]

    def get_document_by_id(self, document_id):
        """按文档主键查询单条文档记录。"""
        with session_scope() as session:
            row = session.get(DocumentEntity, document_id)
            return self._document_to_dict(row) if row else None

    def get_document_by_hash(self, file_hash):
        """按文件哈希查询，用于避免重复导入同一份文件。"""
        with session_scope() as session:
            row = session.query(DocumentEntity).filter(DocumentEntity.file_hash == file_hash).first()
            return self._document_to_dict(row) if row else None

    def add_document(self, record):
        """新增文档记录；如果缺少 id / uploaded_at，会在这里补齐。"""
        payload = dict(record)
        payload.setdefault("id", uuid4().hex)
        payload.setdefault("uploaded_at", self._now())
        payload.setdefault("updated_at", payload["uploaded_at"])
        with session_scope() as session:
            row = DocumentEntity(**self._normalize_document_payload(payload))
            session.add(row)
            session.flush()
            logger.info("storage add_document | document_id=%s | title=%s", row.id, row.title)
            return self._document_to_dict(row)

    def update_document(self, document_id, patch):
        """更新已有文档记录，并返回更新后的对象。"""
        with session_scope() as session:
            row = session.get(DocumentEntity, document_id)
            if not row:
                return None
            normalized_patch = self._normalize_document_payload(patch, partial=True)
            normalized_patch["updated_at"] = self._now()
            for key, value in normalized_patch.items():
                setattr(row, key, value)
            session.flush()
            logger.info("storage update_document | document_id=%s | patch_keys=%s", document_id, list(patch.keys()))
            return self._document_to_dict(row)

    def delete_document(self, document_id):
        """删除文档索引中的一条记录。"""
        with session_scope() as session:
            row = session.get(DocumentEntity, document_id)
            if not row:
                return False
            session.delete(row)
            logger.info("storage delete_document | document_id=%s", document_id)
            return True

    def list_qa_logs(self, limit=None):
        """列出全部问答日志，可选限制返回数量。"""
        with session_scope() as session:
            query = session.query(QALogEntity).order_by(QALogEntity.created_at.desc())
            if limit:
                query = query.limit(limit)
            return [self._qa_log_to_dict(row) for row in query.all()]

    def list_session_logs(self, session_id, limit=None):
        """读取某个会话下的历史问答，用于构建多轮对话上下文。"""
        with session_scope() as session:
            query = (
                session.query(QALogEntity)
                .filter(QALogEntity.session_id == session_id)
                .order_by(QALogEntity.created_at.asc())
            )
            records = [self._qa_log_to_dict(row) for row in query.all()]
            return records[-limit:] if limit else records

    def add_qa_log(self, record):
        """新增一条问答记录。"""
        payload = dict(record)
        payload.setdefault("id", uuid4().hex)
        payload.setdefault("created_at", self._now())
        payload.setdefault("source_docs_json", [])
        with session_scope() as session:
            row = QALogEntity(**self._normalize_qa_log_payload(payload))
            session.add(row)
            session.flush()
            logger.info(
                "storage add_qa_log | qa_log_id=%s | session_id=%s | question_type=%s",
                row.id,
                row.session_id,
                row.question_type,
            )
            return self._qa_log_to_dict(row)

    def list_feedback(self):
        """列出全部反馈记录。"""
        with session_scope() as session:
            rows = session.query(FeedbackLogEntity).order_by(FeedbackLogEntity.created_at.desc()).all()
            return [self._feedback_to_dict(row) for row in rows]

    def get_feedback_by_qa_log_id(self, qa_log_id):
        """根据问答记录 ID 查询是否已经有反馈。"""
        with session_scope() as session:
            row = session.query(FeedbackLogEntity).filter(FeedbackLogEntity.qa_log_id == qa_log_id).first()
            return self._feedback_to_dict(row) if row else None

    def upsert_feedback(self, qa_log_id, rating, comment, session_id):
        """更新已有反馈，或在不存在时插入一条新反馈。"""
        current_time = self._now()
        with session_scope() as session:
            row = session.query(FeedbackLogEntity).filter(FeedbackLogEntity.qa_log_id == qa_log_id).first()
            if row:
                row.rating = rating
                row.comment = comment
                row.session_id = session_id
                row.updated_at = current_time
                session.flush()
                logger.info("storage update_feedback | qa_log_id=%s | rating=%s", qa_log_id, rating)
                return self._feedback_to_dict(row)

            row = FeedbackLogEntity(
                id=uuid4().hex,
                qa_log_id=qa_log_id,
                rating=rating,
                comment=comment,
                session_id=session_id,
                created_at=current_time,
                updated_at=current_time,
            )
            session.add(row)
            session.flush()
            logger.info("storage add_feedback | qa_log_id=%s | rating=%s", qa_log_id, rating)
            return self._feedback_to_dict(row)

    def get_stats(self):
        """统计管理页展示用的核心指标。"""
        with session_scope() as session:
            documents = [self._document_to_dict(row) for row in session.query(DocumentEntity).all()]
            categories = {record.get("category") for record in documents if record.get("category")}
            qa_count = session.query(QALogEntity).count()
            feedback_count = session.query(FeedbackLogEntity).count()
            return {
                "document_count": len(documents),
                "category_count": len(categories),
                "qa_count": qa_count,
                "feedback_count": feedback_count,
            }

    def _normalize_document_payload(self, payload, partial=False):
        normalized = {}
        fields = {
            "id": "",
            "title": "",
            "category": "",
            "version": "",
            "file_name": "",
            "file_type": "",
            "file_hash": "",
            "text_length": 0,
            "chunk_count": 0,
            "status": "processing",
            "source_label": "",
            "raw_path": "",
            "error": "",
            "uploaded_at": self._now(),
            "updated_at": self._now(),
        }
        for key, default in fields.items():
            if key in payload:
                value = payload[key]
                if key in {"text_length", "chunk_count"}:
                    value = int(value or 0)
                normalized[key] = value
            elif not partial:
                normalized[key] = default
        return normalized

    def _normalize_qa_log_payload(self, payload):
        return {
            "id": payload.get("id", uuid4().hex),
            "session_id": payload.get("session_id", ""),
            "question": payload.get("question", ""),
            "answer": payload.get("answer", ""),
            "category": payload.get("category", "全部"),
            "question_type": payload.get("question_type", ""),
            "source_docs_json": payload.get("source_docs_json", payload.get("source_docs", [])) or [],
            "mode": payload.get("mode", "chat"),
            "trace_json": payload.get("trace_json", payload.get("trace")),
            "decision_json": payload.get("decision_json", payload.get("decision")),
            "created_at": payload.get("created_at", self._now()),
        }

    def _document_to_dict(self, row):
        if not row:
            return None
        return {
            "id": row.id,
            "title": row.title,
            "category": row.category,
            "version": row.version,
            "file_name": row.file_name,
            "file_type": row.file_type,
            "file_hash": row.file_hash,
            "text_length": row.text_length,
            "chunk_count": row.chunk_count,
            "status": row.status,
            "source_label": row.source_label,
            "raw_path": row.raw_path,
            "error": row.error,
            "uploaded_at": row.uploaded_at,
            "updated_at": row.updated_at,
        }

    def _qa_log_to_dict(self, row):
        if not row:
            return None
        return {
            "id": row.id,
            "session_id": row.session_id,
            "question": row.question,
            "answer": row.answer,
            "category": row.category,
            "question_type": row.question_type,
            "source_docs": row.source_docs_json or [],
            "source_docs_json": row.source_docs_json or [],
            "mode": row.mode,
            "trace": row.trace_json,
            "trace_json": row.trace_json,
            "decision": row.decision_json,
            "decision_json": row.decision_json,
            "created_at": row.created_at,
        }

    def _feedback_to_dict(self, row):
        if not row:
            return None
        return {
            "id": row.id,
            "qa_log_id": row.qa_log_id,
            "session_id": row.session_id,
            "rating": row.rating,
            "comment": row.comment,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
