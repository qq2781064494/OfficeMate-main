"""Benchmark 评测结果存储。

历史版本使用 JSON + JSONL。
重构后改为 MySQL 持久化摘要与题级明细，兼容已有服务层调用方式。
同时会在入库前清洗 NaN / Infinity，避免 MySQL JSON 字段报错。
"""

from __future__ import annotations

from datetime import datetime
import math
from uuid import uuid4

import config_data as config
from core.bootstrap import bootstrap_runtime
from core.db import session_scope
from models.entities import BenchmarkRunDetailEntity, BenchmarkRunEntity
from utils.log_tool import get_logger


logger = get_logger("benchmark_results")


class BenchmarkResultStore:
    """管理 benchmark run 摘要和 question 级明细。"""

    def __init__(self):
        bootstrap_runtime()

    def list_runs(self, limit: int | None = 20, subset: str | None = None) -> list[dict]:
        with session_scope() as session:
            query = session.query(BenchmarkRunEntity).order_by(BenchmarkRunEntity.created_at.desc())
            if subset:
                query = query.filter(BenchmarkRunEntity.subset == subset)
            if limit:
                query = query.limit(limit)
            return [self._summary_to_dict(row) for row in query.all()]

    def get_run_summary(self, run_id: str) -> dict | None:
        with session_scope() as session:
            row = session.get(BenchmarkRunEntity, run_id)
            return self._summary_to_dict(row) if row else None

    def load_run_details(self, run_id: str) -> list[dict]:
        with session_scope() as session:
            rows = (
                session.query(BenchmarkRunDetailEntity)
                .filter(BenchmarkRunDetailEntity.run_id == run_id)
                .order_by(BenchmarkRunDetailEntity.question_order.asc())
                .all()
            )
            return [row.detail_json or {} for row in rows]

    def save_run(self, summary: dict, details: list[dict]) -> dict:
        run_record = self._sanitize_json_payload(dict(summary))
        run_record.setdefault("run_id", self._build_run_id(summary.get("subset", "benchmark")))
        run_record.setdefault("created_at", self._now())
        run_record["detail_path"] = f"mysql://benchmark_run_details/{run_record['run_id']}"
        details = [self._sanitize_json_payload(item) for item in details]

        with session_scope() as session:
            existing = session.get(BenchmarkRunEntity, run_record["run_id"])
            if existing:
                session.query(BenchmarkRunDetailEntity).filter(
                    BenchmarkRunDetailEntity.run_id == run_record["run_id"]
                ).delete()
                row = existing
            else:
                row = BenchmarkRunEntity(id=run_record["run_id"])
                session.add(row)

            row.run_type = run_record.get("mode", run_record.get("run_type", "benchmark"))
            row.subset = run_record.get("subset", "")
            row.split = run_record.get("split", "")
            row.knowledge_base_id = run_record.get("knowledge_base_id", "")
            row.knowledge_base_name = run_record.get("knowledge_base_name", "")
            row.retriever_strategy = run_record.get("retriever_strategy", "hybrid")
            row.top_k = int(run_record.get("top_k", 5) or 5)
            row.question_count = int(run_record.get("question_count", len(details)) or 0)
            row.enable_query_rewrite = bool(run_record.get("enable_query_rewrite", True))
            row.enable_rerank = bool(run_record.get("enable_rerank", True))
            row.enable_faithfulness = bool(run_record.get("enable_faithfulness", True))
            row.summary_json = run_record
            row.created_at = run_record["created_at"]
            session.flush()

            for index, item in enumerate(details, start=1):
                session.add(
                    BenchmarkRunDetailEntity(
                        run_id=run_record["run_id"],
                        question_order=index,
                        detail_json=item,
                    )
                )

        logger.info(
            "benchmark_results save_run | run_id=%s | subset=%s | question_count=%s",
            run_record["run_id"],
            run_record.get("subset", ""),
            run_record.get("question_count", 0),
        )
        return run_record

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _build_run_id(self, subset: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ragbench_{subset}_{timestamp}_{uuid4().hex[:6]}"

    def _summary_to_dict(self, row: BenchmarkRunEntity | None) -> dict | None:
        if not row:
            return None
        summary = dict(row.summary_json or {})
        summary.setdefault("run_id", row.id)
        summary.setdefault("detail_path", f"mysql://benchmark_run_details/{row.id}")
        summary.setdefault("created_at", row.created_at)
        summary.setdefault("subset", row.subset)
        summary.setdefault("split", row.split)
        summary.setdefault("knowledge_base_id", row.knowledge_base_id)
        summary.setdefault("knowledge_base_name", row.knowledge_base_name)
        return summary

    def _sanitize_json_payload(self, value):
        if isinstance(value, dict):
            return {key: self._sanitize_json_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_json_payload(item) for item in value]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
