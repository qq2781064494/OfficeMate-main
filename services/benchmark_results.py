"""Benchmark 评测结果存储。

职责：
1. 保存每次 run 的摘要信息
2. 保存每道题的详细结果
3. 提供列表和详情读取接口，供页面层展示
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

import config_data as config
from utils.log_tool import get_logger


logger = get_logger("benchmark_results")


class BenchmarkResultStore:
    """管理 benchmark run 摘要和 question 级明细。"""

    _write_lock = Lock()

    def __init__(self):
        config.ensure_runtime_dirs()

    def list_runs(self, limit: int | None = 20, subset: str | None = None) -> list[dict]:
        records = self._read_json_array(config.BENCHMARK_RUN_INDEX_PATH)
        if subset:
            records = [record for record in records if record.get("subset") == subset]
        records = sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)
        return records[:limit] if limit else records

    def get_run_summary(self, run_id: str) -> dict | None:
        for record in self._read_json_array(config.BENCHMARK_RUN_INDEX_PATH):
            if record.get("run_id") == run_id:
                return record
        return None

    def load_run_details(self, run_id: str) -> list[dict]:
        detail_path = self._detail_path(run_id)
        if not detail_path.exists():
            return []
        rows = []
        with detail_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def save_run(self, summary: dict, details: list[dict]) -> dict:
        records = self._read_json_array(config.BENCHMARK_RUN_INDEX_PATH)
        run_record = dict(summary)
        run_record.setdefault("run_id", self._build_run_id(summary.get("subset", "benchmark")))
        run_record.setdefault("created_at", self._now())
        detail_path = self._detail_path(run_record["run_id"])
        run_record["detail_path"] = str(detail_path.relative_to(config.BASE_DIR))

        with self._write_lock:
            detail_path.parent.mkdir(parents=True, exist_ok=True)
            with detail_path.open("w", encoding="utf-8") as file:
                for item in details:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")

            records = [record for record in records if record.get("run_id") != run_record["run_id"]]
            records.append(run_record)
            config.BENCHMARK_RUN_INDEX_PATH.write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        logger.info(
            "benchmark_results save_run | run_id=%s | subset=%s | question_count=%s",
            run_record["run_id"],
            run_record.get("subset", ""),
            run_record.get("question_count", 0),
        )
        return run_record

    def _detail_path(self, run_id: str) -> Path:
        return config.BENCHMARK_RUN_DIR / f"{run_id}_details.jsonl"

    def _read_json_array(self, path: Path) -> list[dict]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _build_run_id(self, subset: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ragbench_{subset}_{timestamp}_{uuid4().hex[:6]}"
