"""把旧版 JSON/JSONL 评测存档补录到新的 MySQL 表。

迁移内容：
1. storage/benchmark_runs/run_index.json + *_details.jsonl -> benchmark_runs / benchmark_run_details
2. storage/local_eval_kb_index.json 与 local_eval_kb/*/index_meta.json -> local_eval_knowledge_bases
3. storage/benchmark_corpus/*/manifest.json 与 benchmark_chroma/*/index_meta.json -> benchmark_corpus_registry

这个脚本是幂等的：重复执行会覆盖同 run_id / knowledge_base_id / subset+registry_type 记录。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import config_data as config
from core.bootstrap import bootstrap_runtime
from core.db import session_scope
from models.entities import BenchmarkCorpusRegistryEntity, LocalEvalKnowledgeBaseEntity
from services.benchmark_results import BenchmarkResultStore


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(sanitize(json.loads(line)))
    return rows


def migrate_benchmark_runs() -> dict:
    store = BenchmarkResultStore()
    run_index_path = config.BENCHMARK_RUN_INDEX_PATH
    summaries = read_json(run_index_path, [])
    migrated = 0
    for summary in summaries:
        run_id = summary.get("run_id")
        detail_path = config.BASE_DIR / str(summary.get("detail_path", ""))
        if not run_id and detail_path.name.endswith("_details.jsonl"):
            run_id = detail_path.name.replace("_details.jsonl", "")
            summary["run_id"] = run_id
        details = read_jsonl(detail_path)
        store.save_run(sanitize(summary), details)
        migrated += 1
    return {"benchmark_runs_migrated": migrated}


def migrate_local_eval_kbs() -> dict:
    candidates: dict[str, dict] = {}
    index_path = config.LOCAL_EVAL_KB_INDEX_PATH
    for record in read_json(index_path, []):
        if record.get("knowledge_base_id"):
            candidates[record["knowledge_base_id"]] = sanitize(record)

    for meta_path in sorted(config.LOCAL_EVAL_KB_DIR.glob("*/index_meta.json")):
        record = sanitize(read_json(meta_path, {}))
        if record.get("knowledge_base_id"):
            candidates[record["knowledge_base_id"]] = record

    migrated = 0
    with session_scope() as session:
        for record in candidates.values():
            kb_id = record["knowledge_base_id"]
            row = session.get(LocalEvalKnowledgeBaseEntity, kb_id)
            if not row:
                row = LocalEvalKnowledgeBaseEntity(knowledge_base_id=kb_id)
                session.add(row)
            row.knowledge_base_name = record.get("knowledge_base_name", kb_id)
            row.document_count = int(record.get("document_count", 0) or 0)
            row.chunk_count = int(record.get("chunk_count", 0) or 0)
            row.chunk_config_json = record.get("chunk_config", {})
            row.created_at = record.get("created_at", "")
            row.updated_at = record.get("updated_at", "")
            row.persist_directory = record.get("persist_directory", "")
            row.manifest_path = record.get("manifest_path", "")
            row.source_files_json = record.get("source_files", [])
            migrated += 1
    return {"local_eval_kbs_migrated": migrated}


def migrate_benchmark_registry() -> dict:
    migrated = 0
    with session_scope() as session:
        for manifest_path in sorted(config.BENCHMARK_CORPUS_DIR.glob("*/manifest.json")):
            subset = manifest_path.parent.name
            manifest = read_json(manifest_path, [])
            row = (
                session.query(BenchmarkCorpusRegistryEntity)
                .filter(
                    BenchmarkCorpusRegistryEntity.subset == subset,
                    BenchmarkCorpusRegistryEntity.registry_type == "corpus",
                )
                .first()
            )
            if not row:
                row = BenchmarkCorpusRegistryEntity(subset=subset, registry_type="corpus")
                session.add(row)
            row.document_count = len(manifest)
            row.chunk_count = 0
            row.manifest_path = str(manifest_path.relative_to(config.BASE_DIR))
            row.chunk_config_json = None
            row.metadata_json = {
                "subset": subset,
                "document_count": len(manifest),
                "manifest_path": str(manifest_path.relative_to(config.BASE_DIR)),
                "status": "migrated",
            }
            migrated += 1

        for meta_path in sorted(config.BENCHMARK_CHROMA_DIR.glob("*/index_meta.json")):
            subset = meta_path.parent.name
            meta = sanitize(read_json(meta_path, {}))
            row = (
                session.query(BenchmarkCorpusRegistryEntity)
                .filter(
                    BenchmarkCorpusRegistryEntity.subset == subset,
                    BenchmarkCorpusRegistryEntity.registry_type == "index",
                )
                .first()
            )
            if not row:
                row = BenchmarkCorpusRegistryEntity(subset=subset, registry_type="index")
                session.add(row)
            row.document_count = int(meta.get("document_count", 0) or 0)
            row.chunk_count = int(meta.get("chunk_count", 0) or 0)
            row.manifest_path = str(meta_path.relative_to(config.BASE_DIR))
            row.chunk_config_json = meta.get("chunk_config")
            row.metadata_json = meta
            migrated += 1
    return {"benchmark_registry_rows_migrated": migrated}


def main() -> None:
    bootstrap_runtime()
    summary = {}
    summary.update(migrate_benchmark_runs())
    summary.update(migrate_local_eval_kbs())
    summary.update(migrate_benchmark_registry())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
