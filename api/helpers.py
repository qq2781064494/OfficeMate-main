"""API 层辅助函数。"""

from __future__ import annotations

from pathlib import Path

from services.benchmark_store import BenchmarkChunkConfig


def build_chunk_config(
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    max_split_char_number: int | None = None,
) -> BenchmarkChunkConfig:
    base = BenchmarkChunkConfig()
    if chunk_size is not None:
        base.chunk_size = int(chunk_size)
    if chunk_overlap is not None:
        base.chunk_overlap = int(chunk_overlap)
    if max_split_char_number is not None:
        base.max_split_char_number = int(max_split_char_number)
    return base


def resolve_sample_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else Path.cwd() / path
