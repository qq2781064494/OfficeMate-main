"""下载并整理 Hugging Face 上的 RAGBench 数据集。

用途：
1. 从 Hugging Face 下载 `galileo-ai/ragbench`
2. 按 subset / split 导出规范化 JSONL
3. 额外生成一份更贴近 OfficeMate 评测格式的样本文件

示例：
    python scripts/download_ragbench.py
    python scripts/download_ragbench.py --all-subsets --splits test
    python scripts/download_ragbench.py --subsets emanual techqa --splits test validation --limit 100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_REPO_ID = "galileo-ai/ragbench"
FALLBACK_REPO_ID = "rungalileo/ragbench"
ALL_SUBSETS = [
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
]
DEFAULT_SUBSETS = ALL_SUBSETS
DEFAULT_SPLITS = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并整理 Hugging Face RAGBench 数据集")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"数据集 repo id，默认 {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--all-subsets",
        action="store_true",
        help="显式指定下载全部 12 个子集；当前脚本默认就是全量下载",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="指定子集名称，例如 --subsets emanual techqa",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="指定 split，默认只下载 test",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个 subset/split 只保留前 N 条，便于快速试验",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "storage" / "ragbench"),
        help="输出目录，默认 storage/ragbench",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subsets = resolve_subsets(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_loader = build_dataset_loader()
    manifest_rows = []

    for subset in subsets:
        for split in args.splits:
            dataset, resolved_repo_id = load_subset_split(
                dataset_loader=dataset_loader,
                repo_id=args.repo_id,
                subset=subset,
                split=split,
            )
            if args.limit:
                dataset = dataset.select(range(min(args.limit, len(dataset))))

            subset_dir = output_dir / subset
            subset_dir.mkdir(parents=True, exist_ok=True)

            normalized_path = subset_dir / f"{split}.jsonl"
            officemate_eval_path = subset_dir / f"{split}_officemate_eval.json"

            if normalized_path.exists() and officemate_eval_path.exists():
                normalized_count = count_jsonl_rows(normalized_path)
                eval_count = count_json_array_items(officemate_eval_path)
                manifest_rows.append(
                    {
                        "repo_id": "cached",
                        "subset": subset,
                        "split": split,
                        "row_count": normalized_count,
                        "officemate_eval_count": eval_count,
                        "normalized_path": str(normalized_path.relative_to(PROJECT_ROOT)),
                        "officemate_eval_path": str(officemate_eval_path.relative_to(PROJECT_ROOT)),
                    }
                )
                print(
                    f"[skip] subset={subset} split={split} rows={normalized_count} "
                    f"eval_samples={eval_count}"
                )
                continue

            normalized_count = write_normalized_jsonl(dataset, subset, split, normalized_path)
            eval_count = write_officemate_eval_samples(dataset, subset, split, officemate_eval_path)

            manifest_rows.append(
                {
                    "repo_id": resolved_repo_id,
                    "subset": subset,
                    "split": split,
                    "row_count": normalized_count,
                    "officemate_eval_count": eval_count,
                    "normalized_path": str(normalized_path.relative_to(PROJECT_ROOT)),
                    "officemate_eval_path": str(officemate_eval_path.relative_to(PROJECT_ROOT)),
                }
            )
            print(
                f"[done] subset={subset} split={split} rows={normalized_count} "
                f"eval_samples={eval_count} repo={resolved_repo_id}"
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[manifest] {manifest_path}")


def resolve_subsets(args: argparse.Namespace) -> list[str]:
    if args.all_subsets:
        return ALL_SUBSETS
    if args.subsets:
        unknown = sorted(set(args.subsets) - set(ALL_SUBSETS))
        if unknown:
            raise ValueError(f"未知 subset: {', '.join(unknown)}")
        return args.subsets
    return DEFAULT_SUBSETS


def build_dataset_loader():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "缺少 datasets 依赖，请先执行 `pip install -r requirements.txt`。"
        ) from exc
    return load_dataset


def load_subset_split(dataset_loader, repo_id: str, subset: str, split: str):
    errors: list[str] = []
    for candidate_repo_id in [repo_id, FALLBACK_REPO_ID]:
        try:
            dataset = dataset_loader(candidate_repo_id, subset, split=split)
            return dataset, candidate_repo_id
        except Exception as exc:  # pragma: no cover - 依赖网络环境
            errors.append(f"{candidate_repo_id}: {exc}")
    raise RuntimeError(
        f"无法加载 subset={subset} split={split}。"
        f"尝试过的 repo: {' | '.join(errors)}。"
        "如果错误里包含 scipy/liblapack，请换一个依赖完整的 Python 环境；"
        "如果包含 503、SSL 或 ProxyError，请检查当前网络对 huggingface.co 的访问。"
    )


def write_normalized_jsonl(dataset, subset: str, split: str, output_path: Path) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for row in dataset:
            normalized = normalize_row(row, subset=subset, split=split)
            file.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_officemate_eval_samples(dataset, subset: str, split: str, output_path: Path) -> int:
    samples = []
    for row in dataset:
        normalized = normalize_row(row, subset=subset, split=split)
        relevant_titles = infer_relevant_titles(
            normalized["documents"],
            normalized["all_relevant_sentence_keys"],
        )
        if not relevant_titles:
            # OfficeMate 当前评测以“期望命中文档标题”为核心，没有可推断标题时先跳过。
            continue
        samples.append(
            {
                "query": normalized["question"],
                "category": subset,
                "expected_titles": relevant_titles,
                "metadata": {
                    "ragbench_id": normalized["id"],
                    "subset": subset,
                    "split": split,
                    "gold_response": normalized["response"],
                },
            }
        )

    output_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(samples)


def normalize_row(row: dict, subset: str, split: str) -> dict:
    documents = [
        parse_document_text(text, index)
        for index, text in enumerate(row.get("documents") or [])
    ]
    return {
        "id": str(row.get("id", "")),
        "subset": subset,
        "split": split,
        "dataset_name": row.get("dataset_name"),
        "question": row.get("question", ""),
        "response": row.get("response", ""),
        "documents": documents,
        "response_sentences": normalize_sentence_pairs(row.get("response_sentences") or []),
        "sentence_support_information": row.get("sentence_support_information") or [],
        "unsupported_response_sentence_keys": list(row.get("unsupported_response_sentence_keys") or []),
        "all_relevant_sentence_keys": list(row.get("all_relevant_sentence_keys") or []),
        "all_utilized_sentence_keys": list(row.get("all_utilized_sentence_keys") or []),
        "adherence_score": row.get("adherence_score"),
        "relevance_score": row.get("relevance_score"),
        "utilization_score": row.get("utilization_score"),
        "completeness_score": row.get("completeness_score"),
        "gpt3_adherence": row.get("gpt3_adherence"),
        "gpt3_context_relevance": row.get("gpt3_context_relevance"),
        "gpt35_utilization": row.get("gpt35_utilization"),
        "trulens_groundedness": row.get("trulens_groundedness"),
        "trulens_context_relevance": row.get("trulens_context_relevance"),
        "ragas_faithfulness": row.get("ragas_faithfulness"),
        "ragas_context_relevance": row.get("ragas_context_relevance"),
    }


def parse_document_text(text: str, index: int) -> dict:
    text = text or ""
    title_match = re.search(r"Title:\s*(.*)", text)
    passage_match = re.search(r"Passage:\s*(.*)", text, flags=re.S)
    title = title_match.group(1).strip() if title_match else f"document_{index}"
    passage = passage_match.group(1).strip() if passage_match else text.strip()
    return {
        "doc_index": index,
        "title": title,
        "passage": passage,
        "raw_text": text,
    }


def normalize_sentence_pairs(items: Iterable) -> list[dict]:
    if not items:
        return []
    normalized = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append({"key": item[0], "text": item[1]})
        else:
            normalized.append({"key": None, "text": str(item)})
    return normalized


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def count_json_array_items(path: Path) -> int:
    return len(json.loads(path.read_text(encoding="utf-8")))


def infer_relevant_titles(documents: list[dict], sentence_keys: list[str]) -> list[str]:
    doc_index_to_title = {document["doc_index"]: document["title"] for document in documents}
    relevant_indices = []
    for key in sentence_keys:
        match = re.match(r"(\d+)", key)
        if not match:
            continue
        relevant_indices.append(int(match.group(1)))

    seen = set()
    titles = []
    for doc_index in relevant_indices:
        title = doc_index_to_title.get(doc_index)
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


if __name__ == "__main__":
    main()
