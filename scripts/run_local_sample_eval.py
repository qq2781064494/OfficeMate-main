"""运行本地 sample_docs 题库评测。

使用独立的本地测评知识库实例，不会复用 app 正常问答数据库。
答案生成仍然走主问答链路，不会走 benchmark 专用 answer prompt。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import config_data as config
from services.benchmark_store import BenchmarkChunkConfig
from services.local_eval_service import LocalEvalConfig, LocalEvalService
from services.local_eval_store import LocalEvalCorpusStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local sample_docs evaluation with the main RAG pipeline.")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate the first N samples. 0 means all.")
    parser.add_argument("--disable-ragas", action="store_true", help="Skip Ragas metrics.")
    parser.add_argument("--dataset", choices=["full", "complex"], default="full", help="Choose which local dataset to evaluate.")
    parser.add_argument("--knowledge-base-id", default="", help="Use an existing local evaluation knowledge base id.")
    parser.add_argument("--knowledge-base-name", default="", help="Create or rebuild a knowledge base with this name when no id is given.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the named knowledge base if it already exists.")
    parser.add_argument("--retriever-strategy", choices=["hybrid", "vector", "bm25"], default="hybrid", help="Retriever strategy.")
    parser.add_argument("--chunk-size", type=int, default=config.benchmark_chunk_size, help="Chunk size for newly built knowledge bases.")
    parser.add_argument("--chunk-overlap", type=int, default=config.benchmark_chunk_overlap, help="Chunk overlap for newly built knowledge bases.")
    parser.add_argument(
        "--max-split-char-number",
        type=int,
        default=config.benchmark_max_split_char_number,
        help="Maximum text length before splitting for newly built knowledge bases.",
    )
    args = parser.parse_args()

    corpus_store = LocalEvalCorpusStore()
    service = LocalEvalService(corpus_store=corpus_store)
    chunk_config = BenchmarkChunkConfig(
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
        max_split_char_number=int(args.max_split_char_number),
    )
    datasets = {item.dataset_key: item for item in corpus_store.list_available_datasets()}
    dataset = datasets["local_sample_complex_20"] if args.dataset == "complex" else datasets["local_sample_220"]

    selected_kb = None
    if args.knowledge_base_id:
        selected_kb = corpus_store.get_knowledge_base(args.knowledge_base_id)
        if not selected_kb:
            raise SystemExit(f"未找到知识库：{args.knowledge_base_id}")
    else:
        kb_name = args.knowledge_base_name or corpus_store.suggest_knowledge_base_name(f"sampledocs_{args.dataset}")
        selected_kb = corpus_store.build_knowledge_base(
            knowledge_base_name=kb_name,
            chunk_config=chunk_config,
            rebuild=args.rebuild,
        )

    result = service.run_evaluation(
        LocalEvalConfig(
            knowledge_base_id=selected_kb["knowledge_base_id"],
            knowledge_base_name=selected_kb["knowledge_base_name"],
            dataset_key=dataset.dataset_key,
            dataset_label=dataset.dataset_label,
            sample_path=dataset.sample_path,
            retriever_strategy=args.retriever_strategy,
            top_k=config.benchmark_default_top_k,
            question_limit=args.limit,
            selected_question_ids=[],
            enable_query_rewrite=True,
            enable_rerank=True,
            enable_ragas=not args.disable_ragas,
            chunk_config=BenchmarkChunkConfig.from_dict(selected_kb.get("chunk_config")),
        )
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.BENCHMARK_RUN_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"local_sample_eval_{timestamp}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("question_count:", result["question_count"])
    print("retrieval_metrics:", json.dumps(result["retrieval_metrics"], ensure_ascii=False))
    print("rerank_metrics:", json.dumps(result["rerank_metrics"], ensure_ascii=False))
    print("ragas_metrics:", json.dumps(result["ragas_metrics"], ensure_ascii=False))
    print("saved_to:", output_path)


if __name__ == "__main__":
    main()
