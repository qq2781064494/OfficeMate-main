"""RAGBench 全局知识库测评页。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config_data as config
from services.benchmark_store import BenchmarkChunkConfig
from services.benchmark_eval_service import BenchmarkEvalConfig, BenchmarkEvalService
from services.benchmark_results import BenchmarkResultStore
from services.benchmark_store import BenchmarkCorpusStore


st.set_page_config(page_title="OfficeMate - RAGBench 测评", layout="wide")


def main() -> None:
    corpus_store = BenchmarkCorpusStore()
    result_store = BenchmarkResultStore()
    eval_service = BenchmarkEvalService(corpus_store=corpus_store, result_store=result_store)

    st.title("RAGBench 测评")
    st.caption("使用全局知识库模式对 RAGBench subset 做检索与生成评测，并保存每次 run 的问题、答案和分数。")
    st.caption(
        "当前评测模型："
        f"生成/裁判 `{config.benchmark_chat_model_name}`，"
        f"Embedding `{config.benchmark_embedding_model_name}`，"
        f"Rerank `{config.benchmark_rerank_model_name}`。"
    )

    available_infos = corpus_store.list_available_subsets()
    if not available_infos:
        st.warning("当前没有检测到已下载的 RAGBench 子集，请先准备 `storage/ragbench/<subset>/...` 数据。")
        return

    subset_options = [item.subset for item in available_infos]
    subset_default_index = subset_options.index(config.benchmark_default_subsets[0]) if config.benchmark_default_subsets[0] in subset_options else 0

    subset = st.selectbox("数据子集", options=subset_options, index=subset_default_index)
    selected_info = next(item for item in available_infos if item.subset == subset)
    split = st.selectbox(
        "数据切分",
        options=selected_info.available_splits or config.benchmark_default_splits,
        index=0,
    )
    retriever_strategy = st.selectbox("检索策略", options=["hybrid", "vector", "bm25"], index=0)
    enable_query_rewrite = st.checkbox("启用 Query Rewrite", value=True)
    enable_rerank = st.checkbox("启用 Rerank", value=True)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=config.benchmark_default_top_k)
    question_limit = st.number_input(
        "本次题量限制",
        min_value=1,
        max_value=max(1, selected_info.question_count or 1),
        value=min(config.benchmark_default_question_limit, max(1, selected_info.question_count or 1)),
        step=1,
    )
    chunk_mode = st.selectbox(
        "切片配置模式",
        options=["预设", "高级参数"],
        index=0,
    )
    if chunk_mode == "预设":
        chunk_preset = st.selectbox(
            "切片规则",
            options=["平衡切片", "整篇优先", "细粒度切片"],
            index=0,
        )
        chunk_config = _resolve_chunk_config(chunk_preset)
    else:
        advanced_chunk_size = st.number_input(
            "chunk_size",
            min_value=100,
            max_value=5000,
            value=config.benchmark_chunk_size,
            step=50,
        )
        advanced_chunk_overlap = st.number_input(
            "chunk_overlap",
            min_value=0,
            max_value=2000,
            value=config.benchmark_chunk_overlap,
            step=20,
        )
        advanced_max_split = st.number_input(
            "max_split_char_number",
            min_value=100,
            max_value=10000,
            value=config.benchmark_max_split_char_number,
            step=100,
        )
        chunk_config = BenchmarkChunkConfig(
            chunk_size=int(advanced_chunk_size),
            chunk_overlap=int(advanced_chunk_overlap),
            max_split_char_number=int(advanced_max_split),
    )
    st.caption(
        f"当前切片参数：chunk_size={chunk_config.chunk_size}，"
        f"chunk_overlap={chunk_config.chunk_overlap}，"
        f"max_split_char_number={chunk_config.max_split_char_number}"
    )
    enable_ragas = st.checkbox("启用 Ragas 生成评测", value=True)
    enable_faithfulness = st.checkbox("启用 Faithfulness", value=True, disabled=not enable_ragas)
    rebuild_corpus = st.checkbox("重建全局语料", value=False)
    rebuild_index = st.checkbox("重建向量索引", value=False)
    action_columns = st.columns(2)
    build_only = action_columns[0].button("只构建知识库", use_container_width=True)
    submitted = action_columns[1].button("开始测评", use_container_width=True)

    st.caption("如果你首次构建某个 subset，建议勾选“重建向量索引”；遇到只读数据库错误时，系统现在也会自动重试一次。")

    _render_subset_overview(corpus_store, subset)

    if build_only:
        status_placeholder = st.empty()
        try:
            status_placeholder.info("正在构建 benchmark 全局语料...")
            corpus_summary = corpus_store.build_subset_corpus(
                subset=subset,
                splits=[split],
                rebuild=rebuild_corpus,
            )
            status_placeholder.info("正在构建 benchmark 向量索引...")
            index_summary = corpus_store.ensure_vector_index(
                subset=subset,
                rebuild=rebuild_index,
                chunk_config=chunk_config,
            )
            status_placeholder.success(
                f"知识库已准备完成：文档 {corpus_summary['document_count']} 篇，"
                f"向量片段 {index_summary.get('chunk_count', 0)} 条。"
            )
            _render_subset_overview(corpus_store, subset)
        except Exception as exc:
            status_placeholder.error(f"构建知识库失败：{exc}")

    if submitted:
        status_placeholder = st.empty()

        def update_status(message: str) -> None:
            status_placeholder.info(message)

        try:
            result = eval_service.run_evaluation(
                BenchmarkEvalConfig(
                    subset=subset,
                    split=split,
                    retriever_strategy=retriever_strategy,
                    top_k=int(top_k),
                    question_limit=int(question_limit),
                    enable_query_rewrite=enable_query_rewrite,
                    enable_ragas=enable_ragas,
                    enable_faithfulness=enable_faithfulness,
                    enable_rerank=enable_rerank,
                    rebuild_corpus=rebuild_corpus,
                    rebuild_index=rebuild_index,
                    chunk_config=chunk_config,
                ),
                status_callback=update_status,
            )
            status_placeholder.success(f"评测完成，run_id = {result['run_id']}")
            _render_run_summary(result_store, result["run_id"])
        except Exception as exc:
            status_placeholder.error(f"评测失败：{exc}")

    st.divider()
    st.subheader("最近 Run")
    recent_runs = result_store.list_runs(limit=20, subset=subset)
    if recent_runs:
        frame = pd.DataFrame(recent_runs)
        if "enable_query_rewrite" not in frame.columns:
            frame["enable_query_rewrite"] = True
        if "enable_rerank" not in frame.columns:
            frame["enable_rerank"] = True
        if "enable_faithfulness" not in frame.columns:
            frame["enable_faithfulness"] = True
        visible_columns = [
            "run_id",
            "created_at",
            "subset",
            "split",
            "retriever_strategy",
            "enable_query_rewrite",
            "enable_faithfulness",
            "question_count",
            "top_k",
        ]
        st.dataframe(frame[visible_columns], use_container_width=True, hide_index=True)
        selected_run_id = st.selectbox("查看历史 run 明细", options=[record["run_id"] for record in recent_runs])
        _render_run_summary(result_store, selected_run_id)
    else:
        st.info("当前还没有任何 benchmark run。")


def _render_subset_overview(corpus_store: BenchmarkCorpusStore, subset: str) -> None:
    st.subheader("Subset 概览")
    splits = []
    question_count = 0
    for info in corpus_store.list_available_subsets():
        if info.subset == subset:
            splits = info.available_splits
            question_count = info.question_count
            break

    corpus_manifest = corpus_store.load_corpus_manifest(subset)
    metric_columns = st.columns(3)
    metric_columns[0].metric("可用切分", ", ".join(splits) if splits else "无")
    metric_columns[1].metric("评测题数", question_count)
    metric_columns[2].metric("全局文档数", len(corpus_manifest))


def _render_run_summary(result_store: BenchmarkResultStore, run_id: str) -> None:
    summary = result_store.get_run_summary(run_id)
    details = result_store.load_run_details(run_id)
    if not summary:
        return

    st.divider()
    st.subheader("Run 结果")

    retrieval_metrics = summary.get("retrieval_metrics", {})
    rerank_metrics = summary.get("rerank_metrics", {})
    ragas_metrics = summary.get("ragas_metrics", {})

    top_columns = st.columns(4)
    top_columns[0].metric("题目数", summary.get("question_count", 0))
    top_columns[1].metric("文档数", summary.get("document_count", 0))
    top_columns[2].metric("Recall@5", _format_metric(retrieval_metrics.get("recall_at_5")))
    top_columns[3].metric("MRR", _format_metric(retrieval_metrics.get("mrr")))
    st.caption(
        f"本次配置：Query Rewrite={'开' if summary.get('enable_query_rewrite', True) else '关'}，"
        f"Rerank={'开' if summary.get('enable_rerank', True) else '关'}，"
        f"Faithfulness={'开' if summary.get('enable_faithfulness', True) else '关'}。"
    )

    second_columns = st.columns(4)
    second_columns[0].metric("HitRate@5", _format_metric(retrieval_metrics.get("hit_rate_at_5")))
    second_columns[1].metric("Faithfulness", _format_metric(ragas_metrics.get("faithfulness")))
    second_columns[2].metric("Answer Relevancy", _format_metric(ragas_metrics.get("answer_relevancy")))
    second_columns[3].metric("Context Precision", _format_metric(ragas_metrics.get("context_precision")))

    if ragas_metrics.get("status") and ragas_metrics.get("status") != "success":
        st.warning(f"Ragas 当前状态：{ragas_metrics.get('status')}，详情：{ragas_metrics.get('error', '无')}")

    st.caption(
        f"retriever={summary.get('retriever_strategy')} | "
        f"rerank={summary.get('enable_rerank')} | "
        f"chunk_config={summary.get('chunk_config', {})}"
    )

    if summary.get("enable_rerank") and rerank_metrics.get("status") == "success":
        st.write("**Rerank 指标**")
        rerank_top = st.columns(4)
        rerank_top[0].metric("Pre MRR", _format_metric(rerank_metrics.get("pre_mrr")))
        rerank_top[1].metric("Post MRR", _format_metric(rerank_metrics.get("post_mrr")))
        rerank_top[2].metric("Delta MRR", _format_metric(rerank_metrics.get("delta_mrr")))
        rerank_top[3].metric("Avg Rank Improvement", _format_metric(rerank_metrics.get("avg_rank_improvement")))

        rerank_second = st.columns(4)
        rerank_second[0].metric("Pre HitRate@1", _format_metric(rerank_metrics.get("pre_hit_rate_at_1")))
        rerank_second[1].metric("Post HitRate@1", _format_metric(rerank_metrics.get("post_hit_rate_at_1")))
        rerank_second[2].metric("Win Rate", _format_metric(rerank_metrics.get("win_rate")))
        rerank_second[3].metric("Lose Rate", _format_metric(rerank_metrics.get("lose_rate")))

    if details:
        detail_frame = pd.DataFrame(details)
        if "pre_rerank_titles" in detail_frame.columns:
            detail_frame["pre_rerank_titles_joined"] = detail_frame["pre_rerank_titles"].apply(lambda items: " | ".join(items))
        detail_frame["retrieved_titles_joined"] = detail_frame["retrieved_titles"].apply(lambda items: " | ".join(items))
        visible_columns = ["question_id", "question", "expected_titles"]
        if summary.get("enable_rerank"):
            visible_columns.extend(
                [
                    "pre_rerank_titles_joined",
                    "pre_rerank_hit",
                    "pre_rerank_first_hit_rank",
                ]
            )
        visible_columns.extend(
            [
                "retrieved_titles_joined",
                "retrieval_hit",
                "first_hit_rank",
                "gold_answer",
                "predicted_answer",
            ]
        )
        st.dataframe(detail_frame[visible_columns], use_container_width=True, hide_index=True)


def _format_metric(value) -> str:
    if value is None:
        return "暂无"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def _resolve_chunk_config(preset: str) -> BenchmarkChunkConfig:
    if preset == "整篇优先":
        return BenchmarkChunkConfig(chunk_size=1400, chunk_overlap=80, max_split_char_number=1800)
    if preset == "细粒度切片":
        return BenchmarkChunkConfig(chunk_size=600, chunk_overlap=120, max_split_char_number=700)
    return BenchmarkChunkConfig(
        chunk_size=config.benchmark_chunk_size,
        chunk_overlap=config.benchmark_chunk_overlap,
        max_split_char_number=config.benchmark_max_split_char_number,
    )


main()
