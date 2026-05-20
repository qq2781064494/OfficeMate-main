"""本地题库主 RAG 测评页。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config_data as config
from services.benchmark_results import BenchmarkResultStore
from services.benchmark_store import BenchmarkChunkConfig
from services.local_eval_service import LocalEvalConfig, LocalEvalService
from services.local_eval_store import LocalEvalCorpusStore


st.set_page_config(page_title="OfficeMate - 本地题库测评", layout="wide")


def main() -> None:
    if "local_eval_selected_kb_id" not in st.session_state:
        st.session_state["local_eval_selected_kb_id"] = ""
    corpus_store = LocalEvalCorpusStore()
    result_store = BenchmarkResultStore()
    eval_service = LocalEvalService(corpus_store=corpus_store, result_store=result_store)

    st.title("本地题库测评")
    st.caption("本页面使用 `sample_docs/*.txt` 构建独立知识库实例，和 app 正常问答页面数据库完全隔离。")
    st.caption("切片策略变化会生成新的知识库实例；只有你明确选择重建时，才会覆盖同名知识库。")
    st.caption("答案生成复用 app 正常知识对话逻辑与提示词，不使用 benchmark 专用答题提示词。")
    st.caption(
        "当前答题链路模型："
        f"主回答 `{config.chat_model_name}`，"
        f"改写 `{config.rewrite_model_name}`，"
        f"任务规划 `{config.task_model_name}`，"
        f"Rerank `{config.rerank_model_name}`。"
    )
    st.caption(
        "当前生成评测模型："
        f"裁判 `{config.benchmark_chat_model_name}`，"
        f"Embedding `{config.benchmark_embedding_model_name}`。"
    )

    available_datasets = corpus_store.list_available_datasets()
    dataset_options = [item.dataset_label for item in available_datasets]
    selected_dataset_label = st.selectbox("评测题集", options=dataset_options, index=0)
    selected_dataset = next(item for item in available_datasets if item.dataset_label == selected_dataset_label)
    samples = corpus_store.load_eval_samples(selected_dataset.sample_path)

    st.subheader("知识库选择")
    kb_mode = st.radio("知识库模式", options=["使用已有知识库", "新建知识库"], horizontal=True)
    available_kbs = corpus_store.list_knowledge_bases()
    selected_kb = None
    pending_kb_name = ""
    pending_chunk_config = BenchmarkChunkConfig()
    rebuild_knowledge_base = False

    if kb_mode == "使用已有知识库":
        if available_kbs:
            kb_options = [f"{item['knowledge_base_name']} ({item['knowledge_base_id']})" for item in available_kbs]
            default_index = 0
            selected_kb_id = st.session_state.get("local_eval_selected_kb_id", "")
            if selected_kb_id:
                for index, item in enumerate(available_kbs):
                    if item["knowledge_base_id"] == selected_kb_id:
                        default_index = index
                        break
            selected_kb_label = st.selectbox("已有知识库", options=kb_options, index=default_index)
            selected_kb = next(
                item for item in available_kbs if f"{item['knowledge_base_name']} ({item['knowledge_base_id']})" == selected_kb_label
            )
            st.session_state["local_eval_selected_kb_id"] = selected_kb["knowledge_base_id"]
            _render_knowledge_base_card(selected_kb)
        else:
            st.warning("当前还没有任何本地题库知识库，请先切换到“新建知识库”。")
    else:
        default_name = corpus_store.suggest_knowledge_base_name("sampledocs_balanced")
        pending_kb_name = st.text_input("知识库名称", value=default_name)
        chunk_mode = st.selectbox("切片配置模式", options=["预设", "高级参数"], index=0)
        if chunk_mode == "预设":
            chunk_preset = st.selectbox("切片规则", options=["平衡切片", "整篇优先", "细粒度切片"], index=0)
            pending_chunk_config = _resolve_chunk_config(chunk_preset)
        else:
            advanced_chunk_size = st.number_input("chunk_size", min_value=100, max_value=5000, value=config.benchmark_chunk_size, step=50)
            advanced_chunk_overlap = st.number_input("chunk_overlap", min_value=0, max_value=2000, value=config.benchmark_chunk_overlap, step=20)
            advanced_max_split = st.number_input(
                "max_split_char_number",
                min_value=100,
                max_value=10000,
                value=config.benchmark_max_split_char_number,
                step=100,
            )
            pending_chunk_config = BenchmarkChunkConfig(
                chunk_size=int(advanced_chunk_size),
                chunk_overlap=int(advanced_chunk_overlap),
                max_split_char_number=int(advanced_max_split),
            )
        st.caption(
            f"待构建切片参数：chunk_size={pending_chunk_config.chunk_size}，"
            f"chunk_overlap={pending_chunk_config.chunk_overlap}，"
            f"max_split_char_number={pending_chunk_config.max_split_char_number}"
        )
        rebuild_knowledge_base = st.checkbox("重建同名知识库", value=False)
        build_only = st.button("构建知识库", use_container_width=True)
        if build_only:
            status_placeholder = st.empty()
            try:
                status_placeholder.info("正在构建本地题库独立知识库...")
                selected_kb = corpus_store.build_knowledge_base(
                    knowledge_base_name=pending_kb_name,
                    chunk_config=pending_chunk_config,
                    rebuild=rebuild_knowledge_base,
                )
                st.session_state["local_eval_selected_kb_id"] = selected_kb["knowledge_base_id"]
                status_placeholder.success(
                    f"知识库构建完成：{selected_kb['knowledge_base_name']}，"
                    f"文档 {selected_kb['document_count']} 篇，片段 {selected_kb['chunk_count']} 条。"
                )
                _render_knowledge_base_card(selected_kb)
            except Exception as exc:
                status_placeholder.error(f"构建知识库失败：{exc}")
        elif st.session_state.get("local_eval_selected_kb_id"):
            selected_kb = corpus_store.get_knowledge_base(st.session_state["local_eval_selected_kb_id"])
            if selected_kb:
                st.caption("当前会话中已选中的知识库：")
                _render_knowledge_base_card(selected_kb)

    st.divider()
    st.subheader("测评配置")
    question_picker_mode = st.radio("题目选择", options=["当前题集全部题目", "手动选择题目"], horizontal=True)
    selected_question_ids: list[int] = []
    if question_picker_mode == "手动选择题目":
        picker_options = [f"{index}. {sample.get('query', '')[:80]}" for index, sample in enumerate(samples, start=1)]
        selected_rows = st.multiselect("选择要评测的题目", options=picker_options, default=picker_options[: min(5, len(picker_options))])
        selected_question_ids = [int(option.split(".", 1)[0]) for option in selected_rows]
        if not selected_question_ids:
            st.warning("当前处于手动选择题目模式，请至少选择 1 道题。")

    retriever_strategy = st.selectbox("检索策略", options=["hybrid", "vector", "bm25"], index=0)
    enable_query_rewrite = st.checkbox("启用 Query Rewrite", value=True)
    enable_rerank = st.checkbox("启用 Rerank", value=True)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=config.benchmark_default_top_k)
    enable_ragas = st.checkbox("启用 Ragas 生成评测", value=True)
    enable_faithfulness = st.checkbox("启用 Faithfulness", value=True, disabled=not enable_ragas)
    max_question_count = len(selected_question_ids) if selected_question_ids else len(samples)
    question_limit = st.number_input("本次题量限制", min_value=1, max_value=max_question_count, value=max_question_count, step=1)
    submit_disabled = selected_kb is None or (question_picker_mode == "手动选择题目" and not selected_question_ids)
    if submit_disabled and kb_mode == "使用已有知识库" and not available_kbs:
        st.info("请先新建至少一个本地题库知识库。")

    _render_dataset_overview(selected_dataset, selected_question_ids, selected_kb)

    if st.button("开始测评", use_container_width=True, disabled=submit_disabled):
        status_placeholder = st.empty()

        def update_status(message: str) -> None:
            status_placeholder.info(message)

        try:
            result = eval_service.run_evaluation(
                LocalEvalConfig(
                    knowledge_base_id=selected_kb["knowledge_base_id"],
                    knowledge_base_name=selected_kb["knowledge_base_name"],
                    dataset_key=selected_dataset.dataset_key,
                    dataset_label=selected_dataset.dataset_label,
                    sample_path=selected_dataset.sample_path,
                    retriever_strategy=retriever_strategy,
                    top_k=int(top_k),
                    question_limit=int(question_limit),
                    selected_question_ids=selected_question_ids,
                    enable_query_rewrite=enable_query_rewrite,
                    enable_ragas=enable_ragas,
                    enable_faithfulness=enable_faithfulness,
                    enable_rerank=enable_rerank,
                    chunk_config=BenchmarkChunkConfig.from_dict(selected_kb.get("chunk_config")),
                ),
                status_callback=update_status,
            )
            status_placeholder.success(f"评测完成，run_id = {result['run_id']}")
            _render_run_summary(result_store, result["run_id"])
        except Exception as exc:
            status_placeholder.error(f"评测失败：{exc}")

    st.divider()
    st.subheader("最近 Run")
    recent_runs = [item for item in result_store.list_runs(limit=50, subset="local_eval_kb") if item.get("mode") == "local_eval_rag"]
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
            "knowledge_base_name",
            "dataset_label",
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
        st.info("当前还没有任何本地题库 run。")


def _render_knowledge_base_card(kb_info: dict) -> None:
    st.caption(
        f"当前知识库：`{kb_info['knowledge_base_name']}` | "
        f"文档数 {kb_info['document_count']} | "
        f"片段数 {kb_info['chunk_count']} | "
        f"更新时间 {kb_info['updated_at']}"
    )
    st.caption(f"切片参数：{kb_info.get('chunk_config', {})}")


def _render_dataset_overview(selected_dataset, selected_question_ids: list[int], selected_kb: dict | None) -> None:
    st.subheader("当前概览")
    metric_columns = st.columns(4)
    metric_columns[0].metric("当前题集", selected_dataset.dataset_label)
    metric_columns[1].metric("题集题数", selected_dataset.question_count)
    metric_columns[2].metric("已选题目数", len(selected_question_ids) if selected_question_ids else selected_dataset.question_count)
    metric_columns[3].metric("当前知识库文档数", selected_kb["document_count"] if selected_kb else 0)


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
    st.caption(f"Context Recall: {_format_metric(ragas_metrics.get('context_recall'))}")

    if ragas_metrics.get("status") and ragas_metrics.get("status") != "success":
        if ragas_metrics.get("status") == "success_with_warnings":
            st.warning("Ragas 已完成，但部分子任务失败或超时。")
        else:
            st.warning(f"Ragas 当前状态：{ragas_metrics.get('status')}，详情：{ragas_metrics.get('error', '无')}")

    job_errors = ragas_metrics.get("job_errors", [])
    timeout_errors = [item for item in job_errors if item.get("is_timeout")]
    if timeout_errors:
        timeout_lines = [
            f"第 {item.get('question_id', item.get('question_row_index', '?'))} 题 / {item.get('metric', 'unknown')}"
            for item in timeout_errors
        ]
        st.warning("Ragas 子任务超时：" + "；".join(timeout_lines))
    non_timeout_errors = [item for item in job_errors if not item.get("is_timeout")]
    if non_timeout_errors:
        error_lines = [
            f"第 {item.get('question_id', item.get('question_row_index', '?'))} 题 / {item.get('metric', 'unknown')} / {item.get('exception_type', 'Error')}"
            for item in non_timeout_errors
        ]
        st.warning("Ragas 子任务异常：" + "；".join(error_lines))

    st.caption(
        f"知识库={summary.get('knowledge_base_name')} | "
        f"题集={summary.get('dataset_label')} | "
        f"retriever={summary.get('retriever_strategy')} | "
        f"chunk_config={summary.get('chunk_config', {})}"
    )
    if summary.get("selected_question_ids"):
        st.caption(f"选中题号：{summary.get('selected_question_ids')}")

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
        if "retrieval_queries" in detail_frame.columns:
            detail_frame["retrieval_queries_joined"] = detail_frame["retrieval_queries"].apply(lambda items: " | ".join(items))
        detail_frame["retrieved_titles_joined"] = detail_frame["retrieved_titles"].apply(lambda items: " | ".join(items))
        visible_columns = ["question_id", "question", "expected_titles", "retrieval_queries_joined"]
        if "effective_category" in detail_frame.columns:
            visible_columns.append("effective_category")
        if summary.get("enable_rerank"):
            visible_columns.extend(["pre_rerank_titles_joined", "pre_rerank_hit", "pre_rerank_first_hit_rank"])
        visible_columns.extend(["retrieved_titles_joined", "retrieval_hit", "first_hit_rank", "gold_answer", "predicted_answer"])
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
