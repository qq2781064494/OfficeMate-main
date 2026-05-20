"""页面层。

这个文件把 Streamlit 的 3 个页面都集中在一起：
- render_chat_page: 智能问答页
- render_upload_page: 知识上传页
- render_management_page: 知识管理页

阅读时可以把它理解成“页面控制器”：
接收用户交互，然后调用 service 层完成真正业务。
"""

import json
import os
from collections import Counter
from uuid import uuid4

import pandas as pd
import streamlit as st

import config_data as config
from services.chat_service import OfficeMateChatService
from services.document_service import DocumentService
from services.storage_service import JsonStorageService
from services.upload_task_manager import upload_task_manager
from utils.log_tool import get_logger


logger = get_logger("ui_pages")


def _stream_text_chunks(text, chunk_size=48):
    """把完整字符串切成小块，交给 Streamlit 做流式渲染。

    注意：
    - Streamlit 的 `st.write_stream` 在消费生成器后会把完整文本再返回一次
    - 所以我们可以一边做流式显示，一边把返回结果保存到 session_state
    """
    for index in range(0, len(text), chunk_size):
        yield text[index:index + chunk_size]


def render_chat_page():
    """渲染主聊天页。"""
    storage = JsonStorageService()
    # 先做页面启动前的准备：提示 API Key、初始化会话、绘制侧边栏。
    _render_api_key_notice()
    _init_chat_session(storage)
    _render_chat_sidebar(storage)

    st.title("OfficeMate：企业内部制度与流程智能助手")
    st.caption("面向企业内部制度、流程、通知与常见 IT 支持问题的轻量级知识助手。")

    if not storage.list_documents():
        st.info("当前知识库为空，请先前往“知识上传”页导入文档或示例知识库。")

    # 先把当前会话里的所有历史消息重新渲染出来。
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("qa_log_id"):
                _render_feedback_form(storage, message)

    # chat_input 是 Streamlit 的聊天输入框；只有用户真正提交内容时 prompt 才非空。
    prompt = st.chat_input("例如：报销差旅费需要提交哪些材料？")
    if prompt:
        question_category = st.session_state.get("selected_category", "全部")
        logger.info(
            "chat_page received_question | session_id=%s | category=%s | question=%s",
            st.session_state["session_id"],
            question_category,
            prompt,
        )
        # 先把用户消息写进前端会话状态，确保界面立即出现这句话。
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                status_placeholder = st.empty()

                def update_status(message):
                    status_placeholder.markdown(f"_{message}_")

                # 真正的问答工作交给 ChatService，它会完成检索、生成、流式输出和日志记录。
                chat_service = OfficeMateChatService()
                answer_stream, result_holder = chat_service.stream_answer_question(
                    question=prompt,
                    session_id=st.session_state["session_id"],
                    category=question_category,
                    status_callback=update_status,
                )
                first_chunk = {"seen": False}

                def ui_stream():
                    for chunk in answer_stream:
                        if chunk and not first_chunk["seen"]:
                            first_chunk["seen"] = True
                            status_placeholder.empty()
                        yield chunk

                streamed_answer = st.write_stream(ui_stream())
                status_placeholder.empty()
                assistant_message = {
                    "role": "assistant",
                    "content": streamed_answer,
                    "qa_log_id": result_holder.get("qa_log_id"),
                    "question_type": result_holder.get("question_type", ""),
                    "question": prompt,
                }
                _render_feedback_form(storage, assistant_message)
                st.session_state["chat_messages"].append(assistant_message)
                logger.info(
                    "chat_page answer_rendered | session_id=%s | qa_log_id=%s | question_type=%s",
                    st.session_state["session_id"],
                    assistant_message["qa_log_id"],
                    assistant_message["question_type"],
                )
            except Exception as exc:
                logger.exception(
                    "chat_page answer_failed | session_id=%s | category=%s | question=%s | error=%s",
                    st.session_state["session_id"],
                    question_category,
                    prompt,
                    exc,
                )
                # 页面层只负责兜底展示错误，不在这里深入处理底层异常。
                error_message = (
                    "### 最终回答\n当前回答失败，可能是模型 Key 未配置或知识库还没有完成初始化。\n\n"
                    "### 操作步骤/材料清单\n请检查 embedding/chat 模型配置、知识库文档和网络连接。\n\n"
                    f"### 风险提示\n原始错误：{exc}"
                )
                st.error(error_message)


def render_upload_page():
    """渲染知识上传页。"""
    service = DocumentService()
    _render_api_key_notice()

    st.title("知识上传")
    st.caption("上传企业内部制度、流程、通知与 FAQ 文档，并补充分类、标题和版本信息。")
    st.info("支持一次多选多个文件，也支持直接上传一个 zip 压缩包批量导入。")

    with st.form("upload_form"):
        # 表单模式下，只有点击提交按钮后，这些输入才会统一被处理。
        category = st.selectbox("文档分类", config.DOCUMENT_CATEGORIES)
        version = st.text_input("文档版本", value=config.DEFAULT_VERSION)
        custom_title = st.text_input("自定义标题（单文件上传时可选）")
        uploaded_files = st.file_uploader(
            "上传文档",
            type=config.SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
        )
        submit_upload = st.form_submit_button("导入知识库")

    if submit_upload:
        if not uploaded_files:
            st.warning("请先选择至少一个文件。")
        else:
            task_id = upload_task_manager.submit_task(
                uploaded_files=uploaded_files,
                category=category,
                version=version,
                custom_title=custom_title if len(uploaded_files) == 1 else "",
            )
            st.session_state.setdefault("upload_task_ids", [])
            st.session_state["upload_task_ids"] = (
                [task_id] + st.session_state["upload_task_ids"]
            )[:config.upload_task_history_limit]
            st.success("上传任务已进入后台队列，下面可以查看进度。")

    _render_upload_task_panel()

    st.divider()
    st.subheader("示例知识库")
    st.write("如果你暂时没有企业制度文档，可以先导入项目自带的示例文档进行演示。")
    if st.button("一键导入示例文档", key="seed_docs"):
        try:
            logger.info("upload_page seed_sample_documents start")
            # 内置样例走的其实也是同一套 ingest_bytes 入库链路。
            for result in service.seed_sample_documents():
                _show_upload_result(result)
        except Exception as exc:
            logger.exception("upload_page seed_sample_documents failed | error=%s", exc)
            st.error(f"示例文档导入失败：{exc}")

    st.divider()
    st.subheader("最近导入文档")
    recent_docs = service.list_documents()[:10]
    if recent_docs:
        dataframe = pd.DataFrame(recent_docs)[
            ["title", "category", "version", "file_type", "chunk_count", "uploaded_at", "status"]
        ]
        st.dataframe(dataframe, use_container_width=True, hide_index=True)
    else:
        st.info("还没有导入任何文档。")


def render_management_page():
    """渲染知识管理页。"""
    storage = JsonStorageService()
    document_service = DocumentService()
    stats = storage.get_stats()
    _render_api_key_notice()

    st.title("知识管理")
    st.caption("查看文档状态、问答记录和用户反馈，便于演示知识库闭环。")

    metric_columns = st.columns(4)
    metric_columns[0].metric("文档数量", stats["document_count"])
    metric_columns[1].metric("覆盖分类", stats["category_count"])
    metric_columns[2].metric("问答记录", stats["qa_count"])
    metric_columns[3].metric("反馈数量", stats["feedback_count"])

    _render_quantification_section(storage, stats)

    st.divider()
    st.subheader("文档列表")
    documents = storage.list_documents()
    if documents:
        document_frame = pd.DataFrame(documents)[
            ["title", "category", "version", "file_name", "chunk_count", "uploaded_at", "status", "source_label"]
        ]
        st.dataframe(document_frame, use_container_width=True, hide_index=True)
    else:
        st.info("暂无文档记录。")

    st.subheader("删除已上传知识")
    if documents:
        # 把文档列表转成“展示标签 -> 实际 document_id”的映射，方便下拉框选择。
        document_options = {
            _build_document_option_label(document): document["id"]
            for document in documents
        }
        selected_label = st.selectbox(
            "选择需要删除的文档",
            options=list(document_options.keys()),
            key="delete_document_selector",
        )
        st.caption("删除后会同步移除原始文件和向量索引；历史问答与反馈记录会保留，便于继续演示使用痕迹。")
        confirm_delete = st.checkbox("我确认删除这份知识文档", key="confirm_delete_document")
        if st.button("删除选中文档", type="primary", key="delete_document_button"):
            if not confirm_delete:
                st.warning("请先勾选确认项，再执行删除。")
            else:
                # 删除动作在 service 层完成，这里只负责调用和反馈结果。
                result = document_service.delete_document(document_options[selected_label])
                if result["status"] == "success":
                    logger.info("management_page delete_document success | label=%s", selected_label)
                    st.success(result["message"])
                    st.rerun()
                elif result["status"] == "not_found":
                    logger.warning("management_page delete_document not_found | label=%s", selected_label)
                    st.warning(result["message"])
                else:
                    logger.error("management_page delete_document failed | label=%s | message=%s", selected_label, result["message"])
                    st.error(result["message"])
    else:
        st.info("当前没有可删除的文档。")

    st.divider()
    st.subheader("最近问答")
    qa_logs = storage.list_qa_logs(limit=20)
    if qa_logs:
        qa_frame = pd.DataFrame(qa_logs)
        qa_frame["source_count"] = qa_frame["source_docs"].apply(len)
        qa_frame = qa_frame[
            ["created_at", "question_type", "category", "question", "source_count", "session_id"]
        ]
        st.dataframe(qa_frame, use_container_width=True, hide_index=True)
    else:
        st.info("暂无问答记录。")

    st.divider()
    st.subheader("用户反馈")
    feedback_logs = storage.list_feedback()
    if feedback_logs:
        # 反馈表中需要展示“对应问题是什么”，所以先把 qa_log 建一个索引表。
        qa_lookup = {record["id"]: record for record in storage.list_qa_logs()}
        feedback_rows = []
        for feedback in feedback_logs:
            qa_log = qa_lookup.get(feedback["qa_log_id"], {})
            feedback_rows.append(
                {
                    "created_at": feedback.get("updated_at", feedback.get("created_at", "")),
                    "rating": _rating_to_label(feedback.get("rating", "")),
                    "comment": feedback.get("comment", ""),
                    "question": qa_log.get("question", ""),
                    "session_id": feedback.get("session_id", ""),
                }
            )
        st.dataframe(pd.DataFrame(feedback_rows), use_container_width=True, hide_index=True)
    else:
        st.info("暂无反馈记录。")


def _render_api_key_notice():
    """在页面顶部提示是否缺少模型配置。"""
    if not config.embedding_base_url:
        logger.warning("api_key_notice missing_openai_compatible_embedding_base_url")
        st.warning("当前未检测到本地 embedding 服务地址。页面可以打开，但向量化会失败。")


def _render_quantification_section(storage, stats):
    """渲染项目量化结果，便于答辩、汇报和截图展示。"""
    qa_logs = storage.list_qa_logs()
    feedback_logs = storage.list_feedback()
    sample_summary = _load_evaluation_sample_summary()
    usage_metrics = _build_usage_metrics(qa_logs, feedback_logs)
    evaluation_metrics = _get_retrieval_evaluation_metrics()

    st.divider()
    st.subheader("效果量化")
    st.caption("把文档规模、业务覆盖、使用闭环和离线检索效果放到同一页展示，避免只凭主观感觉介绍项目。")

    scale_columns = st.columns(4)
    scale_columns[0].metric("核心文档数", stats["document_count"])
    scale_columns[1].metric("业务类别数", stats["category_count"])
    scale_columns[2].metric("评测样本数", sample_summary["sample_count"])
    scale_columns[3].metric("有依据回答率", _format_ratio(usage_metrics["grounded_answer_rate"]))

    effect_columns = st.columns(4)
    effect_columns[0].metric("Recall@5", _format_ratio(evaluation_metrics.get("recall_at_k")))
    effect_columns[1].metric("Hit Rate", _format_ratio(evaluation_metrics.get("hit_rate")))
    effect_columns[2].metric("MRR", _format_ratio(evaluation_metrics.get("mrr")))
    effect_columns[3].metric("平均引用数", f'{usage_metrics["avg_source_count"]:.2f}')

    usage_columns = st.columns(3)
    usage_columns[0].metric("无依据拦截率", _format_ratio(usage_metrics["no_evidence_rate"]))
    usage_columns[1].metric("反馈采集率", _format_ratio(usage_metrics["feedback_capture_rate"]))
    usage_columns[2].metric("正向反馈占比", _format_ratio(usage_metrics["helpful_feedback_rate"]))

    st.caption(
        "说明：Recall@5 / Hit Rate / MRR 基于项目内置评测集；"
        "有依据回答率和反馈指标基于本地问答日志实时统计。"
    )

    coverage_rows = _build_coverage_rows(storage.list_documents(), sample_summary["category_counter"])
    if coverage_rows:
        st.write("**场景覆盖分布**")
        st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)

    if evaluation_metrics.get("error"):
        st.warning(f'离线检索评测暂时未完成：{evaluation_metrics["error"]}')

    with st.expander("报告/答辩可直接引用", expanded=False):
        st.code(
            _build_quantification_summary(stats, sample_summary, usage_metrics, evaluation_metrics),
            language="markdown",
        )


@st.cache_data(show_spinner=False)
def _load_evaluation_sample_summary():
    """读取评测集规模和分类分布。"""
    if not config.EVALUATION_SAMPLE_PATH.exists():
        return {"sample_count": 0, "category_counter": {}}

    samples = json.loads(config.EVALUATION_SAMPLE_PATH.read_text(encoding="utf-8"))
    category_counter = Counter(sample.get("category", "未分类") for sample in samples)
    return {
        "sample_count": len(samples),
        "category_counter": dict(category_counter),
    }


@st.cache_data(show_spinner=False)
def _get_retrieval_evaluation_metrics():
    """执行一次离线检索评测，并把结果缓存到当前页面会话。"""
    try:
        from services.evaluation_service import EvaluationService

        return EvaluationService().evaluate_recall(k=5)
    except Exception as exc:  # pragma: no cover - 页面展示兜底
        logger.exception("management_page evaluation_failed | error=%s", exc)
        return {
            "recall_at_k": None,
            "mrr": None,
            "hit_rate": None,
            "sample_count": 0,
            "error": str(exc),
        }


def _build_usage_metrics(qa_logs, feedback_logs):
    """根据已有日志统计使用效果。"""
    qa_count = len(qa_logs)
    feedback_count = len(feedback_logs)
    grounded_count = sum(1 for record in qa_logs if record.get("source_docs"))
    no_evidence_count = sum(
        1 for record in qa_logs if "未找到明确依据" in record.get("answer", "")
    )
    source_count_sum = sum(len(record.get("source_docs", [])) for record in qa_logs)
    helpful_count = sum(1 for record in feedback_logs if record.get("rating") == "helpful")

    return {
        "grounded_answer_rate": grounded_count / qa_count if qa_count else None,
        "no_evidence_rate": no_evidence_count / qa_count if qa_count else None,
        "feedback_capture_rate": feedback_count / qa_count if qa_count else None,
        "helpful_feedback_rate": helpful_count / feedback_count if feedback_count else None,
        "avg_source_count": source_count_sum / qa_count if qa_count else 0.0,
    }


def _build_coverage_rows(documents, sample_category_counter):
    """把文档覆盖和评测样本覆盖放在一张表里。"""
    document_counter = Counter(
        document.get("category", "未分类") for document in documents if document.get("category")
    )
    rows = []
    for category in config.DOCUMENT_CATEGORIES:
        rows.append(
            {
                "业务类别": category,
                "文档数": document_counter.get(category, 0),
                "评测题数": sample_category_counter.get(category, 0),
            }
        )
    return rows


def _build_quantification_summary(stats, sample_summary, usage_metrics, evaluation_metrics):
    """生成一段可直接复制到报告里的量化总结。"""
    return (
        f"本项目当前共接入 {stats['document_count']} 份核心制度文档，"
        f"覆盖 {stats['category_count']} 个业务类别，并构建 {sample_summary['sample_count']} 条标准评测问题。"
        f"在离线检索评测中，系统的 Recall@5 为 {_format_ratio(evaluation_metrics.get('recall_at_k'))}，"
        f"Hit Rate 为 {_format_ratio(evaluation_metrics.get('hit_rate'))}，"
        f"MRR 为 {_format_ratio(evaluation_metrics.get('mrr'))}。"
        f"结合本地问答日志统计，有依据回答率为 {_format_ratio(usage_metrics.get('grounded_answer_rate'))}，"
        f"无依据拦截率为 {_format_ratio(usage_metrics.get('no_evidence_rate'))}，"
        f"平均每次回答引用 {usage_metrics['avg_source_count']:.2f} 份来源文档。"
        "因此，该项目的量化重点不是文档总量，而是少量高价值制度文档能否稳定支持检索、生成和依据追溯。"
    )


def _format_ratio(value):
    """把 0~1 比例格式化成百分比文案。"""
    if value is None:
        return "暂无"
    return f"{value * 100:.1f}%"


def _init_chat_session(storage):
    """初始化前端会话状态。

    st.session_state 是 Streamlit 提供的会话级状态存储，用来记住：
    - 当前会话 ID
    - 当前选中的分类
    - 当前页面上已经展示过的聊天记录
    """
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"{config.default_session_prefix}_{uuid4().hex[:8]}"
        logger.info("chat_session initialized | session_id=%s", st.session_state["session_id"])

    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = "全部"

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = _load_messages_from_logs(
            storage,
            st.session_state["session_id"],
        )
        if not st.session_state["chat_messages"]:
            # 如果这个会话之前没有日志，就放一条默认欢迎语。
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "### 最终回答\n你好，我是 OfficeMate。"
                        "你可以直接问我请假、报销、采购、IT 支持和通知总结等问题。\n\n"
                        "### 操作步骤/材料清单\n无\n\n"
                        "### 风险提示\n我的回答以知识库中的制度与流程文档为准。"
                    ),
                }
            ]
            logger.info("chat_session initialized_with_default_greeting | session_id=%s", st.session_state["session_id"])


def _render_chat_sidebar(storage):
    """渲染聊天页左侧边栏。"""
    with st.sidebar:
        st.subheader("对话设置")
        st.selectbox(
            "知识范围",
            options=config.CATEGORY_FILTER_OPTIONS,
            key="selected_category",
        )
        st.caption(f"当前会话 ID：`{st.session_state['session_id']}`")

        if st.button("新建会话", key="new_session"):
            # 新会话本质上就是换一个 session_id，并清空前端聊天历史。
            st.session_state["session_id"] = f"{config.default_session_prefix}_{uuid4().hex[:8]}"
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "### 最终回答\n已为你创建新会话，可以继续提问。\n\n"
                        "### 操作步骤/材料清单\n无\n\n"
                        "### 风险提示\n如果需要准确回答，请先确认知识库中已有对应制度文档。"
                    ),
                }
            ]
            logger.info("chat_session reset | session_id=%s", st.session_state["session_id"])
            st.rerun()

        st.divider()
        st.subheader("推荐问题")
        st.markdown(
            "- 年假最晚需要提前几天申请？\n"
            "- 报销差旅费需要哪些材料？\n"
            "- 采购一台显示器应该怎么走流程？\n"
            "- VPN 连接失败怎么处理？"
        )

        st.divider()
        stats = storage.get_stats()
        st.caption(
            f"当前已导入 {stats['document_count']} 份文档，累计记录 {stats['qa_count']} 次问答。"
        )


def _load_messages_from_logs(storage, session_id):
    """把已落盘的问答日志恢复成页面展示用的消息结构。"""
    messages = []
    for log in storage.list_session_logs(session_id):
        messages.append({"role": "user", "content": log["question"]})
        messages.append(
            {
                "role": "assistant",
                "content": log["answer"],
                "qa_log_id": log["id"],
                "question_type": log.get("question_type", ""),
                "question": log.get("question", ""),
            }
        )
    return messages


def _render_feedback_form(storage, message):
    """在每条助手回答下方渲染反馈表单。"""
    qa_log_id = message.get("qa_log_id")
    if not qa_log_id:
        return

    existing_feedback = storage.get_feedback_by_qa_log_id(qa_log_id) or {}
    rating_options = ["未评价", "有帮助", "需改进"]
    default_rating = _rating_to_label(existing_feedback.get("rating", ""))
    default_index = rating_options.index(default_rating) if default_rating in rating_options else 0

    with st.expander("反馈", expanded=False):
        # 用 qa_log_id 作为 key，避免多条回答的表单组件互相冲突。
        rating_label = st.radio(
            "这条回答是否有帮助？",
            rating_options,
            index=default_index,
            horizontal=True,
            key=f"rating_{qa_log_id}",
        )
        comment = st.text_input(
            "补充说明",
            value=existing_feedback.get("comment", ""),
            key=f"comment_{qa_log_id}",
        )
        if st.button("保存反馈", key=f"save_feedback_{qa_log_id}"):
            # upsert 表示：有记录就更新，没有记录就新增。
            storage.upsert_feedback(
                qa_log_id=qa_log_id,
                rating=_label_to_rating(rating_label),
                comment=comment,
                session_id=st.session_state["session_id"],
            )
            logger.info(
                "feedback_saved | session_id=%s | qa_log_id=%s | rating=%s",
                st.session_state["session_id"],
                qa_log_id,
                rating_label,
            )
            st.success("反馈已保存。")


def _label_to_rating(label):
    """把页面展示文案映射成存储层使用的枚举值。"""
    return {
        "未评价": "unrated",
        "有帮助": "helpful",
        "需改进": "needs_improvement",
    }.get(label, "unrated")


def _rating_to_label(rating):
    """把存储层枚举值反向映射成页面展示文案。"""
    return {
        "helpful": "有帮助",
        "needs_improvement": "需改进",
        "unrated": "未评价",
        "": "未评价",
    }.get(rating, "未评价")


def _show_upload_result(result):
    """统一处理上传结果提示，避免页面里重复写 if/else。"""
    if result["status"] == "success":
        st.success(result["message"])
    elif result["status"] == "duplicate":
        st.info(result["message"])
    else:
        st.error(result["message"])


def _show_upload_results_summary(results):
    """展示本次批量导入的汇总结果。"""
    if not results:
        return
    success_count = sum(1 for result in results if result["status"] == "success")
    duplicate_count = sum(1 for result in results if result["status"] == "duplicate")
    failed_count = sum(1 for result in results if result["status"] == "failed")
    st.caption(
        f"本次共处理 {len(results)} 份文档：成功 {success_count}，重复跳过 {duplicate_count}，失败 {failed_count}。"
    )


def _render_upload_task_panel():
    """展示当前会话中的后台上传任务进度。"""
    task_ids = st.session_state.get("upload_task_ids", [])
    if not task_ids:
        return

    st.divider()
    st.subheader("后台导入任务")
    st.caption("任务会在后台继续运行；点击刷新后可以查看最新进度。")
    st.button("刷新任务进度", key="refresh_upload_tasks")

    for task_id in task_ids:
        task = upload_task_manager.get_task(task_id)
        if not task:
            continue
        total = task.get("total_documents", 0)
        completed = task.get("completed_documents", 0)
        progress_value = 0.0 if total == 0 else min(1.0, completed / total)
        title = f"{task['created_at']} | {task['stage']} | {task['status']}"
        with st.expander(title, expanded=task["status"] in {"queued", "running"}):
            st.write(task["message"])
            st.progress(progress_value)
            if task.get("active_document"):
                st.caption(f"当前处理：{task['active_document']}")
            summary_frame = pd.DataFrame(
                [
                    {
                        "总文档数": total,
                        "已处理": completed,
                        "成功": task.get("success_count", 0),
                        "重复": task.get("duplicate_count", 0),
                        "失败": task.get("failed_count", 0),
                    }
                ]
            )
            st.dataframe(summary_frame, use_container_width=True, hide_index=True)
            if task.get("results"):
                result_frame = pd.DataFrame(task["results"])
                st.dataframe(result_frame, use_container_width=True, hide_index=True)


def _build_document_option_label(document):
    """把文档元数据拼成管理页下拉框中的可读标签。"""
    return (
        f"{document.get('title', '未命名文档')} | "
        f"{document.get('category', '未分类')} | "
        f"{document.get('version', '-')} | "
        f"{document.get('file_name', '-')}"
    )
