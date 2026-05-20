"""真正的 create_agent 风格 ReAct-RAG 页面。"""

from __future__ import annotations

from uuid import uuid4

import streamlit as st

import config_data as config
from agent_react_rag.service import AgentReactRagService
from services.storage_service import JsonStorageService
from utils.log_tool import get_logger


logger = get_logger("agent_react_rag_ui")


def _yield_text_chunks(text: str, chunk_size: int = 64):
    """把完整文本切成小块，用于页面流式展示。"""
    for index in range(0, len(text), chunk_size):
        yield text[index:index + chunk_size]


def render_agent_react_rag_page():
    storage = JsonStorageService()
    _init_session(storage)
    _render_sidebar(storage)

    st.title("OfficeMate：Agent ReAct RAG 问答")
    st.caption("这一页使用真正的 create_agent(...)。模型会自己决定是否调用 rewrite、plan、retrieve_and_rerank、generate_final_answer。")

    for message in st.session_state["agent_react_rag_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_trace(message)

    prompt = st.chat_input("例如：我下周出差报销和补贴怎么算，我还需要先走审批吗？")
    if not prompt:
        return

    question_category = st.session_state.get("agent_react_rag_category", "全部")
    logger.info(
        "agent_react_rag_page received_question | session_id=%s | category=%s | question=%s",
        st.session_state["agent_react_rag_session_id"],
        question_category,
        prompt,
    )
    st.session_state["agent_react_rag_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = AgentReactRagService().answer_question(
            question=prompt,
            session_id=st.session_state["agent_react_rag_session_id"],
            category=question_category,
        )
        rendered_text = st.write_stream(_yield_text_chunks(result.answer))
        assistant_message = {
            "role": "assistant",
            "content": rendered_text,
            "qa_log_id": result.qa_log_id,
            "question_type": result.question_type,
            "trace": result.trace,
            "question": prompt,
        }
        _render_trace(assistant_message)
        st.session_state["agent_react_rag_messages"].append(assistant_message)
        logger.info(
            "agent_react_rag_page answer_rendered | session_id=%s | qa_log_id=%s | question_type=%s",
            st.session_state["agent_react_rag_session_id"],
            result.qa_log_id,
            result.question_type,
        )


def _init_session(storage):
    if "agent_react_rag_session_id" not in st.session_state:
        st.session_state["agent_react_rag_session_id"] = f"agent_react_rag_{uuid4().hex[:8]}"
    if "agent_react_rag_category" not in st.session_state:
        st.session_state["agent_react_rag_category"] = "全部"
    if "agent_react_rag_messages" not in st.session_state:
        st.session_state["agent_react_rag_messages"] = _load_messages(storage, st.session_state["agent_react_rag_session_id"])
        if not st.session_state["agent_react_rag_messages"]:
            st.session_state["agent_react_rag_messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "### 最终回答\n这是基于 `create_agent(...)` 的真正工具型 ReAct RAG 页面。"
                        "模型会自己判断要不要调用 rewrite、plan、retrieve_and_rerank、generate_final_answer。\n\n"
                        "### 操作步骤/材料清单\n无\n\n"
                        "### 风险提示\n这是实验页面，用于和固定流水线、Decision-ReAct 进行对比。"
                    ),
                    "trace": [],
                }
            ]


def _render_sidebar(storage):
    with st.sidebar:
        st.subheader("Agent ReAct RAG 设置")
        st.selectbox("知识范围", options=config.CATEGORY_FILTER_OPTIONS, key="agent_react_rag_category")
        st.caption(f"当前会话 ID：`{st.session_state['agent_react_rag_session_id']}`")
        if st.button("新建 Agent ReAct 会话", key="new_agent_react_rag_session"):
            st.session_state["agent_react_rag_session_id"] = f"agent_react_rag_{uuid4().hex[:8]}"
            st.session_state["agent_react_rag_messages"] = []
            st.rerun()

        st.divider()
        stats = storage.get_stats()
        st.caption(f"当前已导入 {stats['document_count']} 份文档，累计记录 {stats['qa_count']} 次问答。")


def _load_messages(storage, session_id):
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
                "trace": log.get("trace", []),
            }
        )
    return messages


def _render_trace(message):
    trace = message.get("trace") or []
    if not trace:
        return

    with st.expander("工具轨迹", expanded=False):
        for item in trace:
            st.markdown(
                f"- `{item.get('step', '')}` | {item.get('duration_ms', 0)} ms\n"
                f"  结果：{item.get('summary', '')}"
            )
