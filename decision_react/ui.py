"""Decision-ReAct 页面。"""

from __future__ import annotations

from uuid import uuid4

import streamlit as st

import config_data as config
from decision_react.service import DecisionReactService
from services.storage_service import JsonStorageService
from utils.log_tool import get_logger


logger = get_logger("decision_react_ui")


def render_decision_react_page():
    """渲染 Decision-Guided ReAct 风格问答页面。"""
    storage = JsonStorageService()
    _init_session(storage)
    _render_sidebar(storage)

    st.title("OfficeMate：Decision-Guided ReAct 问答")
    st.caption("这一页会先判断问题复杂度，再按需走 4 个阶段工具：问题理解、证据检索、答案生成、最终整理。")

    for message in st.session_state["decision_react_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_trace(message)

    prompt = st.chat_input("例如：年假最晚需要提前几天申请？")
    if not prompt:
        return

    question_category = st.session_state.get("decision_react_category", "全部")
    logger.info(
        "decision_react_page received_question | session_id=%s | category=%s | question=%s",
        st.session_state["decision_react_session_id"],
        question_category,
        prompt,
    )
    st.session_state["decision_react_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        def update_status(message):
            status_placeholder.markdown(f"_{message}_")

        result = DecisionReactService().answer_question(
            question=prompt,
            session_id=st.session_state["decision_react_session_id"],
            category=question_category,
            status_callback=update_status,
        )
        status_placeholder.empty()
        st.markdown(result.answer)

        assistant_message = {
            "role": "assistant",
            "content": result.answer,
            "qa_log_id": result.qa_log_id,
            "question_type": result.question_type,
            "decision": result.decision,
            "trace": result.trace,
            "question": prompt,
        }
        _render_trace(assistant_message)
        st.session_state["decision_react_messages"].append(assistant_message)
        logger.info(
            "decision_react_page answer_rendered | session_id=%s | qa_log_id=%s | question_type=%s",
            st.session_state["decision_react_session_id"],
            result.qa_log_id,
            result.question_type,
        )


def _init_session(storage):
    if "decision_react_session_id" not in st.session_state:
        st.session_state["decision_react_session_id"] = f"decision_react_{uuid4().hex[:8]}"
    if "decision_react_category" not in st.session_state:
        st.session_state["decision_react_category"] = "全部"
    if "decision_react_messages" not in st.session_state:
        st.session_state["decision_react_messages"] = _load_messages(storage, st.session_state["decision_react_session_id"])
        if not st.session_state["decision_react_messages"]:
            st.session_state["decision_react_messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "### 最终回答\n你好，这里是 Decision-Guided ReAct 实验页面。"
                        "我会先判断问题复杂度，再决定是否要 rewrite、拆任务和汇总答案。\n\n"
                        "### 操作步骤/材料清单\n无\n\n"
                        "### 风险提示\n这一页用于对比 Agent 决策式编排与固定流水线的差异。"
                    ),
                    "trace": [],
                    "decision": {},
                }
            ]


def _render_sidebar(storage):
    with st.sidebar:
        st.subheader("Decision-ReAct 设置")
        st.selectbox("知识范围", options=config.CATEGORY_FILTER_OPTIONS, key="decision_react_category")
        st.caption(f"当前会话 ID：`{st.session_state['decision_react_session_id']}`")
        if st.button("新建 Decision-ReAct 会话", key="new_decision_react_session"):
            st.session_state["decision_react_session_id"] = f"decision_react_{uuid4().hex[:8]}"
            st.session_state["decision_react_messages"] = []
            st.rerun()

        st.divider()
        stats = storage.get_stats()
        st.caption(
            f"当前已导入 {stats['document_count']} 份文档，累计记录 {stats['qa_count']} 次问答。"
        )


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
                "decision": log.get("decision", {}),
            }
        )
    return messages


def _render_trace(message):
    trace = message.get("trace") or []
    decision = message.get("decision") or {}
    if not trace and not decision:
        return

    with st.expander("执行轨迹", expanded=False):
        if decision:
            st.markdown("**决策结果**")
            st.json(decision)
        if trace:
            st.markdown("**工具执行轨迹**")
            for item in trace:
                st.markdown(
                    f"- `{item.get('step', '')}` | {item.get('duration_ms', 0)} ms\n"
                    f"  结果：{item.get('summary', '')}"
                )
