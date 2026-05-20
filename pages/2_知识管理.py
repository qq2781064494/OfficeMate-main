"""Streamlit 多页面模式下的“知识管理”页面入口。"""

import streamlit as st

from services.ui.management_page import render_management_page


# 管理页入口同样保持极薄，方便把 UI 与业务逻辑拆开。
st.set_page_config(page_title="OfficeMate - 知识管理", layout="wide")
render_management_page()
