"""Streamlit 多页面模式下的“知识上传”页面入口。"""

import streamlit as st

from services.ui.upload_page import render_upload_page


# Streamlit 会把 pages/ 目录中的文件自动识别为侧边栏页面。
st.set_page_config(page_title="OfficeMate - 知识上传", layout="wide")
render_upload_page()
