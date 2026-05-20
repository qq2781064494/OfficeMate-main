"""知识上传页的兼容入口。

教学上可以把它理解为：
- 入口文件负责“把页面打开”
- services.ui_pages 负责“把页面内容画出来”
"""

import streamlit as st

from services.ui.upload_page import render_upload_page


# 上传页和聊天页一样，真正的交互逻辑不在这里，而在 render_upload_page 中。
st.set_page_config(page_title="OfficeMate - 知识上传", layout="wide")
render_upload_page()
