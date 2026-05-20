"""主聊天页入口。

教学阅读建议：
1. Streamlit 应用启动时会先执行这个文件。
2. 这里本身不写业务逻辑，只负责页面初始化。
3. 真正的聊天页内容在 services/ui_pages.py 里的 render_chat_page。
"""

import streamlit as st

from services.ui.chat_page import render_chat_page

# 先设置当前页面的标题和布局，再交给页面层函数去渲染具体内容。
st.set_page_config(page_title="OfficeMate", layout="wide")
render_chat_page()
