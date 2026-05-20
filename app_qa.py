"""聊天页的兼容入口。

这个文件和 app.py 做的事情几乎一样，保留它主要是为了兼容旧的启动方式
或课程演示中的旧入口名称。
"""

import streamlit as st

from services.ui.chat_page import render_chat_page


# 这里仍然只是一个薄入口：设置页面后，把控制权交给聊天页渲染函数。
st.set_page_config(page_title="OfficeMate", layout="wide")
render_chat_page()
