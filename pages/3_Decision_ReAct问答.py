"""Decision-Guided ReAct 风格问答页面入口。"""

import streamlit as st

from decision_react.ui import render_decision_react_page


st.set_page_config(page_title="OfficeMate - Decision ReAct", layout="wide")
render_decision_react_page()
