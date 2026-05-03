from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st

st.session_state.dev_report_member_id_input_enabled = True
st.session_state.setdefault("dev_report_member_id", 1)

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "pages/05_weekly_report.py"),
    run_name="__main__",
)
