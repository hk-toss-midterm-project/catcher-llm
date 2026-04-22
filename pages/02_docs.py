from __future__ import annotations

from pathlib import Path

import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.ingestion_service import discover_source_files, ingest_local_documents
from catcher_llm.ui.components import render_sidebar

ROOT_DIR = Path(__file__).resolve().parents[1]

settings = get_settings()

st.set_page_config(page_title=f"{settings.app_name} | Docs", layout="wide")

render_sidebar(settings)

st.title("Docs")
st.caption("Keep raw source files in data/raw and write derived artifacts to data/processed.")

source_files = discover_source_files(settings)

stats_left, stats_right = st.columns(2)
stats_left.metric("Source files", len(source_files))
stats_right.metric("Vector store dir", 1 if settings.vectorstore_dir.exists() else 0)

if st.button("Build ingestion manifest"):
    result = ingest_local_documents(settings)
    st.success(
        f"Indexed {result['documents']} files into {result['chunks']} chunks and wrote {result['manifest_path']}"
    )

if source_files:
    st.subheader("Detected files")
    st.table([{"path": str(path.relative_to(ROOT_DIR))} for path in source_files])
else:
    st.info("No source files found in data/raw yet.")
