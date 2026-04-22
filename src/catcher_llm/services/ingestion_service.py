from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.retrievers.loaders import (
    iter_source_files,
    load_local_documents,
    load_split_local_documents,
)
from catcher_llm.retrievers.vectorstore import ensure_vectorstore_dir


def discover_source_files(settings: Settings | None = None) -> list[Path]:
    config = settings or get_settings()
    return iter_source_files(config.raw_data_dir)


def ingest_local_documents(settings: Settings | None = None) -> dict[str, str | int]:
    config = settings or get_settings()
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    ensure_vectorstore_dir(config)

    documents = load_local_documents(config.raw_data_dir)
    chunks = load_split_local_documents(
        config.raw_data_dir,
        chunk_size=config.rag_chunk_size,
        chunk_overlap=config.rag_chunk_overlap,
    )
    chunk_counts = Counter(chunk.metadata["source"] for chunk in chunks)
    manifest = [
        {
            "source": document.metadata["source"],
            "chars": len(document.page_content),
            "chunks": chunk_counts.get(document.metadata["source"], 0),
        }
        for document in documents
    ]

    manifest_path = config.processed_data_dir / "ingestion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "manifest_path": str(manifest_path),
    }
