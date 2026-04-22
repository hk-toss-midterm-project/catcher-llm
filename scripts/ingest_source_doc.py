from __future__ import annotations

import argparse
from pathlib import Path

from catcher_llm.config.settings import get_settings
from catcher_llm.services.ingestion_service import ingest_selected_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a selected source document into the local RAG pipeline."
    )
    parser.add_argument("source", help="Path to the source document to ingest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source document not found: {source_path}")

    result = ingest_selected_documents(
        [source_path],
        settings=get_settings(),
        manifest_name=f"{source_path.stem}_ingestion_manifest.json",
    )
    print(f"Indexed {result['documents']} file")
    print(f"Chunks: {result['chunks']}")
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
