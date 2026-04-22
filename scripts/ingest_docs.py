from __future__ import annotations

from catcher_llm.config.settings import get_settings
from catcher_llm.services.ingestion_service import ingest_local_documents


def main() -> None:
    result = ingest_local_documents(get_settings())
    print(f"Indexed {result['documents']} files")
    print(f"Chunks: {result['chunks']}")
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
