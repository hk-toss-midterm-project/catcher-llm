"""지원하는 전체 원본 문서를 로컬 RAG 전처리 파이프라인에 적재한다."""

from __future__ import annotations

from catcher_llm.config.settings import get_settings
from catcher_llm.services.ingestion_service import ingest_local_documents


def main() -> None:
    result = ingest_local_documents(chunk_size=600, chunk_overlap=60, settings=get_settings())
    print(f"Indexed {result['documents']} files")
    print(f"Chunks: {result['chunks']}")
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
