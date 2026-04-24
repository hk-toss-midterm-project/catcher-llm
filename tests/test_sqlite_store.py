from __future__ import annotations

from pathlib import Path

from catcher_llm.config.settings import Settings
from catcher_llm.services.user_data_service import (
    authenticate_user,
    ensure_user_database,
    get_user_transactions,
    list_user_memories,
    save_user_memory,
)


def _write_seed_csvs(csv_dir: Path, *, encoding: str = "utf-8") -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "members_v1.csv").write_text(
        "\n".join(
            [
                "id,name,age,직업,성별,연봉,지역,최상위 카드등급,페르소나",
                "1,김토스,29,개발자,남성,7000,서울,Gold,절약형",
                "2,이캐처,33,기획자,여성,6500,부산,Silver,안정형",
            ]
        ),
        encoding=encoding,
    )
    (csv_dir / "consumption_v1.csv").write_text(
        "\n".join(
            [
                "멤버 id,id,사용 금액,사용 시간,결제 내역,결제 장소 (가맹점 여부),할부 여부,할부 개월,할부 무/유이자 여부,거래 상태 (승인 / 취소),해외 결제,업종 카테고리,결제 방식 (온/오프라인)",
                "1,100,12000,2026-04-01 09:00:00,커피,가맹점,아니오,0,해당없음,승인,아니오,식음료,오프라인",
                "1,101,45000,2026-04-02 18:30:00,마트,가맹점,아니오,0,해당없음,승인,아니오,생활,오프라인",
                "2,102,32000,2026-04-03 12:00:00,점심,가맹점,아니오,0,해당없음,승인,아니오,식음료,오프라인",
            ]
        ),
        encoding=encoding,
    )


def _make_settings(tmp_path: Path, *, csv_encoding: str = "utf-8") -> Settings:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    csv_dir = raw_dir / "csv"
    _write_seed_csvs(csv_dir, encoding=csv_encoding)
    return Settings(
        data_dir=data_dir,
        raw_data_dir=raw_dir,
        processed_data_dir=data_dir / "processed",
        vectorstore_dir=data_dir / "vectordb",
        eval_data_dir=data_dir / "evals",
        sqlite_db_path=data_dir / "sqlite" / "app.sqlite3",
    )


def test_ensure_user_database_seeds_sqlite_from_csv(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    result = ensure_user_database(settings=settings)

    assert settings.sqlite_db_path.exists()
    assert result.user_count == 2
    assert result.transaction_count == 3
    assert result.memory_count == 0


def test_settings_exposes_session_sqlite_path_in_same_directory(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    assert settings.sqlite_db_path == tmp_path / "data" / "sqlite" / "app.sqlite3"
    assert settings.session_sqlite_db_path == tmp_path / "data" / "sqlite" / "session.sqlite3"


def test_ensure_user_database_reads_utf8_bom_csv_headers(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, csv_encoding="utf-8-sig")

    result = ensure_user_database(settings=settings)

    assert result.user_count == 2
    assert result.transaction_count == 3


def test_authenticate_user_and_load_transactions_from_sqlite(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)

    profile = authenticate_user(1, "김토스", settings=settings)
    transactions = get_user_transactions(1, settings=settings)

    assert profile is not None
    assert profile["name"] == "김토스"
    assert profile["card_grade"] == "Gold"
    assert len(transactions) == 2
    assert transactions[0]["user_id"] == 1


def test_save_user_memory_persists_memory_in_same_sqlite_database(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)

    save_user_memory(
        user_id=1,
        memory_key="preferred_category",
        content="식음료 지출을 중요하게 본다.",
        settings=settings,
    )

    memories = list_user_memories(1, settings=settings)

    assert len(memories) == 1
    assert memories[0]["memory_key"] == "preferred_category"
    assert memories[0]["content"] == "식음료 지출을 중요하게 본다."
