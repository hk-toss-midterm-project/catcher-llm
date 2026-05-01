from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from catcher_llm.config.settings import Settings
from catcher_llm.db.models import SessionModel
from catcher_llm.db.session import session_scope
from catcher_llm.services.daily_report_defaults import get_default_daily_report_selection
from catcher_llm.services.user_data_service import (
    UserRegistrationInput,
    authenticate_user,
    ensure_user_database,
    get_user_registration_columns,
    get_user_transactions,
    list_user_memories,
    register_user,
    save_user_memory,
)


def _quote_sqlite_identifier(identifier: str) -> str:
    """SQLite 식별자에 들어갈 큰따옴표를 이스케이프한다."""
    return identifier.replace('"', '""')


def _fetch_sqlite_table_columns(sqlite_db_path: Path, table_name: str) -> list[str]:
    """SQLite 테이블의 실제 컬럼 이름 목록을 조회한다."""
    escaped_table_name = _quote_sqlite_identifier(table_name)
    with sqlite3.connect(sqlite_db_path) as connection:
        rows = connection.execute(f'PRAGMA table_info("{escaped_table_name}")').fetchall()
    return [str(row[1]) for row in rows]


def _fetch_sqlite_cell(
    sqlite_db_path: Path,
    table_name: str,
    column_name: str,
    *,
    row_id: int,
) -> str | None:
    """SQLite 테이블에서 특정 ID 행의 단일 컬럼 값을 직접 조회한다."""
    escaped_table_name = _quote_sqlite_identifier(table_name)
    escaped_column_name = _quote_sqlite_identifier(column_name)
    with sqlite3.connect(sqlite_db_path) as connection:
        row = connection.execute(
            f'SELECT "{escaped_column_name}" FROM "{escaped_table_name}" WHERE id = ?',
            (row_id,),
        ).fetchone()
    if row is None:
        return None
    return row[0]


def _hash_file_for_seed_metadata(path: Path) -> str:
    """시드 메타데이터 테스트에 사용할 파일 SHA-256 해시를 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_seed_csvs(csv_dir: Path, *, encoding: str = "utf-8") -> None:
    """SQLite 시드 테스트에 사용할 v3 사용자·거래 CSV를 작성한다."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "users_v3.csv").write_text(
        "\n".join(
            [
                "id,name,age,occupation,gender,annual_income,region,persona,personal_score,saving_goal_text,target_max_spending_amount",
                "1,김토스,29,개발자,남성,7000,서울,절약형,53,비상금 만들기,1500000",
                "2,이캐처,33,기획자,여성,6500,부산,안정형,55,생활비 관리,1800000",
            ]
        ),
        encoding=encoding,
    )
    (csv_dir / "transactions_v3.csv").write_text(
        "\n".join(
            [
                "id,user_id,amount,transaction_time,description,merchant_name,is_installment,installment_months,is_interest_free,status,is_overseas,category,payment_channel",
                "100,1,12000,2026-04-01 09:00:00,커피,스타벅스,False,0,False,APPROVED,False,식음료,OFFLINE",
                "101,1,45000,2026-04-02 18:30:00,마트,이마트,False,0,False,APPROVED,False,생활,OFFLINE",
                "102,2,32000,2026-04-03 12:00:00,점심,식당,False,0,False,APPROVED,False,식음료,OFFLINE",
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
    """변경된 users/transactions CSV 파일명으로 SQLite 시드가 동작하는지 검증한다."""
    settings = _make_settings(tmp_path)

    result = ensure_user_database(settings=settings)

    assert settings.members_csv_path.name == "users_v3.csv"
    assert settings.consumption_csv_path.name == "transactions_v3.csv"
    assert settings.sqlite_db_path.exists()
    assert result.user_count == 2
    assert result.transaction_count == 3
    assert result.memory_count == 0


def test_ensure_user_database_adds_feedback_reaction_columns_to_session_table(
    tmp_path: Path,
) -> None:
    """세션 테이블에 피드백 반응과 반응 사유를 저장할 컬럼이 준비되는지 검증한다."""
    settings = _make_settings(tmp_path)

    ensure_user_database(settings=settings)

    session_columns = _fetch_sqlite_table_columns(settings.sqlite_db_path, "session")
    assert "feedback_reaction" in session_columns
    assert "feedback_reaction_reason" in session_columns


def test_save_session_feedback_reaction_persists_reaction_and_reason(
    tmp_path: Path,
) -> None:
    """피드백 좋아요/싫어요 반응과 사용자가 입력한 사유가 세션 행에 저장되는지 검증한다."""
    from catcher_llm.services.consumption_feedback.feedback_reaction import (
        save_session_feedback_reaction,
    )

    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)
    with session_scope(settings) as db_session:
        db_session.add(
            SessionModel(
                user_id=1,
                analysis_date="2026-04-02",
                period_type="daily",
                feedback_message="저장된 피드백입니다.",
            )
        )

    save_session_feedback_reaction(
        member_id=1,
        analysis_date=date(2026, 4, 2),
        period_type="daily",
        reaction="like",
        reason=None,
        settings=settings,
    )
    save_session_feedback_reaction(
        member_id=1,
        analysis_date=date(2026, 4, 2),
        period_type="daily",
        reaction="like",
        reason="근거가 명확해서 도움이 됐습니다.",
        settings=settings,
    )

    with session_scope(settings) as db_session:
        saved_session = (
            db_session.query(SessionModel)
            .filter_by(user_id=1, analysis_date="2026-04-02", period_type="daily")
            .one()
        )

    assert saved_session.feedback_reaction == "like"
    assert saved_session.feedback_reaction_reason == "근거가 명확해서 도움이 됐습니다."


def test_default_daily_report_selection_uses_member_day_with_history(tmp_path: Path) -> None:
    """일간 보고서 기본값이 실제 거래와 과거 거래가 모두 있는 회원 날짜를 선택하는지 검증한다."""
    settings = _make_settings(tmp_path)

    selection = get_default_daily_report_selection(settings=settings)

    assert selection.member_id == 1
    assert selection.analysis_date == date(2026, 4, 2)
    assert selection.previous_date == date(2026, 4, 1)


def test_ensure_user_database_rebuilds_sqlite_when_members_csv_changes(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    first_result = ensure_user_database(settings=settings)
    assert first_result.user_count == 2

    settings.members_csv_path.write_text(
        "\n".join(
            [
                "id,name,age,직업,성별,연봉,지역,최상위 카드등급,페르소나",
                "1,김토스,29,개발자,남성,7000,서울,Gold,절약형",
                "2,이캐처,33,기획자,여성,6500,부산,Silver,안정형",
                "3,박업데이트,31,디자이너,여성,6200,대전,Platinum,탐색형",
            ]
        ),
        encoding="utf-8",
    )

    rebuilt_result = ensure_user_database(settings=settings)
    profile = authenticate_user(3, "박업데이트", settings=settings)

    assert rebuilt_result.user_count == 3
    assert rebuilt_result.transaction_count == 3
    assert profile is not None
    assert profile["name"] == "박업데이트"


def test_ensure_user_database_rebuilds_sqlite_when_consumption_csv_changes(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    first_result = ensure_user_database(settings=settings)
    assert first_result.transaction_count == 3

    settings.consumption_csv_path.write_text(
        "\n".join(
            [
                "멤버 id,id,사용 금액,사용 시간,결제 내역,결제 장소 (가맹점 여부),할부 여부,할부 개월,할부 무/유이자 여부,거래 상태 (승인 / 취소),해외 결제,업종 카테고리,결제 방식 (온/오프라인)",
                "1,100,12000,2026-04-01 09:00:00,커피,가맹점,아니오,0,해당없음,승인,아니오,식음료,오프라인",
                "1,101,45000,2026-04-02 18:30:00,마트,가맹점,아니오,0,해당없음,승인,아니오,생활,오프라인",
                "2,102,32000,2026-04-03 12:00:00,점심,가맹점,아니오,0,해당없음,승인,아니오,식음료,오프라인",
                "2,103,18000,2026-04-04 08:15:00,베이커리,가맹점,아니오,0,해당없음,승인,아니오,식음료,오프라인",
            ]
        ),
        encoding="utf-8",
    )

    rebuilt_result = ensure_user_database(settings=settings)
    transactions = get_user_transactions(2, settings=settings)

    assert rebuilt_result.transaction_count == 4
    assert len(transactions) == 2
    assert transactions[-1]["id"] == 103


def test_ensure_user_database_rebuilds_sqlite_with_added_csv_columns(tmp_path: Path) -> None:
    """CSV에 새 헤더가 생기면 SQLite 테이블에도 컬럼과 값이 함께 반영되는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)

    settings.members_csv_path.write_text(
        "\n".join(
            [
                "id,name,age,직업,성별,연봉,지역,최상위 카드등급,페르소나,선호 브랜드",
                "1,김토스,29,개발자,남성,7000,서울,Gold,절약형,토스카드",
                "2,이캐처,33,기획자,여성,6500,부산,Silver,안정형,캐처뱅크",
            ]
        ),
        encoding="utf-8",
    )
    settings.consumption_csv_path.write_text(
        "\n".join(
            [
                "멤버 id,id,사용 금액,사용 시간,결제 내역,결제 장소 (가맹점 여부),할부 여부,할부 개월,할부 무/유이자 여부,거래 상태 (승인 / 취소),해외 결제,업종 카테고리,결제 방식 (온/오프라인),메모 태그",
                "1,100,12000,2026-04-01 09:00:00,커피,가맹점,아니오,0,해당없음,승인,아니오,식음료,오프라인,아침",
                "1,101,45000,2026-04-02 18:30:00,마트,가맹점,아니오,0,해당없음,승인,아니오,생활,오프라인,장보기",
                "2,102,32000,2026-04-03 12:00:00,점심,가맹점,아니오,0,해당없음,승인,아니오,식음료,오프라인,점심",
            ]
        ),
        encoding="utf-8",
    )

    rebuilt_result = ensure_user_database(settings=settings)

    assert rebuilt_result.user_count == 2
    assert rebuilt_result.transaction_count == 3
    assert "선호 브랜드" in _fetch_sqlite_table_columns(settings.sqlite_db_path, "users")
    assert "메모 태그" in _fetch_sqlite_table_columns(settings.sqlite_db_path, "transactions")
    assert (
        _fetch_sqlite_cell(settings.sqlite_db_path, "users", "선호 브랜드", row_id=1) == "토스카드"
    )
    assert (
        _fetch_sqlite_cell(settings.sqlite_db_path, "transactions", "메모 태그", row_id=100)
        == "아침"
    )


def test_ensure_user_database_rebuilds_when_seed_metadata_is_from_older_schema(
    tmp_path: Path,
) -> None:
    """구버전 시드 메타데이터가 남아 있어도 새 스키마 버전으로 재시드하는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)

    settings.members_csv_path.write_text(
        "\n".join(
            [
                "id,name,age,직업,성별,연봉,지역,최상위 카드등급,페르소나,선호 브랜드",
                "1,김토스,29,개발자,남성,7000,서울,Gold,절약형,토스카드",
                "2,이캐처,33,기획자,여성,6500,부산,Silver,안정형,캐처뱅크",
            ]
        ),
        encoding="utf-8",
    )

    seed_metadata_path = settings.sqlite_db_path.with_suffix(".sqlite3.seed-meta.json")
    legacy_signature = {
        "seed_schema_version": 3,
        "members_csv": {
            "path": str(settings.members_csv_path.resolve()),
            "sha256": _hash_file_for_seed_metadata(settings.members_csv_path),
        },
        "consumption_csv": {
            "path": str(settings.consumption_csv_path.resolve()),
            "sha256": _hash_file_for_seed_metadata(settings.consumption_csv_path),
        },
    }
    seed_metadata_path.write_text(json.dumps(legacy_signature, indent=2), encoding="utf-8")

    ensure_user_database(settings=settings)

    assert "선호 브랜드" in _fetch_sqlite_table_columns(settings.sqlite_db_path, "users")
    assert (
        _fetch_sqlite_cell(settings.sqlite_db_path, "users", "선호 브랜드", row_id=1) == "토스카드"
    )


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
    """v3 영문 CSV 헤더의 사용자 ID와 거래 user_id를 SQLite 모델 필드로 매핑하는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)

    profile = authenticate_user(1, "김토스", settings=settings)
    transactions = get_user_transactions(1, settings=settings)

    assert profile is not None
    assert profile["name"] == "김토스"
    assert profile["job"] == "개발자"
    assert profile["income"] == "7000"
    assert profile["saving_goal_text"] == "비상금 만들기"
    assert len(transactions) == 2
    assert transactions[0]["user_id"] == 1
    assert transactions[0]["id"] == 100


def test_get_user_registration_columns_reads_users_v3_header(tmp_path: Path) -> None:
    """회원가입 폼 구성을 위해 v3 사용자 CSV 헤더를 순서대로 읽는지 검증한다."""
    settings = _make_settings(tmp_path)

    columns = get_user_registration_columns(settings=settings)

    assert columns == [
        "id",
        "name",
        "age",
        "occupation",
        "gender",
        "annual_income",
        "region",
        "persona",
        "personal_score",
        "saving_goal_text",
        "target_max_spending_amount",
    ]


def test_register_user_saves_sqlite_without_mutating_csv(tmp_path: Path) -> None:
    """회원가입 입력값과 기본 개인 점수 0을 SQLite에 저장하고 CSV는 변경하지 않는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)
    original_members_csv = settings.members_csv_path.read_text(encoding="utf-8")

    result = register_user(
        UserRegistrationInput(
            name="신규가입",
            age=35,
            occupation="데이터 분석가",
            gender="Female",
            annual_income=72_000_000,
            region="서울 서울-마포구",
            persona="지출 패턴을 꼼꼼히 기록하는 직장인입니다.",
            saving_goal_text="전세 보증금 마련하기",
            target_max_spending_amount=2_100_000,
        ),
        settings=settings,
    )

    profile = authenticate_user(result.user_id, "신규가입", settings=settings)

    assert result.user_id == 3
    assert profile is not None
    assert profile["name"] == "신규가입"
    assert profile["job"] == "데이터 분석가"
    assert profile["income"] == "72000000"
    assert profile["saving_goal_text"] == "전세 보증금 마련하기"
    assert _fetch_sqlite_cell(settings.sqlite_db_path, "users", "personal_score", row_id=3) == "0"
    assert (
        _fetch_sqlite_cell(
            settings.sqlite_db_path,
            "users",
            "target_max_spending_amount",
            row_id=3,
        )
        == "2100000"
    )
    assert ensure_user_database(settings=settings).user_count == 3
    assert settings.members_csv_path.read_text(encoding="utf-8") == original_members_csv


def test_register_user_rejects_empty_name(tmp_path: Path) -> None:
    """회원가입 시 필수 이름이 비어 있으면 CSV와 SQLite를 변경하지 않는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)

    with pytest.raises(ValueError, match="이름"):
        register_user(
            UserRegistrationInput(
                name=" ",
                age=35,
                occupation="데이터 분석가",
                gender="Female",
                annual_income=72_000_000,
                region="서울 서울-마포구",
                persona="지출 패턴을 꼼꼼히 기록하는 직장인입니다.",
                saving_goal_text="전세 보증금 마련하기",
                target_max_spending_amount=2_100_000,
            ),
            settings=settings,
        )

    assert ensure_user_database(settings=settings).user_count == 2


def test_save_user_memory_persists_memory_in_same_sqlite_database(tmp_path: Path) -> None:
    """사용자 메모리가 기간 유형별 요약 구조로 SQLite에 저장되는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)

    save_user_memory(
        user_id=1,
        period_type="daily",
        summary="식음료 지출을 중요하게 본다.",
        settings=settings,
    )

    memories = list_user_memories(1, settings=settings)

    assert len(memories) == 1
    assert memories[0]["period_type"] == "daily"
    assert memories[0]["summary"] == "식음료 지출을 중요하게 본다."
