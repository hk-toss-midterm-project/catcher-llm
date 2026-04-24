from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData, Table, func, inspect, select

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import (
    TRANSACTION_CSV_COLUMN_TO_DB_COLUMN,
    USER_CSV_COLUMN_TO_DB_COLUMN,
    TransactionModel,
    UserMemoryModel,
    UserModel,
)
from catcher_llm.db.session import (
    create_database_tables,
    dispose_engine,
    get_engine,
    session_scope,
)

_SEED_METADATA_SUFFIX = ".seed-meta.json"
_SQLITE_SEED_SCHEMA_VERSION = 2
type SeedCellValue = int | str | datetime | None


@dataclass(slots=True, frozen=True)
class DatabaseSeedResult:
    user_count: int
    transaction_count: int
    memory_count: int
    sqlite_db_path: str


def _get_seed_metadata_path(config: Settings) -> Path:
    """현재 SQLite 파일에 대응하는 시드 메타데이터 파일 경로를 계산한다."""
    return config.sqlite_db_path.with_suffix(
        f"{config.sqlite_db_path.suffix}{_SEED_METADATA_SUFFIX}"
    )


def _hash_file(path: Path) -> str | None:
    """파일이 존재하면 SHA-256 해시를 계산하고 없으면 `None`을 반환한다."""
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_csv_signature(config: Settings) -> dict[str, Any]:
    """회원 CSV와 소비 CSV의 경로 및 해시를 묶은 시드 시그니처를 만든다."""
    return {
        "seed_schema_version": _SQLITE_SEED_SCHEMA_VERSION,
        "members_csv": {
            "path": str(config.members_csv_path.resolve()),
            "sha256": _hash_file(config.members_csv_path),
        },
        "consumption_csv": {
            "path": str(config.consumption_csv_path.resolve()),
            "sha256": _hash_file(config.consumption_csv_path),
        },
    }


def _load_seed_metadata(path: Path) -> dict[str, Any] | None:
    """시드 메타데이터 JSON을 읽어오고 손상된 경우 `None`을 반환한다."""
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _write_seed_metadata(config: Settings, signature: dict[str, Any]) -> None:
    """현재 CSV 시그니처를 시드 메타데이터 파일로 저장한다."""
    metadata_path = _get_seed_metadata_path(config)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")


def _reset_sqlite_database(config: Settings) -> None:
    """기존 SQLite 데이터베이스와 시드 메타데이터를 삭제해 재생성 상태로 되돌린다."""
    dispose_engine(config)
    config.sqlite_db_path.unlink(missing_ok=True)
    _get_seed_metadata_path(config).unlink(missing_ok=True)


def _should_rebuild_sqlite_database(config: Settings, signature: dict[str, Any]) -> bool:
    """현재 CSV 시그니처가 저장된 이력과 다르면 DB를 다시 만들어야 하는지 판단한다."""
    if not config.sqlite_db_path.exists():
        return True

    saved_signature = _load_seed_metadata(_get_seed_metadata_path(config))
    return saved_signature != signature


def _iter_csv_rows(path: Path) -> list[dict[str, str | None]]:
    """BOM과 공백이 섞인 CSV 헤더를 정리해 행 목록으로 읽어온다."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is not None:
            reader.fieldnames = [field.strip() if field else "" for field in reader.fieldnames]
        return [
            {(key.strip() if key else ""): value for key, value in row.items()} for row in reader
        ]


def _parse_int(value: str | None) -> int | None:
    """비어 있을 수 있는 문자열 정수를 정리해 `int` 또는 `None`으로 변환한다."""
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return int(stripped)


def _parse_datetime(value: str | None) -> datetime | None:
    """비어 있을 수 있는 ISO 형식 문자열을 `datetime` 또는 `None`으로 변환한다."""
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return datetime.fromisoformat(stripped)


def _parse_text(value: str | None) -> str | None:
    """비어 있을 수 있는 문자열을 공백 제거 후 `str` 또는 `None`으로 변환한다."""
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return stripped


def _quote_sqlite_identifier(identifier: str) -> str:
    """SQLite 식별자에 들어갈 큰따옴표를 이스케이프한다."""
    return identifier.replace('"', '""')


def _collect_dynamic_column_names(
    rows: Sequence[dict[str, SeedCellValue]],
    *,
    static_column_names: set[str],
) -> list[str]:
    """시드 행 목록에서 정적 스키마에 없는 추가 컬럼 이름만 순서대로 모은다."""
    collected_columns: list[str] = []
    seen_columns: set[str] = set()
    for row in rows:
        for column_name in row:
            if column_name in static_column_names or column_name in seen_columns:
                continue
            seen_columns.add(column_name)
            collected_columns.append(column_name)
    return collected_columns


def _ensure_dynamic_sqlite_columns(
    config: Settings,
    *,
    table_name: str,
    column_names: Sequence[str],
) -> None:
    """SQLite 테이블에 아직 없는 CSV 추가 컬럼을 TEXT 컬럼으로 생성한다."""
    if not column_names:
        return

    engine = get_engine(config)
    existing_column_names = {
        str(column["name"]) for column in inspect(engine).get_columns(table_name)
    }
    missing_column_names = [
        column_name for column_name in column_names if column_name not in existing_column_names
    ]
    if not missing_column_names:
        return

    escaped_table_name = _quote_sqlite_identifier(table_name)
    with engine.begin() as connection:
        for column_name in missing_column_names:
            escaped_column_name = _quote_sqlite_identifier(column_name)
            connection.exec_driver_sql(
                f'ALTER TABLE "{escaped_table_name}" ADD COLUMN "{escaped_column_name}" TEXT'
            )


def _reflect_sqlite_table(config: Settings, *, table_name: str) -> Table:
    """현재 SQLite 스키마를 기준으로 대상 테이블을 반사해 INSERT에 사용한다."""
    metadata = MetaData()
    return Table(table_name, metadata, autoload_with=get_engine(config))


def _insert_seed_rows(
    config: Settings,
    *,
    table_name: str,
    rows: Sequence[dict[str, SeedCellValue]],
) -> None:
    """반사된 테이블 스키마를 사용해 정적/동적 컬럼이 섞인 시드 행을 저장한다."""
    if not rows:
        return

    table = _reflect_sqlite_table(config, table_name=table_name)
    with session_scope(config) as session:
        session.execute(table.insert(), list(rows))


def _build_user_seed_rows(rows: Sequence[dict[str, str | None]]) -> list[dict[str, SeedCellValue]]:
    """회원 CSV 행 목록을 사용자 테이블 INSERT용 딕셔너리 목록으로 변환한다."""
    seed_rows: list[dict[str, SeedCellValue]] = []
    for row in rows:
        seed_row: dict[str, SeedCellValue] = {
            "id": int(row.get("id") or 0),
            "name": (row.get("name") or "").strip(),
            "age": _parse_int(row.get("age")),
            "job": _parse_text(row.get("직업")),
            "gender": _parse_text(row.get("성별")),
            "income": _parse_text(row.get("연봉")),
            "region": _parse_text(row.get("지역")),
            "card_grade": _parse_text(row.get("최상위 카드등급")),
            "persona": _parse_text(row.get("페르소나")),
        }
        for column_name, value in row.items():
            if column_name in USER_CSV_COLUMN_TO_DB_COLUMN or column_name == "":
                continue
            seed_row[column_name] = _parse_text(value)
        seed_rows.append(seed_row)
    return seed_rows


def _build_transaction_seed_rows(
    rows: Sequence[dict[str, str | None]],
) -> list[dict[str, SeedCellValue]]:
    """소비 CSV 행 목록을 거래 테이블 INSERT용 딕셔너리 목록으로 변환한다."""
    seed_rows: list[dict[str, SeedCellValue]] = []
    for row in rows:
        seed_row: dict[str, SeedCellValue] = {
            "id": int(row.get("id") or 0),
            "user_id": int(row.get("멤버 id") or 0),
            "amount": _parse_int(row.get("사용 금액")),
            "used_at": _parse_datetime(row.get("사용 시간")),
            "description": _parse_text(row.get("결제 내역")),
            "merchant_status": _parse_text(row.get("결제 장소 (가맹점 여부)")),
            "installment_flag": _parse_text(row.get("할부 여부")),
            "installment_months": _parse_int(row.get("할부 개월")),
            "installment_interest_type": _parse_text(row.get("할부 무/유이자 여부")),
            "transaction_status": _parse_text(row.get("거래 상태 (승인 / 취소)")),
            "is_overseas": _parse_text(row.get("해외 결제")),
            "category": _parse_text(row.get("업종 카테고리")),
            "payment_channel": _parse_text(row.get("결제 방식 (온/오프라인)")),
        }
        for column_name, value in row.items():
            if column_name in TRANSACTION_CSV_COLUMN_TO_DB_COLUMN or column_name == "":
                continue
            seed_row[column_name] = _parse_text(value)
        seed_rows.append(seed_row)
    return seed_rows


def _build_user_profile(user: UserModel) -> dict[str, int | str | None]:
    """로그인 이후 응답에 사용할 사용자 프로필 딕셔너리를 구성한다."""
    return {
        "name": user.name,
        "age": user.age,
        "job": user.job,
        "gender": user.gender,
        "income": user.income,
        "region": user.region,
        "card_grade": user.card_grade,
        "persona": user.persona,
    }


def _seed_users_if_empty(config: Settings) -> None:
    """회원 테이블이 비어 있을 때만 회원 CSV 데이터를 초기 적재한다."""
    members_path = config.members_csv_path
    if not members_path.exists():
        return

    with session_scope(config) as session:
        existing_count = session.scalar(select(func.count()).select_from(UserModel)) or 0
        if existing_count > 0:
            return

    seed_rows = _build_user_seed_rows(_iter_csv_rows(members_path))
    _ensure_dynamic_sqlite_columns(
        config,
        table_name=UserModel.__tablename__,
        column_names=_collect_dynamic_column_names(
            seed_rows,
            static_column_names=set(UserModel.__table__.columns.keys()),
        ),
    )
    _insert_seed_rows(config, table_name=UserModel.__tablename__, rows=seed_rows)


def _seed_transactions_if_empty(config: Settings) -> None:
    """거래 테이블이 비어 있을 때만 소비 CSV 데이터를 초기 적재한다."""
    consumption_path = config.consumption_csv_path
    if not consumption_path.exists():
        return

    with session_scope(config) as session:
        existing_count = session.scalar(select(func.count()).select_from(TransactionModel)) or 0
        if existing_count > 0:
            return

    seed_rows = _build_transaction_seed_rows(_iter_csv_rows(consumption_path))
    _ensure_dynamic_sqlite_columns(
        config,
        table_name=TransactionModel.__tablename__,
        column_names=_collect_dynamic_column_names(
            seed_rows,
            static_column_names=set(TransactionModel.__table__.columns.keys()),
        ),
    )
    _insert_seed_rows(config, table_name=TransactionModel.__tablename__, rows=seed_rows)


def ensure_user_database(settings: Settings | None = None) -> DatabaseSeedResult:
    """CSV 시그니처를 기준으로 사용자용 SQLite DB를 준비하고 기본 데이터를 보장한다."""
    config = settings or get_settings()
    csv_signature = _build_csv_signature(config)
    if _should_rebuild_sqlite_database(config, csv_signature):
        _reset_sqlite_database(config)

    create_database_tables(config)
    _seed_users_if_empty(config)
    _seed_transactions_if_empty(config)
    _write_seed_metadata(config, csv_signature)

    with session_scope(config) as session:
        user_count = session.scalar(select(func.count()).select_from(UserModel)) or 0
        transaction_count = session.scalar(select(func.count()).select_from(TransactionModel)) or 0
        memory_count = session.scalar(select(func.count()).select_from(UserMemoryModel)) or 0

    return DatabaseSeedResult(
        user_count=user_count,
        transaction_count=transaction_count,
        memory_count=memory_count,
        sqlite_db_path=str(config.sqlite_db_path),
    )


def authenticate_user(
    user_id: int,
    name: str,
    *,
    settings: Settings | None = None,
) -> dict[str, int | str | None] | None:
    """사용자 ID와 이름이 모두 일치하는 회원을 조회해 프로필을 반환한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        user = session.scalar(
            select(UserModel).where(
                UserModel.id == user_id,
                UserModel.name == name.strip(),
            )
        )

    if user is None:
        return None
    return _build_user_profile(user)


def get_user_transactions(
    user_id: int,
    *,
    settings: Settings | None = None,
) -> list[dict[str, int | str | None]]:
    """사용자 거래 내역을 시간순으로 조회해 직렬화 가능한 형태로 반환한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        transactions = session.scalars(
            select(TransactionModel)
            .where(TransactionModel.user_id == user_id)
            .order_by(TransactionModel.used_at.asc(), TransactionModel.id.asc())
        ).all()

    return [
        {
            "id": item.id,
            "user_id": item.user_id,
            "amount": item.amount,
            "used_at": item.used_at.isoformat(sep=" ") if item.used_at else None,
            "description": item.description,
            "category": item.category,
            "transaction_status": item.transaction_status,
        }
        for item in transactions
    ]


def save_user_memory(
    *,
    user_id: int,
    memory_key: str,
    content: str,
    settings: Settings | None = None,
) -> dict[str, int | str]:
    """사용자 메모를 키 기준으로 생성하거나 기존 내용을 덮어쓴다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        memory = session.scalar(
            select(UserMemoryModel).where(
                UserMemoryModel.user_id == user_id,
                UserMemoryModel.memory_key == memory_key,
            )
        )
        if memory is None:
            memory = UserMemoryModel(
                user_id=user_id,
                memory_key=memory_key,
                content=content,
            )
            session.add(memory)
            session.flush()
        else:
            memory.content = content
            session.flush()

        return {
            "id": memory.id,
            "user_id": memory.user_id,
            "memory_key": memory.memory_key,
            "content": memory.content,
        }


def list_user_memories(
    user_id: int,
    *,
    settings: Settings | None = None,
) -> list[dict[str, int | str]]:
    """사용자 메모를 최신 수정 순으로 조회해 직렬화 가능한 목록으로 반환한다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        memories = session.scalars(
            select(UserMemoryModel)
            .where(UserMemoryModel.user_id == user_id)
            .order_by(UserMemoryModel.updated_at.desc(), UserMemoryModel.id.desc())
        ).all()

    return [
        {
            "id": memory.id,
            "user_id": memory.user_id,
            "memory_key": memory.memory_key,
            "content": memory.content,
        }
        for memory in memories
    ]
