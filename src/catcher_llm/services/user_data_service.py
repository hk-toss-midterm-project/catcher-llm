from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData, Table, func, inspect, select

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import (
    TRANSACTION_CSV_COLUMN_TO_DB_COLUMN,
    USER_CSV_COLUMN_TO_DB_COLUMN,
    SessionModel,
    TransactionModel,
    UserFeedbackMemoryModel,
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
_SQLITE_SEED_SCHEMA_VERSION = 5
_SESSION_FEEDBACK_REACTION_COLUMN_NAMES = (
    "feedback_reaction",
    "feedback_reaction_reason",
)
_TRANSACTION_MERCHANT_NAME_COLUMN_NAME = "merchant_name"

# 주기별 메모리 만료 기간 (일 단위)
_MEMORY_EXPIRY_DAYS: dict[str, int] = {
    "daily": 7,
    "weekly": 28,  # 4주
    "monthly": 365,  # 1년
}
_USER_REGISTRATION_COLUMNS: tuple[str, ...] = (
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
)
_USER_REGISTRATION_DYNAMIC_COLUMN_NAMES: tuple[str, ...] = (
    "personal_score",
    "target_max_spending_amount",
)
type SeedCellValue = int | str | datetime | None

_USER_SEED_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id",),
    "name": ("name",),
    "age": ("age",),
    "job": ("직업", "occupation", "job"),
    "gender": ("성별", "gender"),
    "income": ("연봉", "annual_income", "income"),
    "region": ("지역", "region"),
    "card_grade": ("최상위 카드등급", "card_grade"),
    "persona": ("페르소나", "persona"),
    "saving_goal_text": ("saving_goal_text",),
}
_TRANSACTION_SEED_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id",),
    "user_id": ("멤버 id", "user_id"),
    "amount": ("사용 금액", "amount"),
    "used_at": ("사용 시간", "transaction_time", "used_at"),
    "description": ("결제 내역", "description"),
    "merchant_name": ("merchant_name", "가맹점명"),
    "merchant_status": ("결제 장소 (가맹점 여부)",),
    "installment_flag": ("할부 여부", "is_installment"),
    "installment_months": ("할부 개월", "installment_months"),
    "installment_interest_type": ("할부 무/유이자 여부", "is_interest_free"),
    "transaction_status": ("거래 상태 (승인 / 취소)", "status"),
    "is_overseas": ("해외 결제", "is_overseas"),
    "category": ("업종 카테고리", "category"),
    "payment_channel": ("결제 방식 (온/오프라인)", "payment_channel"),
}


@dataclass(slots=True, frozen=True)
class DatabaseSeedResult:
    user_count: int
    transaction_count: int
    memory_count: int
    sqlite_db_path: str


@dataclass(slots=True, frozen=True)
class UserRegistrationInput:
    """회원가입 폼에서 받은 사용자 입력값을 표현한다."""

    name: str
    age: int
    occupation: str
    gender: str
    annual_income: int
    region: str
    persona: str
    saving_goal_text: str
    target_max_spending_amount: int


@dataclass(slots=True, frozen=True)
class UserRegistrationResult:
    """회원가입 저장 결과와 자동 발급된 사용자 ID를 표현한다."""

    user_id: int
    name: str
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


def _read_csv_header(path: Path) -> list[str]:
    """BOM과 공백이 섞인 CSV 헤더를 정리해 순서대로 반환한다."""
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []
    return [column_name.strip() for column_name in header]


def _parse_int(value: str | None) -> int | None:
    """비어 있을 수 있는 문자열 정수를 정리해 `int` 또는 `None`으로 변환한다."""
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return int(stripped)


def _pick_csv_value(row: dict[str, str | None], aliases: Sequence[str]) -> str | None:
    """CSV 행에서 v1 한글 헤더와 v3 영문 헤더 alias를 순서대로 찾아 값을 반환한다."""
    for column_name in aliases:
        if column_name in row:
            return row[column_name]
    return None


def _parse_required_int_from_aliases(
    row: dict[str, str | None],
    aliases: Sequence[str],
    *,
    field_name: str,
) -> int:
    """필수 정수 컬럼을 alias 목록에서 찾아 파싱하고 없으면 명확한 오류를 낸다."""
    value = _pick_csv_value(row, aliases)
    parsed_value = _parse_int(value)
    if parsed_value is None:
        alias_text = ", ".join(aliases)
        raise ValueError(f"{field_name} 값을 찾을 수 없습니다. CSV 컬럼 후보: {alias_text}")
    return parsed_value


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


def _normalize_required_text(value: str, *, field_label: str) -> str:
    """회원가입 필수 문자열 값을 공백 제거 후 검증해 반환한다."""
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{field_label} 값을 입력해주세요.")
    return stripped


def _validate_int_range(
    value: int,
    *,
    field_label: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """회원가입 숫자 값이 허용 범위 안에 있는지 검증한다."""
    if value < minimum:
        raise ValueError(f"{field_label} 값은 {minimum} 이상이어야 합니다.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_label} 값은 {maximum} 이하여야 합니다.")
    return value


def _ensure_registration_csv_columns(fieldnames: Sequence[str]) -> None:
    """회원가입 저장에 필요한 v3 사용자 CSV 컬럼이 모두 있는지 검증한다."""
    missing_columns = [
        column_name for column_name in _USER_REGISTRATION_COLUMNS if column_name not in fieldnames
    ]
    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise ValueError(f"회원가입 CSV에 필요한 컬럼이 없습니다: {joined_columns}")


def _get_next_sqlite_user_id(config: Settings) -> int:
    """SQLite users 테이블에서 가장 큰 ID 다음 값을 회원가입 ID로 계산한다."""
    with session_scope(config) as session:
        max_user_id = session.scalar(select(func.max(UserModel.id))) or 0
    return int(max_user_id) + 1


def _build_user_registration_db_row(
    payload: UserRegistrationInput,
    *,
    user_id: int,
) -> dict[str, SeedCellValue]:
    """회원가입 입력값을 SQLite users 테이블에 저장할 행으로 변환한다."""
    age = _validate_int_range(payload.age, field_label="나이", minimum=0, maximum=130)
    annual_income = _validate_int_range(
        payload.annual_income,
        field_label="연소득",
        minimum=0,
    )
    target_max_spending_amount = _validate_int_range(
        payload.target_max_spending_amount,
        field_label="목표 최대 소비 금액",
        minimum=0,
    )

    return {
        "id": user_id,
        "name": _normalize_required_text(payload.name, field_label="이름"),
        "age": age,
        "job": _normalize_required_text(payload.occupation, field_label="직업"),
        "gender": _normalize_required_text(payload.gender, field_label="성별"),
        "income": str(annual_income),
        "region": _normalize_required_text(payload.region, field_label="지역"),
        "persona": _normalize_required_text(payload.persona, field_label="페르소나"),
        "personal_score": "0",
        "saving_goal_text": _normalize_required_text(
            payload.saving_goal_text,
            field_label="절약 목표",
        ),
        "target_max_spending_amount": str(target_max_spending_amount),
    }


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
    """SQLite 테이블에 아직 없는 동적 TEXT 컬럼을 생성한다."""
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


def _ensure_session_feedback_reaction_columns(config: Settings) -> None:
    """기존 세션 테이블에 피드백 반응 저장 컬럼을 보강한다."""
    _ensure_dynamic_sqlite_columns(
        config,
        table_name=SessionModel.__tablename__,
        column_names=_SESSION_FEEDBACK_REACTION_COLUMN_NAMES,
    )


def _ensure_user_feedback_memories_table(config: Settings) -> None:
    """user_feedback_memories 테이블이 없으면 생성하고, 기존 컬럼 데이터를 마이그레이션한다."""
    from sqlalchemy import text

    engine = get_engine(config)

    # 테이블 존재 여부 확인 후 CREATE
    with engine.connect() as conn:
        existing_tables = inspect(engine).get_table_names()
        if UserFeedbackMemoryModel.__tablename__ not in existing_tables:
            UserFeedbackMemoryModel.__table__.create(engine, checkfirst=True)

        # 구 컬럼이 아직 user_memories에 남아있으면 데이터 마이그레이션 후 제거
        cols = [c["name"] for c in inspect(engine).get_columns("user_memories")]
        if "user_feedback_memory" in cols:
            _migrate_feedback_memory_column_to_table(config)
            # SQLite 3.35+ DROP COLUMN 지원
            try:
                conn.execute(text("ALTER TABLE user_memories DROP COLUMN user_feedback_memory"))
                conn.commit()
            except Exception:
                pass  # 이미 제거됐거나 구 SQLite 버전일 때 무시


def _migrate_feedback_memory_column_to_table(config: Settings) -> None:
    """user_memories.user_feedback_memory 텍스트를 user_feedback_memories 개별 행으로 이전한다."""
    import re

    def _extract_reason(line: str) -> str:
        line = line.strip()
        match = re.search(r"—\s*이유:\s*(.+)", line)
        if match:
            return match.group(1).strip()
        return re.sub(r"^\[거부/제약\]\s*", "", line).strip()

    with session_scope(config) as session:
        from sqlalchemy import text as sa_text

        rows = list(
            session.execute(
                sa_text(
                    "SELECT id, user_id, period_type, user_feedback_memory FROM user_memories"
                    " WHERE user_feedback_memory IS NOT NULL AND user_feedback_memory != ''"
                )
            )
        )
        now = datetime.now(UTC)
        for row in rows:
            raw: str = row[3] or ""
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            seen: set[str] = set()
            for line in lines:
                reason = _extract_reason(line)
                if reason and reason not in seen:
                    seen.add(reason)
                    session.add(
                        UserFeedbackMemoryModel(
                            user_id=row[1],
                            period_type=row[2],
                            reason=reason,
                            created_at=now,
                        )
                    )


def _ensure_transaction_merchant_name_column(config: Settings) -> None:
    """기존 거래 테이블에 실제 가맹점명 컬럼을 보강한다."""
    _ensure_dynamic_sqlite_columns(
        config,
        table_name=TransactionModel.__tablename__,
        column_names=(_TRANSACTION_MERCHANT_NAME_COLUMN_NAME,),
    )


def reset_expired_user_memories(settings: Settings | None = None) -> dict[str, int]:
    """만료된 user_memories 요약과 user_feedback_memories 항목을 오래된 순으로 초기화한다.

    주기별 만료 기간:
    - daily  : 7일
    - weekly : 28일 (4주)
    - monthly: 365일 (1년)

    Returns:
        period_type별 초기화된 user_memories 행 수 딕셔너리
    """
    config = settings or get_settings()
    now = datetime.now(UTC)
    reset_counts: dict[str, int] = {}

    with session_scope(config) as session:
        for period_type, days in _MEMORY_EXPIRY_DAYS.items():
            cutoff = now - timedelta(days=days)

            # 1) user_memories.summary 초기화 (오래된 것부터)
            expired_memories = list(
                session.scalars(
                    select(UserMemoryModel)
                    .where(
                        UserMemoryModel.period_type == period_type,
                        UserMemoryModel.updated_at < cutoff,
                    )
                    .order_by(UserMemoryModel.updated_at.asc())
                )
            )
            for mem in expired_memories:
                mem.summary = ""
                mem.updated_at = now
            reset_counts[period_type] = len(expired_memories)

            # 2) user_feedback_memories 개별 항목 삭제 (오래된 것부터)
            expired_fb = list(
                session.scalars(
                    select(UserFeedbackMemoryModel)
                    .where(
                        UserFeedbackMemoryModel.period_type == period_type,
                        UserFeedbackMemoryModel.created_at < cutoff,
                    )
                    .order_by(UserFeedbackMemoryModel.created_at.asc())
                )
            )
            for fb in expired_fb:
                session.delete(fb)

    return reset_counts


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
            "id": _parse_required_int_from_aliases(
                row,
                _USER_SEED_COLUMN_ALIASES["id"],
                field_name="사용자 ID",
            ),
            "name": (_pick_csv_value(row, _USER_SEED_COLUMN_ALIASES["name"]) or "").strip(),
            "age": _parse_int(_pick_csv_value(row, _USER_SEED_COLUMN_ALIASES["age"])),
            "job": _parse_text(_pick_csv_value(row, _USER_SEED_COLUMN_ALIASES["job"])),
            "gender": _parse_text(_pick_csv_value(row, _USER_SEED_COLUMN_ALIASES["gender"])),
            "income": _parse_text(_pick_csv_value(row, _USER_SEED_COLUMN_ALIASES["income"])),
            "region": _parse_text(_pick_csv_value(row, _USER_SEED_COLUMN_ALIASES["region"])),
            "card_grade": _parse_text(
                _pick_csv_value(row, _USER_SEED_COLUMN_ALIASES["card_grade"])
            ),
            "persona": _parse_text(_pick_csv_value(row, _USER_SEED_COLUMN_ALIASES["persona"])),
            "saving_goal_text": _parse_text(
                _pick_csv_value(row, _USER_SEED_COLUMN_ALIASES["saving_goal_text"])
            ),
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
            "id": _parse_required_int_from_aliases(
                row,
                _TRANSACTION_SEED_COLUMN_ALIASES["id"],
                field_name="거래 ID",
            ),
            "user_id": _parse_required_int_from_aliases(
                row,
                _TRANSACTION_SEED_COLUMN_ALIASES["user_id"],
                field_name="거래 사용자 ID",
            ),
            "amount": _parse_int(_pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["amount"])),
            "used_at": _parse_datetime(
                _pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["used_at"])
            ),
            "description": _parse_text(
                _pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["description"])
            ),
            "merchant_name": _parse_text(
                _pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["merchant_name"])
            ),
            "merchant_status": _parse_text(
                _pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["merchant_status"])
            ),
            "installment_flag": _parse_text(
                _pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["installment_flag"])
            ),
            "installment_months": _parse_int(
                _pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["installment_months"])
            ),
            "installment_interest_type": _parse_text(
                _pick_csv_value(
                    row,
                    _TRANSACTION_SEED_COLUMN_ALIASES["installment_interest_type"],
                )
            ),
            "transaction_status": _parse_text(
                _pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["transaction_status"])
            ),
            "is_overseas": _parse_text(
                _pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["is_overseas"])
            ),
            "category": _parse_text(
                _pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["category"])
            ),
            "payment_channel": _parse_text(
                _pick_csv_value(row, _TRANSACTION_SEED_COLUMN_ALIASES["payment_channel"])
            ),
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
        "personal_score": user.personal_score,
        "saving_goal_text": user.saving_goal_text,
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
    _ensure_session_feedback_reaction_columns(config)
    _ensure_transaction_merchant_name_column(config)
    _ensure_user_feedback_memories_table(config)
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


def get_user_registration_columns(settings: Settings | None = None) -> list[str]:
    """회원가입 화면 구성을 위해 현재 사용자 CSV 헤더 목록을 반환한다."""
    config = settings or get_settings()
    columns = _read_csv_header(config.members_csv_path)
    if not columns:
        raise ValueError(f"사용자 CSV 헤더를 읽을 수 없습니다: {config.members_csv_path}")
    _ensure_registration_csv_columns(columns)
    return columns


def register_user(
    payload: UserRegistrationInput,
    *,
    settings: Settings | None = None,
) -> UserRegistrationResult:
    """회원가입 입력값을 SQLite users 테이블에 새 사용자로 저장한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)
    user_id = _get_next_sqlite_user_id(config)
    db_row = _build_user_registration_db_row(payload, user_id=user_id)
    _ensure_dynamic_sqlite_columns(
        config,
        table_name=UserModel.__tablename__,
        column_names=_USER_REGISTRATION_DYNAMIC_COLUMN_NAMES,
    )
    users_table = _reflect_sqlite_table(config, table_name=UserModel.__tablename__)
    with session_scope(config) as session:
        session.execute(users_table.insert(), [db_row])

    return UserRegistrationResult(
        user_id=user_id,
        name=str(db_row["name"]),
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
    period_type: str,
    summary: str,
    settings: Settings | None = None,
) -> dict[str, int | str]:
    """사용자 메모리를 기간 유형 기준으로 생성하거나 기존 요약을 덮어쓴다."""
    config = settings or get_settings()
    ensure_user_database(config)

    with session_scope(config) as session:
        memory = session.scalar(
            select(UserMemoryModel).where(
                UserMemoryModel.user_id == user_id,
                UserMemoryModel.period_type == period_type,
            )
        )
        if memory is None:
            memory = UserMemoryModel(
                user_id=user_id,
                period_type=period_type,
                summary=summary,
            )
            session.add(memory)
            session.flush()
        else:
            memory.summary = summary
            session.flush()

        return {
            "id": memory.id,
            "user_id": memory.user_id,
            "period_type": memory.period_type,
            "summary": memory.summary,
        }


def list_user_memories(
    user_id: int,
    *,
    settings: Settings | None = None,
) -> list[dict[str, int | str]]:
    """사용자 메모리 요약을 최신 수정 순으로 조회해 직렬화 가능한 목록으로 반환한다."""
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
            "period_type": memory.period_type,
            "summary": memory.summary,
        }
        for memory in memories
    ]
