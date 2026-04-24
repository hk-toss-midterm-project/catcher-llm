from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import TransactionModel, UserMemoryModel, UserModel
from catcher_llm.db.session import create_database_tables, session_scope


@dataclass(slots=True, frozen=True)
class DatabaseSeedResult:
    user_count: int
    transaction_count: int
    memory_count: int
    sqlite_db_path: str


def _iter_csv_rows(path: Path) -> list[dict[str, str | None]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is not None:
            reader.fieldnames = [field.strip() if field else "" for field in reader.fieldnames]
        return [
            {(key.strip() if key else ""): value for key, value in row.items()} for row in reader
        ]


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return int(stripped)


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return datetime.fromisoformat(stripped)


def _build_user_profile(user: UserModel) -> dict[str, int | str | None]:
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
    members_path = config.members_csv_path
    if not members_path.exists():
        return

    with session_scope(config) as session:
        existing_count = session.scalar(select(func.count()).select_from(UserModel)) or 0
        if existing_count > 0:
            return

        session.add_all(
            [
                UserModel(
                    id=int(row["id"]),
                    name=(row.get("name") or "").strip(),
                    age=_parse_int(row.get("age")),
                    job=row.get("직업"),
                    gender=row.get("성별"),
                    income=row.get("연봉"),
                    region=row.get("지역"),
                    card_grade=row.get("최상위 카드등급"),
                    persona=row.get("페르소나"),
                )
                for row in _iter_csv_rows(members_path)
            ]
        )


def _seed_transactions_if_empty(config: Settings) -> None:
    consumption_path = config.consumption_csv_path
    if not consumption_path.exists():
        return

    with session_scope(config) as session:
        existing_count = session.scalar(select(func.count()).select_from(TransactionModel)) or 0
        if existing_count > 0:
            return

        session.add_all(
            [
                TransactionModel(
                    id=int(row["id"]),
                    user_id=int(row["멤버 id"]),
                    amount=_parse_int(row.get("사용 금액")),
                    used_at=_parse_datetime(row.get("사용 시간")),
                    description=row.get("결제 내역"),
                    merchant_status=row.get("결제 장소 (가맹점 여부)"),
                    installment_flag=row.get("할부 여부"),
                    installment_months=_parse_int(row.get("할부 개월")),
                    installment_interest_type=row.get("할부 무/유이자 여부"),
                    transaction_status=row.get("거래 상태 (승인 / 취소)"),
                    is_overseas=row.get("해외 결제"),
                    category=row.get("업종 카테고리"),
                    payment_channel=row.get("결제 방식 (온/오프라인)"),
                )
                for row in _iter_csv_rows(consumption_path)
            ]
        )


def ensure_user_database(settings: Settings | None = None) -> DatabaseSeedResult:
    config = settings or get_settings()
    create_database_tables(config)
    _seed_users_if_empty(config)
    _seed_transactions_if_empty(config)

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
