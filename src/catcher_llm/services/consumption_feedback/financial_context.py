from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import MetaData, Table, select
from sqlalchemy.engine import RowMapping

from catcher_llm.config.settings import Settings
from catcher_llm.db.models import UserModel
from catcher_llm.db.session import get_engine, session_scope
from catcher_llm.services.user_data_service import ensure_user_database

_INCOME_COLUMN = "income"
_TARGET_MAX_SPENDING_COLUMN = "target_max_spending_amount"


@dataclass(frozen=True, slots=True)
class UserFinancialContext:
    """소비 분석에 주입할 사용자 연소득과 목표 소비 한도를 표현한다."""

    annual_income: float | None = None
    monthly_income: float | None = None
    target_max_spending_amount: float | None = None


def _parse_positive_amount(value: object) -> float | None:
    """DB에서 읽은 숫자형 문자열을 양수 금액으로 변환한다."""
    if value is None:
        return None

    if isinstance(value, int | float):
        amount = float(value)
    else:
        normalized = str(value).strip().replace(",", "")
        if not normalized:
            return None
        try:
            amount = float(normalized)
        except ValueError:
            return None

    if amount <= 0:
        return None
    return amount


def _read_optional_row_value(row: RowMapping, column_name: str) -> object | None:
    """동적 컬럼이 없을 수 있는 RowMapping에서 값을 안전하게 읽는다."""
    if column_name not in row:
        return None
    return row[column_name]


def load_user_financial_context(
    *,
    member_id: int,
    settings: Settings,
) -> UserFinancialContext:
    """SQLite users 테이블에서 소비 분석용 연소득과 월 목표 소비 한도를 조회한다."""
    ensure_user_database(settings)
    users_table = Table(
        UserModel.__tablename__,
        MetaData(),
        autoload_with=get_engine(settings),
    )

    with session_scope(settings) as session:
        row = (
            session.execute(select(users_table).where(users_table.c["id"] == member_id))
            .mappings()
            .first()
        )

    if row is None:
        return UserFinancialContext()

    annual_income = _parse_positive_amount(_read_optional_row_value(row, _INCOME_COLUMN))
    monthly_income = annual_income / 12 if annual_income is not None else None
    target_max_spending_amount = _parse_positive_amount(
        _read_optional_row_value(row, _TARGET_MAX_SPENDING_COLUMN)
    )

    return UserFinancialContext(
        annual_income=annual_income,
        monthly_income=monthly_income,
        target_max_spending_amount=target_max_spending_amount,
    )
