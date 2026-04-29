from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import select

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import TransactionModel
from catcher_llm.db.session import session_scope
from catcher_llm.services.user_data_service import ensure_user_database


@dataclass(frozen=True, slots=True)
class DailyReportSelection:
    """일간 보고서 실행에 사용할 기본 회원과 날짜 조합을 담는다."""

    member_id: int
    analysis_date: date
    previous_date: date


def get_default_daily_report_selection(settings: Settings | None = None) -> DailyReportSelection:
    """SQLite 거래 내역에서 과거 비교가 가능한 첫 회원·분석일 기본값을 찾는다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)

    fallback_analysis_date = date.today()
    fallback_selection = DailyReportSelection(
        member_id=1,
        analysis_date=fallback_analysis_date,
        previous_date=fallback_analysis_date - timedelta(days=1),
    )

    statement = (
        select(TransactionModel.user_id, TransactionModel.used_at)
        .where(TransactionModel.used_at.is_not(None))
        .order_by(TransactionModel.user_id.asc(), TransactionModel.used_at.asc())
    )

    first_transaction_date_by_user: dict[int, date] = {}
    with session_scope(config) as session:
        rows = session.execute(statement).all()

    for raw_user_id, raw_used_at in rows:
        if not isinstance(raw_user_id, int) or not isinstance(raw_used_at, datetime):
            continue

        transaction_date = raw_used_at.date()
        first_transaction_date = first_transaction_date_by_user.get(raw_user_id)
        if first_transaction_date is None:
            first_transaction_date_by_user[raw_user_id] = transaction_date
            continue

        if transaction_date > first_transaction_date:
            return DailyReportSelection(
                member_id=raw_user_id,
                analysis_date=transaction_date,
                previous_date=transaction_date - timedelta(days=1),
            )

    return fallback_selection
