from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import select

from catcher_llm.analysis.user_daily_analysis import build_daily_consumption_analysis_from_frames
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import TransactionModel
from catcher_llm.db.session import session_scope
from catcher_llm.schemas.consumption_feedback import JsonObject
from catcher_llm.services.user_data_service import ensure_user_database


def _parse_analysis_date(value: str | date) -> date:
    """문자열 또는 date 입력을 일 단위 분석 기준일로 변환한다."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _transaction_rows_to_frame(transactions: list[TransactionModel]) -> pd.DataFrame:
    """SQLite 거래 모델 목록을 일일 소비 분석 함수가 기대하는 DataFrame으로 변환한다."""
    rows = [
        {
            "멤버 id": transaction.user_id,
            "id": transaction.id,
            "사용 금액": transaction.amount or 0,
            "사용 시간": transaction.used_at,
            "결제 내역": transaction.description or "",
            "업종 카테고리": transaction.category or "",
        }
        for transaction in transactions
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "멤버 id",
            "id",
            "사용 금액",
            "사용 시간",
            "결제 내역",
            "업종 카테고리",
        ],
    )


def _load_user_transaction_frames(
    *,
    member_id: int,
    analysis_day: date,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SQLite 거래 테이블에서 유저의 과거 거래와 분석일 거래 DataFrame을 조회한다."""
    ensure_user_database(settings)
    with session_scope(settings) as session:
        transactions = list(
            session.scalars(
                select(TransactionModel)
                .where(
                    TransactionModel.user_id == member_id,
                    TransactionModel.used_at.is_not(None),
                )
                .order_by(TransactionModel.used_at.asc(), TransactionModel.id.asc())
            )
        )

    past_transactions = [
        transaction
        for transaction in transactions
        if transaction.used_at is not None and transaction.used_at.date() < analysis_day
    ]
    today_transactions = [
        transaction
        for transaction in transactions
        if transaction.used_at is not None and transaction.used_at.date() == analysis_day
    ]
    return (
        _transaction_rows_to_frame(past_transactions),
        _transaction_rows_to_frame(today_transactions),
    )


def build_daily_consumption_analysis_json(
    *,
    member_id: int = 1,
    analysis_date: str | date = "2024-04-01",
    previous_date: str | date = "2024-03-31",
    settings: Settings | None = None,
) -> JsonObject:
    """SQLite 거래 테이블에서 유저 거래를 읽어 일일 소비 분석 JSON 객체를 만든다."""
    config = settings or get_settings()
    analysis_day = _parse_analysis_date(analysis_date)
    past_frame, today_frame = _load_user_transaction_frames(
        member_id=member_id,
        analysis_day=analysis_day,
        settings=config,
    )

    return build_daily_consumption_analysis_from_frames(
        past_frame,
        today_frame,
        member_id=member_id,
        analysis_date=analysis_day,
        previous_date=previous_date,
        past_source_path=config.sqlite_db_path,
        today_source_path=config.sqlite_db_path,
    )
