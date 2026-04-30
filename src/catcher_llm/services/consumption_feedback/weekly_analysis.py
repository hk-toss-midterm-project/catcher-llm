from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import select

from catcher_llm.analysis.user_weekly_analysis import build_weekly_consumption_analysis_from_frames
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import TransactionModel
from catcher_llm.db.session import session_scope
from catcher_llm.schemas.consumption_feedback import JsonObject
from catcher_llm.services.user_data_service import ensure_user_database


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _transaction_rows_to_frame(transactions: list[TransactionModel]) -> pd.DataFrame:
    """SQLite 거래 모델 목록을 주간 소비 분석 함수가 기대하는 DataFrame으로 변환한다."""
    rows = [
        {
            "멤버 id": t.user_id,
            "id": t.id,
            "사용 금액": t.amount or 0,
            "사용 시간": t.used_at,
            "결제 내역": t.merchant_name or t.description or "",
            "업종 카테고리": t.category or "",
            # 주간 분석에는 결제 방식이 필요하지 않으나 DataFrame 통일성을 위해 포함
            "결제 방식 (온/오프라인)": t.payment_channel or "",
        }
        for t in transactions
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
            "결제 방식 (온/오프라인)",
        ],
    )


def _load_all_transaction_frame(
    *,
    member_id: int,
    settings: Settings,
) -> pd.DataFrame:
    """SQLite에서 멤버의 전체 거래 내역을 시간순으로 조회해 DataFrame으로 반환한다."""
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
    return _transaction_rows_to_frame(transactions)


def build_weekly_consumption_analysis_json(
    *,
    member_id: int = 1,
    week_start: str | date = "2024-04-01",
    week_end: str | date = "2024-04-07",
    settings: Settings | None = None,
) -> JsonObject:
    """SQLite 거래 테이블에서 유저 거래를 읽어 주간 소비 분석 JSON 객체를 만든다.

    Parameters
    ----------
    member_id:
        분석 대상 멤버 ID.
    week_start:
        분석할 주의 시작일 (YYYY-MM-DD 또는 date 객체).
    week_end:
        분석할 주의 종료일 (YYYY-MM-DD 또는 date 객체).
    settings:
        앱 설정 (None이면 기본 설정을 사용).

    Returns
    -------
    JsonObject
        주간 소비 분석 결과 딕셔너리.
    """
    config = settings or get_settings()
    all_frame = _load_all_transaction_frame(member_id=member_id, settings=config)

    return build_weekly_consumption_analysis_from_frames(
        all_frame,
        member_id=member_id,
        week_start=week_start,
        week_end=week_end,
        source_path=config.sqlite_db_path,
    )
