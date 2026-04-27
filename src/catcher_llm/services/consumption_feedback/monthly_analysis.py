from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from catcher_llm.analysis.user_monthly_analysis import (
    build_monthly_consumption_analysis_from_frames,
)
from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import TransactionModel
from catcher_llm.db.session import session_scope
from catcher_llm.schemas.consumption_feedback import JsonObject
from catcher_llm.services.user_data_service import ensure_user_database


def _transaction_rows_to_frame(transactions: list[TransactionModel]) -> pd.DataFrame:
    """SQLite 거래 모델 목록을 월간 소비 분석 함수가 기대하는 DataFrame으로 변환한다."""
    rows = [
        {
            "멤버 id": t.user_id,
            "id": t.id,
            "사용 금액": t.amount or 0,
            "사용 시간": t.used_at,
            "결제 내역": t.description or "",
            "업종 카테고리": t.category or "",
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


def build_monthly_consumption_analysis_json(
    *,
    member_id: int = 1,
    analysis_month: str = "2024-04",
    settings: Settings | None = None,
) -> JsonObject:
    """SQLite 거래 테이블에서 유저 거래를 읽어 월간 소비 분석 JSON 객체를 만든다.

    Parameters
    ----------
    member_id:
        분석 대상 멤버 ID.
    analysis_month:
        분석할 연-월 (``'YYYY-MM'`` 형식).
    settings:
        앱 설정 (None이면 기본 설정을 사용).

    Returns
    -------
    JsonObject
        월간 소비 분석 결과 딕셔너리.
    """
    config = settings or get_settings()
    all_frame = _load_all_transaction_frame(member_id=member_id, settings=config)

    return build_monthly_consumption_analysis_from_frames(
        all_frame,
        member_id=member_id,
        analysis_month=analysis_month,
        source_path=config.sqlite_db_path,
    )
