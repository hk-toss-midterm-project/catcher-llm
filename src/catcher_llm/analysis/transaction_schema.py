from __future__ import annotations

import pandas as pd

_TRANSACTION_COLUMN_ALIASES: dict[str, str] = {
    "user_id": "멤버 id",
    "amount": "사용 금액",
    "transaction_time": "사용 시간",
    "used_at": "사용 시간",
    "description": "결제 내역",
    "category": "업종 카테고리",
    "payment_channel": "결제 방식 (온/오프라인)",
}
_PAYMENT_CHANNEL_ALIASES: dict[str, str] = {
    "ONLINE": "온라인",
    "OFFLINE": "오프라인",
}


def normalize_transaction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """v1 한글 거래 CSV와 v3 영문 거래 CSV 컬럼을 분석 함수 표준 컬럼으로 정규화한다."""
    normalized_frame = frame.copy()
    rename_columns = {
        source_column: target_column
        for source_column, target_column in _TRANSACTION_COLUMN_ALIASES.items()
        if source_column in normalized_frame.columns
        and target_column not in normalized_frame.columns
    }
    if rename_columns:
        normalized_frame = normalized_frame.rename(columns=rename_columns)

    payment_column = "결제 방식 (온/오프라인)"
    if payment_column in normalized_frame.columns:
        normalized_frame[payment_column] = (
            normalized_frame[payment_column]
            .astype("string")
            .replace(_PAYMENT_CHANNEL_ALIASES)
            .fillna("")
        )
    return normalized_frame
