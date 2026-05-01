from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import TransactionModel, UserModel
from catcher_llm.db.session import session_scope
from catcher_llm.llm.models import get_chat_model
from catcher_llm.services.user_data_service import ensure_user_database

_UPLOAD_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "used_at": (
        "transaction_time",
        "used_at",
        "사용 시간",
        "거래 일시",
        "거래일시",
        "결제 일시",
        "결제일시",
        "승인 일시",
        "승인일시",
        "이용일",
        "이용 일",
        "이용일자",
        "이용 일자",
        "이용일시",
        "이용 일시",
        "거래 일자",
        "거래일자",
        "결제 일자",
        "결제일자",
        "날짜",
        "일시",
    ),
    "merchant_name": (
        "merchant_name",
        "가맹점명",
        "가맹점 명",
        "가맹점",
        "상호명",
        "매장명",
        "사용처",
        "결제처",
        "결제 내역",
        "결제내역",
    ),
    "amount": (
        "amount",
        "사용 금액",
        "사용금액",
        "결제 금액",
        "결제금액",
        "승인 금액",
        "승인금액",
        "금액",
    ),
    "description": ("description", "상세내역", "상세 내역", "내용", "메모", "상품명"),
    "category": ("category", "업종 카테고리", "업종카테고리", "카테고리", "업종"),
    "payment_channel": (
        "payment_channel",
        "결제 방식",
        "결제방식",
        "온오프라인",
        "온라인오프라인",
        "채널",
    ),
    "transaction_status": (
        "status",
        "transaction_status",
        "승인상태",
        "승인 상태",
        "거래상태",
        "거래 상태",
    ),
    "merchant_status": ("merchant_status", "가맹점 여부", "가맹점여부"),
    "installment_flag": ("is_installment", "할부 여부", "할부여부"),
    "installment_months": ("installment_months", "할부 개월", "할부개월"),
    "installment_interest_type": (
        "is_interest_free",
        "할부 무/유이자 여부",
        "할부무유이자여부",
        "무이자 여부",
        "무이자여부",
    ),
    "is_overseas": ("is_overseas", "해외 결제", "해외결제", "해외 여부", "해외여부"),
}
_REQUIRED_UPLOAD_FIELDS: tuple[str, ...] = ("used_at", "merchant_name", "amount")
_DEFAULT_MERCHANT_STATUS = "가맹점"
_DEFAULT_TRANSACTION_STATUS = "APPROVED"
_DEFAULT_BOOLEAN_TEXT = "False"
_DEFAULT_PAYMENT_CHANNEL = "OFFLINE"
_DEFAULT_CATEGORY = "기타"


class MerchantInference(BaseModel):
    """가맹점명으로 추론한 거래 설명, 카테고리, 결제 채널을 표현한다."""

    merchant_name: str = Field(description="입력으로 받은 가맹점명")
    description: str = Field(description="사용자에게 보여줄 짧은 결제 내역 설명")
    category: str = Field(
        description="소비 카테고리. 예: 카페, 식비, 교통, 쇼핑, 생활, 의료, 여가, 기타"
    )
    payment_channel: str = Field(description="ONLINE 또는 OFFLINE 중 하나")


class MerchantInferenceBatch(BaseModel):
    """여러 가맹점명에 대한 거래 필드 추론 결과 묶음을 표현한다."""

    items: list[MerchantInference] = Field(description="가맹점별 추론 결과 목록")


class _StructuredInferenceRunnable(Protocol):
    def invoke(self, input_value: object) -> MerchantInferenceBatch:
        """구조화 출력 체인에 프롬프트 값을 전달하고 추론 결과를 반환한다."""
        ...


class _SupportsStructuredOutput(Protocol):
    def with_structured_output(
        self,
        schema: type[MerchantInferenceBatch],
    ) -> _StructuredInferenceRunnable:
        """LangChain 모델을 지정한 Pydantic 스키마 구조화 출력 모드로 감싼다."""
        ...


@dataclass(slots=True, frozen=True)
class TransactionUploadResult:
    """업로드된 거래 CSV를 SQLite에 저장한 결과를 표현한다."""

    inserted_count: int
    inferred_count: int
    skipped_count: int
    column_mapping: dict[str, str]
    sqlite_db_path: str


@dataclass(slots=True, frozen=True)
class _ParsedUploadRow:
    """CSV 한 행에서 transactions 테이블 저장에 필요한 값을 정리한 내부 구조다."""

    amount: int
    used_at: datetime
    merchant_name: str
    description: str | None
    category: str | None
    payment_channel: str | None
    merchant_status: str | None
    installment_flag: str | None
    installment_months: int | None
    installment_interest_type: str | None
    transaction_status: str | None
    is_overseas: str | None


def _normalize_column_name(column_name: str) -> str:
    """컬럼명 비교를 위해 공백, 구분자, 괄호를 제거하고 소문자로 정규화한다."""
    lowered = column_name.strip().lower()
    return re.sub(r"[\s_\-()/（）]+", "", lowered)


def resolve_transaction_upload_column_mapping(columns: Sequence[str]) -> dict[str, str]:
    """업로드 CSV 컬럼명을 transactions 테이블 필드명으로 매핑한다."""
    normalized_to_original: dict[str, str] = {}
    for column_name in columns:
        normalized_to_original.setdefault(
            _normalize_column_name(str(column_name)), str(column_name)
        )

    mapping: dict[str, str] = {}
    for field_name, aliases in _UPLOAD_COLUMN_ALIASES.items():
        for alias in aliases:
            matched_column = normalized_to_original.get(_normalize_column_name(alias))
            if matched_column is not None:
                mapping[field_name] = matched_column
                break
    return mapping


def _stringify_cell(value: object) -> str | None:
    """DataFrame 셀 값을 저장 가능한 문자열로 정리하고 빈 값은 None으로 반환한다."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if text == "":
        return None
    return text


def _get_record_value(record: dict[str, object], source_column: str | None) -> str | None:
    """업로드 행에서 매핑된 원본 컬럼 값을 문자열로 읽어온다."""
    if source_column is None:
        return None
    return _stringify_cell(record.get(source_column))


def _parse_amount(value: str | None) -> int:
    """쉼표, 원화 기호가 포함될 수 있는 금액 문자열을 정수 원 단위로 변환한다."""
    if value is None:
        raise ValueError("amount 값이 비어 있습니다.")
    normalized = value.replace(",", "")
    numeric_text = re.sub(r"[^0-9.\-]", "", normalized)
    if numeric_text in {"", "-", ".", "-."}:
        raise ValueError(f"amount 값을 숫자로 변환할 수 없습니다: {value}")
    return int(float(numeric_text))


def _parse_optional_int(value: str | None) -> int | None:
    """비어 있을 수 있는 정수 문자열을 int 또는 None으로 변환한다."""
    if value is None:
        return None
    return _parse_amount(value)


def _parse_used_at(value: str | None) -> datetime:
    """업로드 CSV의 거래 일시 문자열을 datetime으로 변환한다."""
    if value is None:
        raise ValueError("used_at 값이 비어 있습니다.")

    parsed_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed_value):
        raise ValueError(f"used_at 값을 날짜로 변환할 수 없습니다: {value}")
    if isinstance(parsed_value, pd.Timestamp):
        return parsed_value.to_pydatetime()
    if isinstance(parsed_value, datetime):
        return parsed_value
    raise ValueError(f"used_at 값을 날짜로 변환할 수 없습니다: {value}")


def _normalize_payment_channel(value: str | None) -> str | None:
    """결제 채널 텍스트를 ONLINE 또는 OFFLINE으로 정규화한다."""
    if value is None:
        return None
    normalized = value.strip().upper()
    if normalized in {"ONLINE", "ON", "온라인", "APP", "앱", "모바일"}:
        return "ONLINE"
    if normalized in {"OFFLINE", "OFF", "오프라인", "매장", "현장"}:
        return "OFFLINE"
    return normalized


def _build_inference_prompt() -> ChatPromptTemplate:
    """가맹점명 기반 거래 필드 추론에 사용할 LangChain 프롬프트를 생성한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 카드 결제 가맹점명을 보고 거래 내역을 정규화하는 분류기다. "
                "각 가맹점명에 대해 description, category, payment_channel을 한국어 서비스에 맞게 추론한다. "
                "payment_channel은 온라인 쇼핑몰, 배달앱, 구독 서비스, 앱 결제처럼 비대면 결제가 명확하면 ONLINE, "
                "그 외 매장 방문 가능성이 높으면 OFFLINE으로 답한다. 모르면 OFFLINE으로 둔다.",
            ),
            (
                "human",
                "가맹점명 목록:\n{merchant_names}\n\n"
                "카테고리는 카페, 식비, 편의점, 마트, 교통, 쇼핑, 의료, 통신, 구독, 여가, 교육, 생활, 기타 중 가장 가까운 값을 고른다. "
                "description은 가맹점명을 포함한 짧은 결제 설명으로 작성한다.",
            ),
        ]
    )


def infer_merchant_transaction_fields(
    merchant_names: Sequence[str],
    *,
    settings: Settings | None = None,
    llm: object | None = None,
) -> dict[str, MerchantInference]:
    """LangChain 구조화 출력으로 가맹점명별 거래 설명, 카테고리, 결제 채널을 추론한다."""
    unique_merchant_names = [name for name in dict.fromkeys(merchant_names) if name.strip()]
    if not unique_merchant_names:
        return {}

    config = settings or get_settings()
    chat_model = cast(
        _SupportsStructuredOutput,
        llm if llm is not None else get_chat_model(config, temperature=0.0),
    )
    inference_prompt = _build_inference_prompt()
    prompt_value = inference_prompt.invoke(
        {"merchant_names": "\n".join(f"- {name}" for name in unique_merchant_names)}
    )
    chain = chat_model.with_structured_output(MerchantInferenceBatch)
    result = chain.invoke(prompt_value)

    inference_by_merchant: dict[str, MerchantInference] = {}
    for item in result.items:
        merchant_name = item.merchant_name.strip()
        if merchant_name == "":
            continue
        inference_by_merchant[merchant_name] = MerchantInference(
            merchant_name=merchant_name,
            description=item.description.strip() or merchant_name,
            category=item.category.strip() or _DEFAULT_CATEGORY,
            payment_channel=_normalize_payment_channel(item.payment_channel)
            or _DEFAULT_PAYMENT_CHANNEL,
        )
    return inference_by_merchant


def _validate_required_columns(mapping: dict[str, str]) -> None:
    """거래 업로드에 필요한 필수 컬럼 매핑이 모두 있는지 검증한다."""
    missing_fields = [
        field_name for field_name in _REQUIRED_UPLOAD_FIELDS if field_name not in mapping
    ]
    if missing_fields:
        raise ValueError("필수 거래 컬럼을 찾을 수 없습니다: " + ", ".join(missing_fields))


def _parse_upload_row(record: dict[str, object], mapping: dict[str, str]) -> _ParsedUploadRow:
    """업로드 CSV 한 행을 transactions 테이블 저장용 내부 행으로 변환한다."""
    merchant_name = _get_record_value(record, mapping.get("merchant_name"))
    if merchant_name is None:
        raise ValueError("merchant_name 값이 비어 있습니다.")

    return _ParsedUploadRow(
        amount=_parse_amount(_get_record_value(record, mapping.get("amount"))),
        used_at=_parse_used_at(_get_record_value(record, mapping.get("used_at"))),
        merchant_name=merchant_name,
        description=_get_record_value(record, mapping.get("description")),
        category=_get_record_value(record, mapping.get("category")),
        payment_channel=_normalize_payment_channel(
            _get_record_value(record, mapping.get("payment_channel"))
        ),
        merchant_status=_get_record_value(record, mapping.get("merchant_status")),
        installment_flag=_get_record_value(record, mapping.get("installment_flag")),
        installment_months=_parse_optional_int(
            _get_record_value(record, mapping.get("installment_months"))
        ),
        installment_interest_type=_get_record_value(
            record,
            mapping.get("installment_interest_type"),
        ),
        transaction_status=_get_record_value(record, mapping.get("transaction_status")),
        is_overseas=_get_record_value(record, mapping.get("is_overseas")),
    )


def _needs_inference(row: _ParsedUploadRow) -> bool:
    """description, category, payment_channel 중 하나라도 비어 있으면 추론 필요 여부를 반환한다."""
    return row.description is None or row.category is None or row.payment_channel is None


def _get_next_transaction_id(settings: Settings) -> int:
    """현재 transactions 테이블의 최대 ID 다음 값을 계산한다."""
    with session_scope(settings) as session:
        max_id = session.scalar(select(func.max(TransactionModel.id))) or 0
    return max_id + 1


def _ensure_user_exists(settings: Settings, *, user_id: int) -> None:
    """업로드 대상 사용자가 SQLite users 테이블에 존재하는지 검증한다."""
    with session_scope(settings) as session:
        exists = session.scalar(select(UserModel.id).where(UserModel.id == user_id))
    if exists is None:
        raise ValueError(f"사용자 {user_id}번을 찾을 수 없습니다.")


def upload_transactions_dataframe(
    dataframe: pd.DataFrame,
    *,
    user_id: int,
    settings: Settings | None = None,
    llm: object | None = None,
) -> TransactionUploadResult:
    """업로드된 CSV DataFrame을 transactions 테이블 구조에 맞게 변환해 SQLite에 저장한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)
    _ensure_user_exists(config, user_id=user_id)

    column_mapping = resolve_transaction_upload_column_mapping(
        [str(column_name) for column_name in dataframe.columns]
    )
    _validate_required_columns(column_mapping)

    records = cast(list[dict[str, object]], dataframe.to_dict(orient="records"))
    parsed_rows = [_parse_upload_row(record, column_mapping) for record in records]
    rows_needing_inference = [row for row in parsed_rows if _needs_inference(row)]
    merchant_names_to_infer = [row.merchant_name for row in rows_needing_inference]
    inference_by_merchant = infer_merchant_transaction_fields(
        merchant_names_to_infer,
        settings=config,
        llm=llm,
    )

    next_transaction_id = _get_next_transaction_id(config)
    transactions: list[TransactionModel] = []
    for offset, row in enumerate(parsed_rows):
        inference = inference_by_merchant.get(row.merchant_name)
        description = row.description or (inference.description if inference else row.merchant_name)
        category = row.category or (inference.category if inference else _DEFAULT_CATEGORY)
        payment_channel = row.payment_channel or (
            inference.payment_channel if inference else _DEFAULT_PAYMENT_CHANNEL
        )

        transactions.append(
            TransactionModel(
                id=next_transaction_id + offset,
                user_id=user_id,
                amount=row.amount,
                used_at=row.used_at,
                description=description,
                merchant_name=row.merchant_name,
                merchant_status=row.merchant_status or _DEFAULT_MERCHANT_STATUS,
                installment_flag=row.installment_flag or _DEFAULT_BOOLEAN_TEXT,
                installment_months=row.installment_months or 0,
                installment_interest_type=row.installment_interest_type or _DEFAULT_BOOLEAN_TEXT,
                transaction_status=row.transaction_status or _DEFAULT_TRANSACTION_STATUS,
                is_overseas=row.is_overseas or _DEFAULT_BOOLEAN_TEXT,
                category=category,
                payment_channel=payment_channel,
            )
        )

    with session_scope(config) as session:
        session.add_all(transactions)

    return TransactionUploadResult(
        inserted_count=len(transactions),
        inferred_count=len(inference_by_merchant),
        skipped_count=0,
        column_mapping=column_mapping,
        sqlite_db_path=str(config.sqlite_db_path),
    )
