from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from catcher_llm.config.settings import Settings
from catcher_llm.db.models import TransactionModel
from catcher_llm.db.session import session_scope
from catcher_llm.services.transaction_upload_service import (
    MerchantInference,
    MerchantInferenceBatch,
    resolve_transaction_upload_column_mapping,
    upload_transactions_dataframe,
)


def _write_seed_csvs(csv_dir: Path) -> None:
    """거래 업로드 테스트에 사용할 최소 사용자·거래 CSV를 작성한다."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "users_v4.csv").write_text(
        "\n".join(
            [
                "id,name,age,occupation,gender,annual_income,region,persona,personal_score,saving_goal_text,target_max_spending_amount",
                "1,김토스,29,개발자,남성,7000,서울,절약형,53,비상금 만들기,1500000",
            ]
        ),
        encoding="utf-8",
    )
    (csv_dir / "transactions_v4.csv").write_text(
        "\n".join(
            [
                "id,user_id,amount,transaction_time,description,merchant_name,is_installment,installment_months,is_interest_free,status,is_overseas,category,payment_channel",
                "100,1,12000,2026-04-01 09:00:00,커피,스타벅스,False,0,False,APPROVED,False,식음료,OFFLINE",
            ]
        ),
        encoding="utf-8",
    )


def _make_settings(tmp_path: Path) -> Settings:
    """거래 업로드 테스트용 임시 설정을 생성한다."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    _write_seed_csvs(raw_dir / "csv")
    return Settings(
        data_dir=data_dir,
        raw_data_dir=raw_dir,
        processed_data_dir=data_dir / "processed",
        vectorstore_dir=data_dir / "vectordb",
        eval_data_dir=data_dir / "evals",
        sqlite_db_path=data_dir / "sqlite" / "app.sqlite3",
    )


class _FakeInferenceChain:
    """LangChain 구조화 출력 체인 호출 입력을 기록하고 고정 추론 결과를 반환한다."""

    def __init__(self) -> None:
        self.payloads: list[str] = []

    def invoke(self, payload: object) -> MerchantInferenceBatch:
        """가맹점명 기반 추론 요청을 기록하고 테스트 결과를 반환한다."""
        self.payloads.append(str(payload))
        return MerchantInferenceBatch(
            items=[
                MerchantInference(
                    merchant_name="스타벅스 강남점",
                    description="스타벅스 강남점 결제",
                    category="카페",
                    payment_channel="OFFLINE",
                )
            ]
        )


class _FakeInferenceLlm:
    """with_structured_output 호출을 지원하는 테스트용 LangChain 모델 대역이다."""

    def __init__(self, chain: _FakeInferenceChain) -> None:
        self.chain = chain
        self.schema_names: list[str] = []

    def with_structured_output(self, schema: type[MerchantInferenceBatch]) -> _FakeInferenceChain:
        """요청된 구조화 출력 스키마를 기록하고 가짜 체인을 반환한다."""
        self.schema_names.append(schema.__name__)
        return self.chain


def test_resolve_transaction_upload_column_mapping_accepts_korean_headers() -> None:
    """업로드 CSV의 한글 컬럼명을 transactions 테이블 필드로 매핑하는지 검증한다."""
    mapping = resolve_transaction_upload_column_mapping(["이용일", "가맹점 명", "금액", "승인상태"])

    assert mapping == {
        "used_at": "이용일",
        "merchant_name": "가맹점 명",
        "amount": "금액",
        "transaction_status": "승인상태",
    }


def test_upload_transactions_dataframe_infers_fields_and_saves_sqlite(tmp_path: Path) -> None:
    """가맹점명으로 누락 필드를 LangChain 추론한 뒤 거래를 SQLite에 저장하는지 검증한다."""
    settings = _make_settings(tmp_path)
    inference_chain = _FakeInferenceChain()
    fake_llm = _FakeInferenceLlm(inference_chain)
    dataframe = pd.DataFrame(
        [
            {
                "이용일": "2026-05-01",
                "가맹점 명": "스타벅스 강남점",
                "금액": "5,800원",
            }
        ]
    )

    result = upload_transactions_dataframe(
        dataframe,
        user_id=1,
        settings=settings,
        llm=fake_llm,
    )

    with session_scope(settings) as session:
        transaction = session.scalar(
            select(TransactionModel).where(TransactionModel.merchant_name == "스타벅스 강남점")
        )

    assert result.inserted_count == 1
    assert result.inferred_count == 1
    assert result.column_mapping["used_at"] == "이용일"
    assert result.column_mapping["merchant_name"] == "가맹점 명"
    assert fake_llm.schema_names == ["MerchantInferenceBatch"]
    assert "스타벅스 강남점" in inference_chain.payloads[0]
    assert transaction is not None
    assert transaction.user_id == 1
    assert transaction.amount == 5800
    assert transaction.used_at == datetime(2026, 5, 1)
    assert transaction.description == "스타벅스 강남점 결제"
    assert transaction.category == "카페"
    assert transaction.payment_channel == "OFFLINE"
    assert transaction.transaction_status == "APPROVED"


def test_upload_transactions_dataframe_requires_core_columns(tmp_path: Path) -> None:
    """거래 업로드에 필요한 핵심 컬럼이 없으면 명확한 오류를 반환하는지 검증한다."""
    settings = _make_settings(tmp_path)
    dataframe = pd.DataFrame([{"가맹점 명": "스타벅스 강남점", "금액": 5800}])

    with pytest.raises(ValueError, match="used_at"):
        upload_transactions_dataframe(
            dataframe, user_id=1, settings=settings, llm=_FakeInferenceLlm(_FakeInferenceChain())
        )
