from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

USER_CSV_COLUMN_TO_DB_COLUMN: dict[str, str] = {
    "id": "id",
    "name": "name",
    "age": "age",
    "occupation": "job",
    "직업": "job",
    "gender": "gender",
    "성별": "gender",
    "annual_income": "income",
    "연봉": "income",
    "지역": "region",
    "region": "region",
    "card_grade": "card_grade",
    "최상위 카드등급": "card_grade",
    "persona": "persona",
    "페르소나": "persona",
    "saving_goal_text": "saving_goal_text",
}

TRANSACTION_CSV_COLUMN_TO_DB_COLUMN: dict[str, str] = {
    "id": "id",
    "user_id": "user_id",
    "멤버 id": "user_id",
    "amount": "amount",
    "사용 금액": "amount",
    "transaction_time": "used_at",
    "used_at": "used_at",
    "사용 시간": "used_at",
    "description": "description",
    "결제 내역": "description",
    "결제 장소 (가맹점 여부)": "merchant_status",
    "is_installment": "installment_flag",
    "할부 여부": "installment_flag",
    "installment_months": "installment_months",
    "할부 개월": "installment_months",
    "is_interest_free": "installment_interest_type",
    "할부 무/유이자 여부": "installment_interest_type",
    "status": "transaction_status",
    "거래 상태 (승인 / 취소)": "transaction_status",
    "is_overseas": "is_overseas",
    "해외 결제": "is_overseas",
    "category": "category",
    "업종 카테고리": "category",
    "payment_channel": "payment_channel",
    "결제 방식 (온/오프라인)": "payment_channel",
}


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int | None]
    job: Mapped[str | None] = mapped_column(String(100))
    gender: Mapped[str | None] = mapped_column(String(30))
    income: Mapped[str | None] = mapped_column(String(50))
    region: Mapped[str | None] = mapped_column(String(100))
    card_grade: Mapped[str | None] = mapped_column(String(50))
    persona: Mapped[str | None] = mapped_column(String(100))
    saving_goal_text: Mapped[str | None] = mapped_column(Text, nullable=True)


class TransactionModel(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    amount: Mapped[int | None]
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    description: Mapped[str | None] = mapped_column(Text)
    merchant_status: Mapped[str | None] = mapped_column(String(50))
    installment_flag: Mapped[str | None] = mapped_column(String(20))
    installment_months: Mapped[int | None]
    installment_interest_type: Mapped[str | None] = mapped_column(String(50))
    transaction_status: Mapped[str | None] = mapped_column(String(20))
    is_overseas: Mapped[str | None] = mapped_column(String(20))
    category: Mapped[str | None] = mapped_column(String(100))
    payment_channel: Mapped[str | None] = mapped_column(String(30))


class UserMemoryModel(Base):
    __tablename__ = "user_memories"
    __table_args__ = (UniqueConstraint("user_id", "period_type", name="uq_user_memory_period"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    period_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # 'daily', 'weekly', 'monthly'
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    user_feedback_memory: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )


class SessionModel(Base):
    __tablename__ = "session"
    __table_args__ = (
        UniqueConstraint("user_id", "analysis_date", "period_type", name="uq_session_period"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    analysis_date: Mapped[str] = mapped_column(String(20), nullable=False, default="")
    period_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="daily"
    )  # 'daily', 'weekly', 'monthly'
    analysis_result: Mapped[str | None] = mapped_column(Text)
    feedback_message: Mapped[str | None] = mapped_column(
        Text
    )  # 자연어 피드백 본문 (scolding_message / feedback_message)
    feedback_reason: Mapped[str | None] = mapped_column(Text)
    feedback_reaction: Mapped[str | None] = mapped_column(String(20))
    feedback_reaction_reason: Mapped[str | None] = mapped_column(Text)
    todo_tomorrow: Mapped[str | None] = mapped_column(Text)

    @property
    def daily_analysis_result(self) -> str | None:
        """기존 일일 세션 코드가 공통 분석 결과 컬럼을 읽을 수 있게 별칭을 제공한다."""
        return self.analysis_result

    @daily_analysis_result.setter
    def daily_analysis_result(self, value: str | None) -> None:
        """기존 일일 세션 코드가 저장한 분석 결과를 공통 컬럼에 반영한다."""
        self.analysis_result = value

    @property
    def weekly_analysis_result(self) -> str | None:
        """기존 주간 세션 코드가 공통 분석 결과 컬럼을 읽을 수 있게 별칭을 제공한다."""
        return self.analysis_result

    @weekly_analysis_result.setter
    def weekly_analysis_result(self, value: str | None) -> None:
        """기존 주간 세션 코드가 저장한 분석 결과를 공통 컬럼에 반영한다."""
        self.analysis_result = value

    @property
    def monthly_analysis_result(self) -> str | None:
        """기존 월간 세션 코드가 공통 분석 결과 컬럼을 읽을 수 있게 별칭을 제공한다."""
        return self.analysis_result

    @monthly_analysis_result.setter
    def monthly_analysis_result(self, value: str | None) -> None:
        """기존 월간 세션 코드가 저장한 분석 결과를 공통 컬럼에 반영한다."""
        self.analysis_result = value
