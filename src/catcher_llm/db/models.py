from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

USER_CSV_COLUMN_TO_DB_COLUMN: dict[str, str] = {
    "id": "id",
    "name": "name",
    "age": "age",
    "occupation": "job",
    "吏곸뾽": "job",
    "gender": "gender",
    "?깅퀎": "gender",
    "annual_income": "income",
    "?곕큺": "income",
    "吏??": "region",
    "region": "region",
    "card_grade": "card_grade",
    "理쒖긽??移대뱶?깃툒": "card_grade",
    "persona": "persona",
    "?섎Ⅴ?뚮굹": "persona",
    "saving_goal_text": "saving_goal_text",
}

TRANSACTION_CSV_COLUMN_TO_DB_COLUMN: dict[str, str] = {
    "id": "id",
    "user_id": "user_id",
    "硫ㅻ쾭 id": "user_id",
    "amount": "amount",
    "?ъ슜 湲덉븸": "amount",
    "transaction_time": "used_at",
    "used_at": "used_at",
    "?ъ슜 ?쒓컙": "used_at",
    "description": "description",
    "寃곗젣 ?댁뿭": "description",
    "merchant_name": "merchant_name",
    "媛留뱀젏紐?": "merchant_name",
    "寃곗젣 ?μ냼 (媛留뱀젏 ?щ?)": "merchant_status",
    "is_installment": "installment_flag",
    "?좊? ?щ?": "installment_flag",
    "installment_months": "installment_months",
    "?좊? 媛쒖썡": "installment_months",
    "is_interest_free": "installment_interest_type",
    "?좊? 臾??좎씠???щ?": "installment_interest_type",
    "status": "transaction_status",
    "嫄곕옒 ?곹깭 (?뱀씤 / 痍⑥냼)": "transaction_status",
    "is_overseas": "is_overseas",
    "?댁쇅 寃곗젣": "is_overseas",
    "category": "category",
    "?낆쥌 移댄뀒怨좊━": "category",
    "payment_channel": "payment_channel",
    "寃곗젣 諛⑹떇 (???ㅽ봽?쇱씤)": "payment_channel",
}


class Base(DeclarativeBase):
    """프로젝트 SQLAlchemy 모델의 공통 베이스를 정의한다."""


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
    personal_score: Mapped[int | None]
    saving_goal_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_max_spending_amount: Mapped[int | None]


class TransactionModel(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    amount: Mapped[int | None]
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    description: Mapped[str | None] = mapped_column(Text)
    merchant_name: Mapped[str | None] = mapped_column(Text)
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
    period_type: Mapped[str] = mapped_column(String(20), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )


class UserFeedbackMemoryModel(Base):
    """사용자 피드백 거부 이유를 개별 행으로 저장하는 테이블.

    user_memories.user_feedback_memory 컬럼을 분리해 독립 테이블로 관리한다.
    created_at 기준으로 오래된 항목부터 주기별 초기화가 이루어진다.
    """

    __tablename__ = "user_feedback_memories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    period_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC),
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
    period_type: Mapped[str] = mapped_column(String(20), nullable=False, default="daily")
    analysis_result: Mapped[str | None] = mapped_column(Text)
    feedback_message: Mapped[str | None] = mapped_column(Text)
    feedback_reason: Mapped[str | None] = mapped_column(Text)
    feedback_reaction: Mapped[str | None] = mapped_column(String(20))
    feedback_reaction_reason: Mapped[str | None] = mapped_column(Text)
    todo_tomorrow: Mapped[str | None] = mapped_column(Text)

    @property
    def daily_analysis_result(self) -> str | None:
        """기존 일간 세션 코드가 공통 분석 결과 컬럼을 읽도록 호환 속성을 제공한다."""
        return self.analysis_result

    @daily_analysis_result.setter
    def daily_analysis_result(self, value: str | None) -> None:
        """기존 일간 세션 코드가 저장한 분석 결과를 공통 컬럼에 반영한다."""
        self.analysis_result = value

    @property
    def weekly_analysis_result(self) -> str | None:
        """기존 주간 세션 코드가 공통 분석 결과 컬럼을 읽도록 호환 속성을 제공한다."""
        return self.analysis_result

    @weekly_analysis_result.setter
    def weekly_analysis_result(self, value: str | None) -> None:
        """기존 주간 세션 코드가 저장한 분석 결과를 공통 컬럼에 반영한다."""
        self.analysis_result = value

    @property
    def monthly_analysis_result(self) -> str | None:
        """기존 월간 세션 코드가 공통 분석 결과 컬럼을 읽도록 호환 속성을 제공한다."""
        return self.analysis_result

    @monthly_analysis_result.setter
    def monthly_analysis_result(self, value: str | None) -> None:
        """기존 월간 세션 코드가 저장한 분석 결과를 공통 컬럼에 반영한다."""
        self.analysis_result = value


class GroupModel(Base):
    __tablename__ = "groups"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)


class GroupMembershipModel(Base):
    __tablename__ = "group_memberships"
    __table_args__ = (UniqueConstraint("group_id", "user_id", name="uq_group_membership"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="member")
    joined_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)


class CompetitionModel(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(100), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)


class SharedTransactionModel(Base):
    __tablename__ = "shared_transactions"
    __table_args__ = (UniqueConstraint("group_id", "transaction_id", name="uq_group_shared_tx"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"), index=True, nullable=False)
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.id"),
        index=True,
        nullable=False,
    )
    shared_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=False
    )
    competition_id: Mapped[int | None] = mapped_column(ForeignKey("competitions.id"), index=True)
    comment: Mapped[str | None] = mapped_column(Text)
    shared_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)


class GroupPointLedgerModel(Base):
    __tablename__ = "group_point_ledger"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    competition_id: Mapped[int | None] = mapped_column(ForeignKey("competitions.id"), index=True)
    shared_transaction_id: Mapped[int | None] = mapped_column(
        ForeignKey("shared_transactions.id"),
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    points: Mapped[int] = mapped_column(nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)
