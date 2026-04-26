from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

USER_CSV_COLUMN_TO_DB_COLUMN: dict[str, str] = {
    "id": "id",
    "name": "name",
    "age": "age",
    "직업": "job",
    "성별": "gender",
    "연봉": "income",
    "지역": "region",
    "최상위 카드등급": "card_grade",
    "페르소나": "persona",
    "saving_goal_text": "saving_goal_text",
}

TRANSACTION_CSV_COLUMN_TO_DB_COLUMN: dict[str, str] = {
    "id": "id",
    "멤버 id": "user_id",
    "사용 금액": "amount",
    "사용 시간": "used_at",
    "결제 내역": "description",
    "결제 장소 (가맹점 여부)": "merchant_status",
    "할부 여부": "installment_flag",
    "할부 개월": "installment_months",
    "할부 무/유이자 여부": "installment_interest_type",
    "거래 상태 (승인 / 취소)": "transaction_status",
    "해외 결제": "is_overseas",
    "업종 카테고리": "category",
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
    __table_args__ = (UniqueConstraint("user_id", "memory_key", name="uq_user_memory_key"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    memory_key: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
