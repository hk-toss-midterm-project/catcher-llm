from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import Integer, cast, func, select

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import (
    CompetitionModel,
    GroupMembershipModel,
    GroupModel,
    GroupPointLedgerModel,
    SessionModel,
    SharedTransactionModel,
    TransactionModel,
    UserModel,
)
from catcher_llm.db.session import session_scope
from catcher_llm.services.user_data_service import ensure_user_database

_DEFAULT_SHARE_POINTS = 10
_FEEDBACK_WINDOW_DAYS = 7


@dataclass(slots=True, frozen=True)
class GroupCreateInput:
    """그룹 생성에 필요한 입력값을 담는다."""

    owner_user_id: int
    name: str
    description: str
    member_user_ids: list[int]


@dataclass(slots=True, frozen=True)
class GroupCreateResult:
    """그룹 생성 결과와 기본 멤버 수를 반환한다."""

    group_id: int
    owner_user_id: int
    member_count: int


@dataclass(slots=True, frozen=True)
class CompetitionCreateInput:
    """그룹 대회 생성에 필요한 입력값을 담는다."""

    group_id: int
    title: str
    start_date: date
    end_date: date


@dataclass(slots=True, frozen=True)
class CompetitionCreateResult:
    """생성된 대회의 식별자와 기간을 반환한다."""

    competition_id: int
    group_id: int
    start_date: date
    end_date: date


@dataclass(slots=True, frozen=True)
class TransactionShareInput:
    """그룹에 소비를 공유할 때 필요한 입력값을 담는다."""

    group_id: int
    shared_by_user_id: int
    transaction_id: int
    competition_id: int | None
    comment: str


@dataclass(slots=True, frozen=True)
class TransactionShareResult:
    """소비 공유 결과와 적립 포인트를 반환한다."""

    shared_transaction_id: int
    awarded_points: int


def _normalize_required_text(value: str, *, field_name: str) -> str:
    """필수 텍스트 입력을 공백 제거 후 검증해 반환한다."""
    normalized_value = value.strip()
    if normalized_value == "":
        raise ValueError(f"{field_name} 값을 입력해 주세요.")
    return normalized_value


def _get_user_or_raise(settings: Settings, *, user_id: int) -> UserModel:
    """주어진 사용자 ID가 존재하는지 확인하고 사용자 모델을 반환한다."""
    with session_scope(settings) as session:
        user = session.scalar(select(UserModel).where(UserModel.id == user_id))
    if user is None:
        raise ValueError(f"사용자 {user_id}번을 찾을 수 없습니다.")
    return user


def _get_group_or_raise(settings: Settings, *, group_id: int) -> GroupModel:
    """주어진 그룹 ID가 존재하는지 확인하고 그룹 모델을 반환한다."""
    with session_scope(settings) as session:
        group = session.scalar(select(GroupModel).where(GroupModel.id == group_id))
    if group is None:
        raise ValueError(f"그룹 {group_id}번을 찾을 수 없습니다.")
    return group


def _get_competition_or_raise(settings: Settings, *, competition_id: int) -> CompetitionModel:
    """주어진 대회 ID가 존재하는지 확인하고 대회 모델을 반환한다."""
    with session_scope(settings) as session:
        competition = session.scalar(
            select(CompetitionModel).where(CompetitionModel.id == competition_id)
        )
    if competition is None:
        raise ValueError(f"대회 {competition_id}번을 찾을 수 없습니다.")
    return competition


def _get_membership_or_raise(
    settings: Settings,
    *,
    group_id: int,
    user_id: int,
) -> GroupMembershipModel:
    """사용자가 그룹 멤버인지 확인하고 멤버십 모델을 반환한다."""
    with session_scope(settings) as session:
        membership = session.scalar(
            select(GroupMembershipModel).where(
                GroupMembershipModel.group_id == group_id,
                GroupMembershipModel.user_id == user_id,
            )
        )
    if membership is None:
        raise ValueError(f"사용자 {user_id}번은 그룹 {group_id}번의 멤버가 아닙니다.")
    return membership


def _get_transaction_or_raise(settings: Settings, *, transaction_id: int) -> TransactionModel:
    """주어진 거래 ID가 존재하는지 확인하고 거래 모델을 반환한다."""
    with session_scope(settings) as session:
        transaction = session.scalar(select(TransactionModel).where(TransactionModel.id == transaction_id))
    if transaction is None:
        raise ValueError(f"거래 {transaction_id}번을 찾을 수 없습니다.")
    return transaction


def create_group(
    payload: GroupCreateInput,
    *,
    settings: Settings | None = None,
) -> GroupCreateResult:
    """오너와 초대 멤버를 포함한 새 그룹을 생성한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)

    _get_user_or_raise(config, user_id=payload.owner_user_id)
    member_user_ids = sorted(set(payload.member_user_ids + [payload.owner_user_id]))
    for member_user_id in member_user_ids:
        _get_user_or_raise(config, user_id=member_user_id)

    group_name = _normalize_required_text(payload.name, field_name="그룹 이름")
    group_description = payload.description.strip()

    with session_scope(config) as session:
        group = GroupModel(
            name=group_name,
            description=group_description or None,
            owner_user_id=payload.owner_user_id,
        )
        session.add(group)
        session.flush()

        memberships = [
            GroupMembershipModel(
                group_id=group.id,
                user_id=member_user_id,
                role="owner" if member_user_id == payload.owner_user_id else "member",
            )
            for member_user_id in member_user_ids
        ]
        session.add_all(memberships)
        session.flush()

        return GroupCreateResult(
            group_id=group.id,
            owner_user_id=payload.owner_user_id,
            member_count=len(memberships),
        )


def create_competition(
    payload: CompetitionCreateInput,
    *,
    settings: Settings | None = None,
) -> CompetitionCreateResult:
    """그룹 안에서 기간이 있는 대회를 생성한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)
    _get_group_or_raise(config, group_id=payload.group_id)

    competition_title = _normalize_required_text(payload.title, field_name="대회 이름")
    if payload.start_date > payload.end_date:
        raise ValueError("대회 시작일은 종료일보다 늦을 수 없습니다.")

    with session_scope(config) as session:
        competition = CompetitionModel(
            group_id=payload.group_id,
            title=competition_title,
            start_date=payload.start_date,
            end_date=payload.end_date,
        )
        session.add(competition)
        session.flush()

        return CompetitionCreateResult(
            competition_id=competition.id,
            group_id=competition.group_id,
            start_date=competition.start_date,
            end_date=competition.end_date,
        )


def share_transaction_to_group(
    payload: TransactionShareInput,
    *,
    settings: Settings | None = None,
) -> TransactionShareResult:
    """그룹 멤버가 소비를 공유하고 포인트를 적립한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)
    _get_membership_or_raise(config, group_id=payload.group_id, user_id=payload.shared_by_user_id)
    transaction = _get_transaction_or_raise(config, transaction_id=payload.transaction_id)
    if transaction.user_id != payload.shared_by_user_id:
        raise ValueError("자신의 소비 내역만 그룹에 공유할 수 있습니다.")

    if payload.competition_id is not None:
        competition = _get_competition_or_raise(config, competition_id=payload.competition_id)
        if competition.group_id != payload.group_id:
            raise ValueError("선택한 대회가 그룹과 일치하지 않습니다.")

    with session_scope(config) as session:
        existing_share = session.scalar(
            select(SharedTransactionModel).where(
                SharedTransactionModel.group_id == payload.group_id,
                SharedTransactionModel.transaction_id == payload.transaction_id,
            )
        )
        if existing_share is not None:
            raise ValueError("같은 소비는 같은 그룹에 한 번만 공유할 수 있습니다.")

        shared_transaction = SharedTransactionModel(
            group_id=payload.group_id,
            transaction_id=payload.transaction_id,
            shared_by_user_id=payload.shared_by_user_id,
            competition_id=payload.competition_id,
            comment=payload.comment.strip() or None,
        )
        session.add(shared_transaction)
        session.flush()

        user = session.get(UserModel, payload.shared_by_user_id)
        if user is None:
            raise ValueError(f"사용자 {payload.shared_by_user_id}번을 찾을 수 없습니다.")
        user.personal_score = int(user.personal_score or 0) + _DEFAULT_SHARE_POINTS

        ledger_entry = GroupPointLedgerModel(
            group_id=payload.group_id,
            user_id=payload.shared_by_user_id,
            competition_id=payload.competition_id,
            shared_transaction_id=shared_transaction.id,
            event_type="share_transaction",
            points=_DEFAULT_SHARE_POINTS,
            reason="소비 내역 공유",
        )
        session.add(ledger_entry)
        session.flush()

        return TransactionShareResult(
            shared_transaction_id=shared_transaction.id,
            awarded_points=ledger_entry.points,
        )


def get_group_feed(
    group_id: int,
    *,
    settings: Settings | None = None,
) -> list[dict[str, int | str | None]]:
    """그룹에 공유된 소비 피드를 최신순으로 반환한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)
    _get_group_or_raise(config, group_id=group_id)

    with session_scope(config) as session:
        rows = session.execute(
            select(SharedTransactionModel, TransactionModel, UserModel)
            .join(TransactionModel, TransactionModel.id == SharedTransactionModel.transaction_id)
            .join(UserModel, UserModel.id == SharedTransactionModel.shared_by_user_id)
            .where(SharedTransactionModel.group_id == group_id)
            .order_by(SharedTransactionModel.shared_at.desc(), SharedTransactionModel.id.desc())
        ).all()

    return [
        {
            "shared_transaction_id": shared_transaction.id,
            "transaction_id": transaction.id,
            "shared_by_user_id": user.id,
            "shared_by_name": user.name,
            "amount": transaction.amount,
            "description": transaction.description,
            "comment": shared_transaction.comment,
            "shared_at": shared_transaction.shared_at.isoformat(),
        }
        for shared_transaction, transaction, user in rows
    ]


def list_user_groups(
    user_id: int,
    *,
    settings: Settings | None = None,
) -> list[dict[str, int | str]]:
    """사용자가 속한 그룹 목록을 멤버 수와 역할과 함께 반환한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)
    _get_user_or_raise(config, user_id=user_id)

    with session_scope(config) as session:
        rows = session.execute(
            select(
                GroupModel.id,
                GroupModel.name,
                GroupModel.description,
                GroupModel.owner_user_id,
                GroupMembershipModel.role,
                func.count(GroupMembershipModel.id)
                .over(partition_by=GroupMembershipModel.group_id)
                .label("member_count"),
            )
            .join(GroupMembershipModel, GroupMembershipModel.group_id == GroupModel.id)
            .where(GroupMembershipModel.user_id == user_id)
            .order_by(GroupModel.id.asc())
        ).all()

    return [
        {
            "group_id": int(row.id),
            "name": str(row.name),
            "description": str(row.description or ""),
            "owner_user_id": int(row.owner_user_id),
            "role": str(row.role),
            "member_count": int(row.member_count),
        }
        for row in rows
    ]


def list_group_competitions(
    group_id: int,
    *,
    settings: Settings | None = None,
) -> list[dict[str, int | str]]:
    """그룹에 등록된 대회 목록을 최신 생성순으로 반환한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)
    _get_group_or_raise(config, group_id=group_id)

    with session_scope(config) as session:
        competitions = session.scalars(
            select(CompetitionModel)
            .where(CompetitionModel.group_id == group_id)
            .order_by(CompetitionModel.id.desc())
        ).all()

    return [
        {
            "competition_id": competition.id,
            "title": competition.title,
            "start_date": competition.start_date.isoformat(),
            "end_date": competition.end_date.isoformat(),
            "status": competition.status,
        }
        for competition in competitions
    ]


def get_group_leaderboard_for_group(
    group_id: int,
    *,
    settings: Settings | None = None,
) -> list[dict[str, int | str]]:
    """그룹 멤버를 현재 personal_score 기준으로 정렬해 반환한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)
    _get_group_or_raise(config, group_id=group_id)
    score_expr = func.coalesce(cast(UserModel.personal_score, Integer), 0)

    with session_scope(config) as session:
        rows = session.execute(
            select(
                GroupMembershipModel.user_id,
                UserModel.name,
                score_expr.label("total_points"),
            )
            .join(UserModel, UserModel.id == GroupMembershipModel.user_id)
            .where(GroupMembershipModel.group_id == group_id)
            .order_by(
                score_expr.desc(),
                GroupMembershipModel.user_id.asc(),
            )
        ).all()

    leaderboard: list[dict[str, int | str]] = []
    for index, row in enumerate(rows, start=1):
        leaderboard.append(
            {
                "rank": index,
                "user_id": int(row.user_id),
                "user_name": str(row.name),
                "points": int(row.total_points),
            }
        )
    return leaderboard


def get_group_member_feedback_status(
    group_id: int,
    *,
    settings: Settings | None = None,
) -> list[dict[str, int | str | None]]:
    """그룹의 최신 daily 기준 최근 7일 피드백 반응과 미션을 요약해 반환한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)
    leaderboard = get_group_leaderboard_for_group(group_id, settings=config)

    user_ids = [int(item["user_id"]) for item in leaderboard]
    if not user_ids:
        return []

    with session_scope(config) as session:
        session_rows = session.scalars(
            select(SessionModel)
            .where(
                SessionModel.user_id.in_(user_ids),
                SessionModel.period_type == "daily",
            )
            .order_by(
                SessionModel.user_id.asc(),
                SessionModel.analysis_date.desc(),
                SessionModel.id.desc(),
            )
        ).all()

    parsed_dates = [
        _parse_session_analysis_date(session_row.analysis_date) for session_row in session_rows
    ]
    valid_dates = [parsed_date for parsed_date in parsed_dates if parsed_date is not None]
    if not valid_dates:
        return [
            {
                "user_id": int(item["user_id"]),
                "user_name": str(item["user_name"]),
                "points": int(item["points"]),
                "feedback_checked_count": 0,
                "positive_reaction_count": 0,
                "feedback_acceptance_rate": 0,
                "latest_mission": None,
            }
            for item in leaderboard
        ]

    latest_daily_date = max(valid_dates)
    window_start_date = latest_daily_date - timedelta(days=_FEEDBACK_WINDOW_DAYS - 1)

    sessions_by_user_id: dict[int, list[SessionModel]] = {user_id: [] for user_id in user_ids}
    for session_row in session_rows:
        parsed_date = _parse_session_analysis_date(session_row.analysis_date)
        if parsed_date is None or parsed_date < window_start_date:
            continue
        sessions_by_user_id.setdefault(session_row.user_id, []).append(session_row)

    feedback_status: list[dict[str, int | str | None]] = []
    for item in leaderboard:
        member_user_id = int(item["user_id"])
        member_sessions = sessions_by_user_id.get(member_user_id, [])
        checked_sessions = [
            session_row
            for session_row in member_sessions
            if session_row.feedback_reaction is not None and session_row.feedback_reaction.strip() != ""
        ]
        positive_reaction_count = sum(
            1 for session_row in checked_sessions if session_row.feedback_reaction == "like"
        )
        checked_count = len(checked_sessions)
        acceptance_rate = (
            int(round((positive_reaction_count / checked_count) * 100)) if checked_count > 0 else 0
        )
        latest_mission = next(
            (
                session_row.todo_tomorrow
                for session_row in member_sessions
                if session_row.todo_tomorrow is not None and session_row.todo_tomorrow.strip() != ""
            ),
            None,
        )
        feedback_status.append(
            {
                "user_id": member_user_id,
                "user_name": str(item["user_name"]),
                "points": int(item["points"]),
                "feedback_checked_count": checked_count,
                "positive_reaction_count": positive_reaction_count,
                "feedback_acceptance_rate": acceptance_rate,
                "latest_mission": latest_mission,
            }
        )
    return feedback_status


def _parse_session_analysis_date(value: str) -> date | None:
    """세션의 analysis_date 문자열을 날짜로 변환하고 실패하면 None을 반환한다."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def get_group_leaderboard(
    competition_id: int,
    *,
    settings: Settings | None = None,
) -> list[dict[str, int | str]]:
    """대회가 속한 그룹의 현재 personal_score 리더보드를 반환한다."""
    config = settings or get_settings()
    ensure_user_database(settings=config)
    competition = _get_competition_or_raise(config, competition_id=competition_id)
    return get_group_leaderboard_for_group(competition.group_id, settings=config)
