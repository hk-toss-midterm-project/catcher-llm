from __future__ import annotations

from datetime import date
from pathlib import Path

from catcher_llm.config.settings import Settings
from catcher_llm.db.models import UserModel
from catcher_llm.db.session import session_scope
from catcher_llm.services.group_competition_service import (
    CompetitionCreateInput,
    GroupCreateInput,
    TransactionShareInput,
    create_competition,
    create_group,
    get_group_feed,
    get_group_leaderboard,
    share_transaction_to_group,
)
from catcher_llm.services.user_data_service import ensure_user_database


def _write_seed_csvs(csv_dir: Path) -> None:
    """그룹 경쟁 테스트에 필요한 최소 사용자·거래 CSV를 작성한다."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "users_v3.csv").write_text(
        "\n".join(
            [
                "id,name,age,occupation,gender,annual_income,region,persona,personal_score,saving_goal_text,target_max_spending_amount",
                "1,김하나,29,개발자,Female,70000000,서울,계획형 소비자,53,비상금 300만원 만들기,1500000",
                "2,박둘,31,디자이너,Male,68000000,부산,기록형 소비자,57,식비 줄이기,1300000",
            ]
        ),
        encoding="utf-8",
    )
    (csv_dir / "transactions_v3.csv").write_text(
        "\n".join(
            [
                "id,user_id,amount,transaction_time,description,merchant_name,is_installment,installment_months,is_interest_free,status,is_overseas,category,payment_channel",
                "100,1,12000,2026-04-01 09:00:00,커피,스타벅스,False,0,False,APPROVED,False,식음료,OFFLINE",
                "101,2,18000,2026-04-02 08:30:00,아침 식사,샌드위치샵,False,0,False,APPROVED,False,식음료,OFFLINE",
            ]
        ),
        encoding="utf-8",
    )


def _make_settings(tmp_path: Path) -> Settings:
    """그룹 경쟁 테스트용 임시 데이터 디렉터리와 설정을 만든다."""
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


def test_create_group_creates_owner_membership(tmp_path: Path) -> None:
    """그룹을 만들면 생성자가 오너 권한의 첫 멤버로 함께 등록되는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)

    result = create_group(
        GroupCreateInput(
            owner_user_id=1,
            name="절약 원정대",
            description="한 달 생활비 절약을 함께 보는 그룹",
            member_user_ids=[2],
        ),
        settings=settings,
    )

    assert result.group_id == 1
    assert result.owner_user_id == 1
    assert result.member_count == 2


def test_share_transaction_to_group_records_feed_and_points(tmp_path: Path) -> None:
    """그룹 멤버가 자신의 소비를 공유하면 피드가 생성되고 personal_score가 증가하는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)
    group = create_group(
        GroupCreateInput(
            owner_user_id=1,
            name="소비 공개방",
            description="지출을 함께 돌아보는 그룹",
            member_user_ids=[2],
        ),
        settings=settings,
    )
    competition = create_competition(
        CompetitionCreateInput(
            group_id=group.group_id,
            title="4월 절약 대결",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 30),
        ),
        settings=settings,
    )

    share_result = share_transaction_to_group(
        TransactionShareInput(
            group_id=group.group_id,
            shared_by_user_id=1,
            transaction_id=100,
            competition_id=competition.competition_id,
            comment="오늘은 커피만 가볍게 샀어요.",
        ),
        settings=settings,
    )
    feed_items = get_group_feed(group.group_id, settings=settings)
    leaderboard = get_group_leaderboard(competition.competition_id, settings=settings)
    with session_scope(settings) as session:
        user = session.get(UserModel, 1)

    assert share_result.awarded_points == 10
    assert user is not None
    assert user.personal_score == 63
    assert len(feed_items) == 1
    assert feed_items[0]["transaction_id"] == 100
    assert feed_items[0]["shared_by_user_id"] == 1
    assert leaderboard[0]["user_id"] == 1
    assert leaderboard[0]["points"] == 63


def test_group_leaderboard_aggregates_points_per_competition(tmp_path: Path) -> None:
    """대회 리더보드가 그룹 멤버의 현재 personal_score를 기준으로 정렬되는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)
    group = create_group(
        GroupCreateInput(
            owner_user_id=1,
            name="랭킹 테스트",
            description="포인트 집계를 확인하는 그룹",
            member_user_ids=[2],
        ),
        settings=settings,
    )
    competition = create_competition(
        CompetitionCreateInput(
            group_id=group.group_id,
            title="4월 대결",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 30),
        ),
        settings=settings,
    )

    share_transaction_to_group(
        TransactionShareInput(
            group_id=group.group_id,
            shared_by_user_id=1,
            transaction_id=100,
            competition_id=competition.competition_id,
            comment="1번 사용자 공유",
        ),
        settings=settings,
    )
    share_transaction_to_group(
        TransactionShareInput(
            group_id=group.group_id,
            shared_by_user_id=2,
            transaction_id=101,
            competition_id=competition.competition_id,
            comment="2번 사용자 공유",
        ),
        settings=settings,
    )

    leaderboard = get_group_leaderboard(competition.competition_id, settings=settings)

    assert [item["user_id"] for item in leaderboard] == [2, 1]
    assert [item["points"] for item in leaderboard] == [67, 63]
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[1]["rank"] == 2
