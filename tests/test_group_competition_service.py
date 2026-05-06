from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from catcher_llm.config.settings import Settings
from catcher_llm.db.models import SessionModel
from catcher_llm.db.session import session_scope
from catcher_llm.services.user_data_service import ensure_user_database

_SERVICE_SPEC = importlib.util.spec_from_file_location(
    "test_group_competition_service_module",
    Path("src/catcher_llm/services/group_competition_service.py"),
)
if _SERVICE_SPEC is None or _SERVICE_SPEC.loader is None:
    raise RuntimeError("group_competition_service 모듈을 불러올 수 없습니다.")
_SERVICE_MODULE = importlib.util.module_from_spec(_SERVICE_SPEC)
sys.modules[_SERVICE_SPEC.name] = _SERVICE_MODULE
_SERVICE_SPEC.loader.exec_module(_SERVICE_MODULE)

GroupCreateInput = _SERVICE_MODULE.GroupCreateInput
create_group = _SERVICE_MODULE.create_group
get_group_leaderboard_for_group = _SERVICE_MODULE.get_group_leaderboard_for_group
get_group_member_feedback_status = _SERVICE_MODULE.get_group_member_feedback_status
list_user_groups = _SERVICE_MODULE.list_user_groups


def _write_seed_csvs(csv_dir: Path) -> None:
    """그룹 경쟁 테스트에 필요한 최소 사용자와 거래 CSV를 만든다."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "users_v4.csv").write_text(
        "\n".join(
            [
                "id,name,age,occupation,gender,annual_income,region,persona,personal_score,saving_goal_text,target_max_spending_amount",
                "1,김하나,29,개발자,Female,70000000,서울,계획형 소비자,53,비상금 300만원 만들기,1500000",
                "2,박둘,31,디자이너,Male,68000000,부산,기록형 소비자,57,식비 줄이기,1300000",
            ]
        ),
        encoding="utf-8",
    )
    (csv_dir / "transactions_v4.csv").write_text(
        "\n".join(
            [
                "id,user_id,amount,transaction_time,description,merchant_name,is_installment,installment_months,is_interest_free,status,is_overseas,category,payment_channel",
                "100,1,12000,2026-04-01 09:00:00,커피,스타벅스,False,0,False,APPROVED,False,음료,OFFLINE",
                "101,2,18000,2026-04-02 08:30:00,아침 식사,샌드위치샵,False,0,False,APPROVED,False,음식,OFFLINE",
            ]
        ),
        encoding="utf-8",
    )


def _write_numeric_sort_seed_csvs(csv_dir: Path) -> None:
    """문자열 점수 컬럼에서도 숫자 정렬을 검증할 수 있는 사용자 CSV를 만든다."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "users_v4.csv").write_text(
        "\n".join(
            [
                "id,name,age,occupation,gender,annual_income,region,persona,personal_score,saving_goal_text,target_max_spending_amount",
                "1,조한별,29,개발자,Female,70000000,서울,계획형 소비자,711,비상금 만들기,1500000",
                "2,장규범,31,디자이너,Male,68000000,부산,기록형 소비자,622,식비 줄이기,1300000",
                "3,이정익,27,학생,Male,32000000,인천,도전형 소비자,55,저축 습관 만들기,900000",
                "4,강상현,30,기획자,Female,61000000,대전,분석형 소비자,68,교통비 줄이기,1100000",
            ]
        ),
        encoding="utf-8",
    )
    (csv_dir / "transactions_v4.csv").write_text(
        "\n".join(
            [
                "id,user_id,amount,transaction_time,description,merchant_name,is_installment,installment_months,is_interest_free,status,is_overseas,category,payment_channel",
                "100,1,12000,2026-04-01 09:00:00,커피,스타벅스,False,0,False,APPROVED,False,음료,OFFLINE",
            ]
        ),
        encoding="utf-8",
    )


def _make_settings(tmp_path: Path) -> Settings:
    """그룹 경쟁 테스트용 임시 데이터 디렉터리 설정을 만든다."""
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


def _make_numeric_sort_settings(tmp_path: Path) -> Settings:
    """문자열 personal_score 정렬 검증용 임시 데이터 설정을 만든다."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    _write_numeric_sort_seed_csvs(raw_dir / "csv")
    return Settings(
        data_dir=data_dir,
        raw_data_dir=raw_dir,
        processed_data_dir=data_dir / "processed",
        vectorstore_dir=data_dir / "vectordb",
        eval_data_dir=data_dir / "evals",
        sqlite_db_path=data_dir / "sqlite" / "app.sqlite3",
    )


def test_create_group_creates_owner_membership(tmp_path: Path) -> None:
    """그룹을 만들면 생성자가 오너 멤버로 함께 등록되는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)

    result = create_group(
        GroupCreateInput(
            owner_user_id=1,
            name="절약 원정대",
            description="식비와 생활비를 함께 보는 그룹",
            member_user_ids=[2],
        ),
        settings=settings,
    )

    assert result.group_id == 1
    assert result.owner_user_id == 1
    assert result.member_count == 2


def test_list_user_groups_counts_all_group_members(tmp_path: Path) -> None:
    """내 그룹 목록의 멤버 수가 조회 사용자뿐 아니라 전체 그룹원을 세는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)
    create_group(
        GroupCreateInput(
            owner_user_id=1,
            name="멤버 수 검증반",
            description="목록에 전체 멤버 수를 표시하는 그룹",
            member_user_ids=[2],
        ),
        settings=settings,
    )

    owner_groups = list_user_groups(1, settings=settings)
    member_groups = list_user_groups(2, settings=settings)

    assert owner_groups[0]["member_count"] == 2
    assert owner_groups[0]["role"] == "owner"
    assert member_groups[0]["member_count"] == 2
    assert member_groups[0]["role"] == "member"


def test_group_leaderboard_for_group_uses_personal_score_without_competition(
    tmp_path: Path,
) -> None:
    """그룹만 만들어도 personal_score 기준 리더보드가 바로 조회되는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)
    group = create_group(
        GroupCreateInput(
            owner_user_id=1,
            name="소비 피드백반",
            description="지출을 함께 돌아보는 그룹",
            member_user_ids=[2],
        ),
        settings=settings,
    )

    leaderboard = get_group_leaderboard_for_group(group.group_id, settings=settings)

    assert [item["user_id"] for item in leaderboard] == [2, 1]
    assert [item["points"] for item in leaderboard] == [57, 53]
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[1]["rank"] == 2


def test_group_leaderboard_sorts_text_personal_score_as_numbers(tmp_path: Path) -> None:
    """personal_score 컬럼이 TEXT여도 리더보드가 숫자 크기대로 정렬되는지 검증한다."""
    settings = _make_numeric_sort_settings(tmp_path)
    ensure_user_database(settings=settings)
    group = create_group(
        GroupCreateInput(
            owner_user_id=1,
            name="랭킹 검증반",
            description="문자열 점수 정렬을 확인하는 그룹",
            member_user_ids=[2, 3, 4],
        ),
        settings=settings,
    )

    leaderboard = get_group_leaderboard_for_group(group.group_id, settings=settings)

    assert [item["user_id"] for item in leaderboard] == [1, 2, 4, 3]
    assert [item["points"] for item in leaderboard] == [711, 622, 68, 55]


def test_group_member_feedback_status_uses_recent_daily_window(tmp_path: Path) -> None:
    """피드백 현황이 그룹의 최신 daily 기준 최근 7일 데이터만 집계하는지 검증한다."""
    settings = _make_settings(tmp_path)
    ensure_user_database(settings=settings)
    group = create_group(
        GroupCreateInput(
            owner_user_id=1,
            name="미션 체크반",
            description="서로의 실천 정도를 확인하는 그룹",
            member_user_ids=[2],
        ),
        settings=settings,
    )

    with session_scope(settings) as session:
        session.add_all(
            [
                SessionModel(
                    user_id=1,
                    analysis_date="2026-04-01",
                    period_type="daily",
                    feedback_message="오래된 피드백",
                    feedback_reaction="like",
                    todo_tomorrow="오래된 미션",
                ),
                SessionModel(
                    user_id=1,
                    analysis_date="2026-04-09",
                    period_type="daily",
                    feedback_message="최근 피드백 1",
                    feedback_reaction="like",
                    todo_tomorrow="배달앱 지우기",
                ),
                SessionModel(
                    user_id=1,
                    analysis_date="2026-04-10",
                    period_type="daily",
                    feedback_message="최근 피드백 2",
                    feedback_reaction="like",
                    todo_tomorrow="야식 줄이기",
                ),
                SessionModel(
                    user_id=1,
                    analysis_date="2026-04-10",
                    period_type="weekly",
                    feedback_message="주간 피드백",
                    feedback_reaction="dislike",
                    todo_tomorrow="주간 미션",
                ),
                SessionModel(
                    user_id=2,
                    analysis_date="2026-04-10",
                    period_type="daily",
                    feedback_message="최근 피드백",
                    feedback_reaction="dislike",
                    todo_tomorrow="간식 줄이기",
                ),
            ]
        )

    feedback_status = get_group_member_feedback_status(group.group_id, settings=settings)

    assert feedback_status[0]["user_id"] == 2
    assert feedback_status[0]["feedback_checked_count"] == 1
    assert feedback_status[0]["positive_reaction_count"] == 0
    assert feedback_status[0]["feedback_acceptance_rate"] == 0
    assert feedback_status[0]["latest_mission"] == "간식 줄이기"
    assert feedback_status[1]["user_id"] == 1
    assert feedback_status[1]["feedback_checked_count"] == 2
    assert feedback_status[1]["positive_reaction_count"] == 2
    assert feedback_status[1]["feedback_acceptance_rate"] == 100
    assert feedback_status[1]["latest_mission"] == "야식 줄이기"
