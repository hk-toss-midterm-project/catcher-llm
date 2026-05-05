"""LangChain으로 사용자 동향 지표 CSV를 읽어 Markdown 보고서를 생성한다."""

from __future__ import annotations

import argparse
import os
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model

DEFAULT_METRICS_ROOT = Path("data/processed/user_trend_metrics")
DEFAULT_REPORT_ROOT = Path("data/raw/markdown/users_report")
DEFAULT_REPORT_FILENAME = "users_trend_report.md"
DEFAULT_MONTHLY_REPORT_DIRNAME = "monthly"
DEFAULT_QUARTERLY_REPORT_DIRNAME = "quarterly"

ReportType = Literal["period", "monthly", "quarterly"]

_METRIC_FILES = {
    "monthly_trends": "monthly_trends.csv",
    "category_monthly_trends": "category_monthly_trends.csv",
    "segment_monthly_trends": "segment_monthly_trends.csv",
    "user_monthly_metrics": "user_monthly_metrics.csv",
    "weekly_trends": "weekly_trends.csv",
    "daily_trends": "daily_trends.csv",
    "llm_trend_metrics": "llm_trend_metrics.csv",
}


@dataclass(frozen=True)
class UserReportScope:
    """RAG용 사용자 소비 보고서가 다루는 기간 범위와 출력 제목을 표현한다."""

    report_type: ReportType
    period: str
    months: tuple[str, ...]
    period_label: str
    title: str


@dataclass(frozen=True)
class ReportGenerationResult:
    """생성된 보고서 경로와 사용한 지표 디렉터리를 함께 반환한다."""

    metrics_dir: Path
    output_path: Path
    report_markdown: str
    report_scope: UserReportScope | None = None


@dataclass(frozen=True)
class ReportBatchGenerationResult:
    """여러 RAG용 사용자 소비 보고서 생성 결과를 묶어서 반환한다."""

    metrics_dir: Path
    reports: tuple[ReportGenerationResult, ...]


@dataclass(frozen=True)
class UserReportEvidenceCard:
    """RAG 검색에 직접 사용할 사용자 소비 동향 근거 카드 한 장을 표현한다."""

    issue_id: str
    period_type: ReportType
    period: str
    category: str
    keywords: tuple[str, ...]
    metric_name: str
    metric_value: int | float | str
    metric_unit: str
    change_rate_percent: float | None
    source_table: str
    usable_claim: str
    feedback_use: str
    personal_feedback_use_case: str
    caution: str


def _format_number(value: float | int) -> str:
    """보고서 컨텍스트에서 읽기 쉽도록 숫자에 천 단위 구분자를 붙인다."""
    return f"{float(value):,.4f}".rstrip("0").rstrip(".")


def _format_plain_number(value: float | int) -> str:
    """YAML과 근거 카드에서 쓰기 좋은 구분자 없는 숫자 문자열을 만든다."""
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _read_metric_csv(path: Path) -> pd.DataFrame:
    """지표 CSV 하나를 UTF-8 BOM 가능성을 고려해 DataFrame으로 읽는다."""
    return pd.read_csv(path, encoding="utf-8-sig")


def read_metric_tables(metrics_dir: Path) -> dict[str, pd.DataFrame]:
    """지표 디렉터리의 표준 CSV 파일들을 이름별 DataFrame으로 읽는다."""
    tables: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []
    for table_name, filename in _METRIC_FILES.items():
        csv_path = metrics_dir / filename
        if not csv_path.exists():
            missing_files.append(filename)
            continue
        tables[table_name] = _read_metric_csv(csv_path)

    if missing_files:
        joined_files = ", ".join(missing_files)
        raise FileNotFoundError(f"지표 디렉터리에 필요한 CSV가 없습니다: {joined_files}")
    return tables


def resolve_metrics_dir(
    metrics_dir: Path | None = None,
    *,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
) -> Path:
    """명시된 지표 디렉터리 또는 최신 기간 지표 디렉터리를 반환한다."""
    if metrics_dir is not None:
        if not metrics_dir.exists():
            raise FileNotFoundError(f"지표 디렉터리를 찾을 수 없습니다: {metrics_dir}")
        return metrics_dir

    if (metrics_root / "llm_trend_metrics.csv").exists():
        return metrics_root

    candidates = sorted(
        path
        for path in metrics_root.iterdir()
        if path.is_dir() and (path / "llm_trend_metrics.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"사용 가능한 지표 디렉터리가 없습니다: {metrics_root}")
    return candidates[-1]


def infer_period_label(metrics_dir: Path, tables: dict[str, pd.DataFrame]) -> str:
    """지표 디렉터리명 또는 일별 지표에서 보고서 기간 라벨을 추론한다."""
    name = metrics_dir.name
    if name.startswith("period_") and "_to_" in name:
        start, end = name.removeprefix("period_").split("_to_", maxsplit=1)
        return f"{start} ~ {end}"

    daily_trends = tables.get("daily_trends", pd.DataFrame())
    if not daily_trends.empty and "date" in daily_trends.columns:
        dates = daily_trends["date"].dropna().astype("string")
        if not dates.empty:
            return f"{dates.min()} ~ {dates.max()}"

    monthly_trends = tables.get("monthly_trends", pd.DataFrame())
    if not monthly_trends.empty and "month" in monthly_trends.columns:
        months = monthly_trends["month"].dropna().astype("string")
        if not months.empty:
            return f"{months.min()} ~ {months.max()}"
    return "기간 미상"


def _parse_month_label(month: str) -> tuple[int, int]:
    """YYYY-MM 월 라벨을 연도와 월 숫자로 변환한다."""
    year_text, month_text = month.split("-", maxsplit=1)
    return int(year_text), int(month_text)


def _month_date_range_label(months: tuple[str, ...]) -> str:
    """월 라벨 묶음을 일자 범위 문자열로 변환한다."""
    if not months:
        return "기간 미상"
    sorted_months = tuple(sorted(months))
    first_year, first_month = _parse_month_label(sorted_months[0])
    last_year, last_month = _parse_month_label(sorted_months[-1])
    last_day = monthrange(last_year, last_month)[1]
    return (
        f"{first_year:04d}-{first_month:02d}-01 ~ {last_year:04d}-{last_month:02d}-{last_day:02d}"
    )


def _quarter_label_from_month(month: str) -> str:
    """YYYY-MM 월 라벨을 YYYY-Qn 분기 라벨로 변환한다."""
    year, month_number = _parse_month_label(month)
    quarter = ((month_number - 1) // 3) + 1
    return f"{year:04d}-Q{quarter}"


def _available_months(tables: dict[str, pd.DataFrame]) -> tuple[str, ...]:
    """월별 지표 테이블에서 보고서 생성 가능한 월 목록을 추출한다."""
    monthly_trends = tables.get("monthly_trends", pd.DataFrame())
    if monthly_trends.empty or "month" not in monthly_trends.columns:
        return ()
    months = monthly_trends["month"].dropna().astype("string")
    return tuple(sorted(dict.fromkeys(str(month) for month in months)))


def build_user_report_scopes(
    metrics_dir: Path,
    *,
    include_monthly: bool = True,
    include_quarterly: bool = True,
) -> tuple[UserReportScope, ...]:
    """지표 디렉터리에서 월간·분기 RAG 보고서 생성 범위 목록을 만든다."""
    tables = read_metric_tables(metrics_dir)
    months = _available_months(tables)
    scopes: list[UserReportScope] = []

    if include_monthly:
        for month in months:
            scopes.append(
                UserReportScope(
                    report_type="monthly",
                    period=month,
                    months=(month,),
                    period_label=_month_date_range_label((month,)),
                    title="# 사용자 소비 동향 월간 보고서",
                )
            )

    if include_quarterly:
        quarter_months: dict[str, list[str]] = {}
        for month in months:
            quarter_months.setdefault(_quarter_label_from_month(month), []).append(month)
        for quarter in sorted(quarter_months):
            quarter_scope_months = tuple(sorted(quarter_months[quarter]))
            scopes.append(
                UserReportScope(
                    report_type="quarterly",
                    period=quarter,
                    months=quarter_scope_months,
                    period_label=_month_date_range_label(quarter_scope_months),
                    title="# 사용자 소비 동향 분기 보고서",
                )
            )

    return tuple(scopes)


def _frame_to_csv_block(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    """LLM 입력에 넣기 좋은 작은 CSV 블록으로 DataFrame을 직렬화한다."""
    if frame.empty:
        return "(데이터 없음)"
    return frame.head(max_rows).to_csv(index=False).strip()


def _filter_frame_by_months(
    frame: pd.DataFrame,
    column: str,
    months: tuple[str, ...],
) -> pd.DataFrame:
    """지정 컬럼의 YYYY-MM 값이 보고서 대상 월에 포함되는 행만 남긴다."""
    if frame.empty or column not in frame.columns:
        return frame.copy()
    month_values = set(months)
    mask = frame[column].astype("string").fillna("").isin(month_values)
    return frame[mask].copy()


def _filter_frame_by_date_months(
    frame: pd.DataFrame,
    column: str,
    months: tuple[str, ...],
) -> pd.DataFrame:
    """일자 컬럼의 앞 7자리 YYYY-MM 기준으로 보고서 대상 월 행만 남긴다."""
    if frame.empty or column not in frame.columns:
        return frame.copy()
    month_values = set(months)
    mask = frame[column].astype("string").fillna("").str[:7].isin(month_values)
    return frame[mask].copy()


def _filter_weekly_frame_by_months(
    frame: pd.DataFrame,
    months: tuple[str, ...],
) -> pd.DataFrame:
    """주간 시작일 또는 종료일이 보고서 대상 월에 걸친 주간 행만 남긴다."""
    if frame.empty:
        return frame.copy()
    month_values = set(months)
    masks: list[pd.Series] = []
    for column in ("week_start", "week_end"):
        if column in frame.columns:
            masks.append(frame[column].astype("string").fillna("").str[:7].isin(month_values))
    if not masks:
        return frame.copy()
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = combined_mask | mask
    return frame[combined_mask].copy()


def _filter_llm_metric_frame_by_months(
    frame: pd.DataFrame,
    months: tuple[str, ...],
) -> pd.DataFrame:
    """LLM long-form 지표 중 대상 월에 해당하는 월 단위 지표만 남긴다."""
    if frame.empty or "period" not in frame.columns:
        return frame.copy()
    month_values = set(months)
    mask = frame["period"].astype("string").fillna("").isin(month_values)
    return frame[mask].copy()


def _filter_metric_tables_for_scope(
    tables: dict[str, pd.DataFrame],
    report_scope: UserReportScope | None,
) -> dict[str, pd.DataFrame]:
    """보고서 스코프가 지정된 경우 지표 테이블을 해당 월 묶음으로 제한한다."""
    if report_scope is None or report_scope.report_type == "period":
        return {name: frame.copy() for name, frame in tables.items()}

    months = report_scope.months
    filtered_tables: dict[str, pd.DataFrame] = {}
    for table_name, frame in tables.items():
        if table_name in {
            "monthly_trends",
            "category_monthly_trends",
            "segment_monthly_trends",
            "user_monthly_metrics",
        }:
            filtered_tables[table_name] = _filter_frame_by_months(frame, "month", months)
        elif table_name == "daily_trends":
            filtered_tables[table_name] = _filter_frame_by_date_months(frame, "date", months)
        elif table_name == "weekly_trends":
            filtered_tables[table_name] = _filter_weekly_frame_by_months(frame, months)
        elif table_name == "llm_trend_metrics":
            filtered_tables[table_name] = _filter_llm_metric_frame_by_months(frame, months)
        else:
            filtered_tables[table_name] = frame.copy()
    return filtered_tables


def _series_first_row(frame: pd.DataFrame) -> pd.Series | None:
    """비어 있지 않은 DataFrame의 첫 번째 행을 Series로 반환한다."""
    if frame.empty:
        return None
    return frame.iloc[0]


def _series_last_row(frame: pd.DataFrame) -> pd.Series | None:
    """비어 있지 않은 DataFrame의 마지막 행을 Series로 반환한다."""
    if frame.empty:
        return None
    return frame.iloc[-1]


def _numeric_value(value: object, default: float = 0.0) -> float:
    """CSV 셀 값을 근거 카드 계산에 사용할 숫자로 안전하게 변환한다."""
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_numeric_value(value: object) -> float | None:
    """비어 있을 수 있는 CSV 셀 값을 선택적 숫자로 변환한다."""
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_value(value: object, default: str = "") -> str:
    """CSV 셀 값을 공백이 정리된 문자열로 변환한다."""
    if value is None or pd.isna(value):
        return default
    return str(value).strip()


def _period_slug(report_scope: UserReportScope) -> str:
    """보고서 기간 라벨을 evidence_card ID에 쓰기 좋은 문자열로 변환한다."""
    return report_scope.period.replace("-", "_")


def _card_issue_id(report_scope: UserReportScope, suffix: str) -> str:
    """보고서 타입과 기간, 카드 종류를 묶은 안정적인 issue_id를 만든다."""
    return f"{report_scope.report_type}_{_period_slug(report_scope)}_{suffix}"


def _format_card_metric_value(value: int | float | str) -> str:
    """근거 카드의 metric_value 필드를 문자열로 직렬화한다."""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_plain_number(value)
    return value


def _card_block(card: UserReportEvidenceCard) -> str:
    """근거 카드 모델을 Markdown 섹션 한 덩어리로 변환한다."""
    lines = [
        f"### evidence_card: {card.issue_id}",
        f"- issue_id: {card.issue_id}",
        f"- period_type: {card.period_type}",
        f"- period: {card.period}",
        f"- category: {card.category}",
        f"- keywords: {', '.join(card.keywords)}",
        f"- metric_name: {card.metric_name}",
        f"- metric_value: {_format_card_metric_value(card.metric_value)}",
        f"- metric_unit: {card.metric_unit}",
    ]
    if card.change_rate_percent is not None:
        lines.append(f"- change_rate_percent: {_format_plain_number(card.change_rate_percent)}")
    lines.extend(
        [
            f"- source_table: {card.source_table}",
            f"- usable_claim: {card.usable_claim}",
            f"- feedback_use: {card.feedback_use}",
            f"- personal_feedback_use_case: {card.personal_feedback_use_case}",
            f"- caution: {card.caution}",
        ]
    )
    return "\n".join(lines)


def _mean_numeric_column(frame: pd.DataFrame, column: str) -> float:
    """DataFrame 숫자 컬럼의 결측치를 제외한 평균값을 반환한다."""
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def _change_direction_label(change_rate: float) -> str:
    """전월 대비 변화율을 증가·감소·유지 라벨로 변환한다."""
    if change_rate > 0:
        return "증가"
    if change_rate < 0:
        return "감소"
    return "유지"


def _build_total_amount_card(
    report_scope: UserReportScope,
    monthly_trends: pd.DataFrame,
) -> UserReportEvidenceCard | None:
    """월간·분기 전체 소비 총액 근거 카드를 지표에서 결정적으로 만든다."""
    if monthly_trends.empty:
        return None

    last_row = _series_last_row(monthly_trends.sort_values("month"))
    if last_row is None:
        return None

    if report_scope.report_type == "monthly":
        amount = int(round(_numeric_value(last_row.get("total_amount"))))
        change_rate = _optional_numeric_value(last_row.get("prev_month_change_rate_percent"))
        change_phrase = (
            f" 전월 대비 {_format_number(change_rate)}% 변화했다."
            if change_rate is not None
            else " 전월 비교값은 없다."
        )
        usable_claim = (
            f"{report_scope.period} 전체 앱 사용자 승인 소비 총액은 "
            f"{_format_number(amount)}원이며,{change_phrase}"
        )
        metric_value: int | float | str = amount
        metric_name = "monthly_total_amount"
        metric_unit = "KRW"
    else:
        amount = int(round(float(monthly_trends["total_amount"].sum())))
        usable_claim = (
            f"{report_scope.period} 포함 월({', '.join(report_scope.months)})의 "
            f"전체 앱 사용자 승인 소비 총액 합계는 {_format_number(amount)}원이다."
        )
        change_rate = None
        metric_value = amount
        metric_name = "period_total_amount"
        metric_unit = "KRW"

    return UserReportEvidenceCard(
        issue_id=_card_issue_id(report_scope, "total_amount"),
        period_type=report_scope.report_type,
        period=report_scope.period,
        category="전체 소비 동향",
        keywords=("월간 소비", "총소비", "전월 대비", "전체 사용자 동향"),
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        change_rate_percent=change_rate,
        source_table="monthly_trends",
        usable_claim=usable_claim,
        feedback_use=(
            "개인 월간 피드백에서 사용자의 월간 총소비 수준을 전체 앱 사용자 흐름과 "
            "비교하는 보조 근거로 사용한다."
        ),
        personal_feedback_use_case=(
            "사용자의 월간 총지출이 전월보다 늘었거나 예산 대비 높은 수준일 때, "
            "개인 변화가 전체 사용자 소비 흐름과 같은 방향인지 비교한다."
        ),
        caution="전체 사용자 집계이므로 개인의 소비 증가나 감소 원인을 단정하지 않는다.",
    )


def _build_top_category_card(
    report_scope: UserReportScope,
    monthly_trends: pd.DataFrame,
    category_trends: pd.DataFrame,
) -> UserReportEvidenceCard | None:
    """월간·분기 최상위 카테고리 근거 카드를 지표에서 결정적으로 만든다."""
    if report_scope.report_type == "monthly":
        row = _series_last_row(monthly_trends.sort_values("month"))
        if row is None:
            return None
        category = _string_value(row.get("top_category"), "미상")
        amount = int(round(_numeric_value(row.get("top_category_amount"))))
        ratio = _numeric_value(row.get("top_category_ratio_percent"))
    else:
        if category_trends.empty:
            return None
        category_totals = (
            category_trends.groupby("category", as_index=False)
            .agg(total_amount=("total_amount", "sum"))
            .sort_values("total_amount", ascending=False)
        )
        row = _series_first_row(category_totals)
        if row is None:
            return None
        category = _string_value(row.get("category"), "미상")
        amount = int(round(_numeric_value(row.get("total_amount"))))
        total_amount = _numeric_value(category_totals["total_amount"].sum())
        ratio = (amount / total_amount * 100) if total_amount else 0.0

    return UserReportEvidenceCard(
        issue_id=_card_issue_id(report_scope, f"top_category_{category}"),
        period_type=report_scope.report_type,
        period=report_scope.period,
        category=category,
        keywords=(category, "카테고리 비중", "월간 소비", "전체 사용자 동향"),
        metric_name="top_category_amount",
        metric_value=amount,
        metric_unit="KRW",
        change_rate_percent=None,
        source_table="monthly_trends/category_monthly_trends",
        usable_claim=(
            f"{report_scope.period} 전체 앱 사용자 소비에서 {category} 카테고리는 "
            f"{_format_number(amount)}원, {_format_number(ratio)}% 비중의 주요 소비 항목이다."
        ),
        feedback_use=(
            f"개인 월간 피드백에서 {category} 지출이 두드러질 때 전체 사용자도 크게 쓰는 "
            "카테고리인지 비교하는 근거로 사용한다."
        ),
        personal_feedback_use_case=(
            f"사용자의 {category} 지출이 월간 상위 카테고리일 때, 해당 지출이 전체 사용자에게도 "
            "큰 공통 지출인지 개인 특이 지출인지 구분한다."
        ),
        caution="카테고리 비중은 전체 사용자 집계이므로 개인의 해당 지출을 낭비로 단정하지 않는다.",
    )


def _build_category_ratio_cards(
    report_scope: UserReportScope,
    category_trends: pd.DataFrame,
) -> tuple[UserReportEvidenceCard, ...]:
    """카테고리별 월간·기간 소비 비중 근거 카드를 생성한다."""
    if category_trends.empty or "category" not in category_trends.columns:
        return ()

    if report_scope.report_type == "monthly":
        ratio_rows = category_trends.copy()
        ratio_column = "monthly_ratio_percent"
        metric_name = "category_monthly_ratio_percent"
    else:
        if "total_amount" not in category_trends.columns:
            return ()
        ratio_rows = (
            category_trends.groupby("category", as_index=False)
            .agg(total_amount=("total_amount", "sum"))
            .sort_values("total_amount", ascending=False)
        )
        period_total_amount = _numeric_value(ratio_rows["total_amount"].sum())
        ratio_rows["period_ratio_percent"] = (
            ratio_rows["total_amount"] / period_total_amount * 100 if period_total_amount else 0.0
        )
        ratio_column = "period_ratio_percent"
        metric_name = "category_period_ratio_percent"

    if "total_amount" not in ratio_rows.columns or ratio_column not in ratio_rows.columns:
        return ()

    cards: list[UserReportEvidenceCard] = []
    sorted_rows = ratio_rows.sort_values("total_amount", ascending=False)
    for _, row in sorted_rows.iterrows():
        category = _string_value(row.get("category"), "미상")
        amount = int(round(_numeric_value(row.get("total_amount"))))
        ratio = _numeric_value(row.get(ratio_column))
        cards.append(
            UserReportEvidenceCard(
                issue_id=_card_issue_id(report_scope, f"category_ratio_{category}"),
                period_type=report_scope.report_type,
                period=report_scope.period,
                category=category,
                keywords=(
                    category,
                    "카테고리 월간 비중",
                    "절약 우선순위",
                    "전체 사용자 비교",
                ),
                metric_name=metric_name,
                metric_value=ratio,
                metric_unit="percent",
                change_rate_percent=None,
                source_table="category_monthly_trends",
                usable_claim=(
                    f"{report_scope.period} 전체 앱 사용자 소비에서 {category} 카테고리 비중은 "
                    f"{_format_number(ratio)}%이고 금액은 {_format_number(amount)}원이다."
                ),
                feedback_use=(
                    f"개인 월간 피드백에서 {category} 지출의 절약 우선순위를 정할 때 "
                    "전체 사용자 카테고리 비중과 비교하는 근거로 사용한다."
                ),
                personal_feedback_use_case=(
                    f"사용자의 {category} 지출 비중이 개인 월간 상위권이거나 전월 대비 증가했을 때, "
                    "전체 사용자에게도 큰 공통 지출인지 개인 특이 지출인지 구분한다."
                ),
                caution=(
                    "전체 사용자 비중이 높다는 사실만으로 개인 지출을 줄여야 한다고 단정하지 않는다."
                ),
            )
        )
    return tuple(cards)


def _build_monthly_category_change_cards(
    report_scope: UserReportScope,
    category_trends: pd.DataFrame,
) -> tuple[UserReportEvidenceCard, ...]:
    """월간 카테고리별 전월 대비 증가·감소 근거 카드를 생성한다."""
    required_columns = {
        "category",
        "total_amount",
        "prev_month_category_amount",
        "prev_month_change_rate_percent",
    }
    if category_trends.empty or not required_columns.issubset(category_trends.columns):
        return ()

    cards: list[UserReportEvidenceCard] = []
    rows = category_trends.copy()
    rows["absolute_change_rate"] = pd.to_numeric(
        rows["prev_month_change_rate_percent"], errors="coerce"
    ).abs()
    sorted_rows = rows.sort_values("absolute_change_rate", ascending=False)
    for _, row in sorted_rows.iterrows():
        change_rate = _optional_numeric_value(row.get("prev_month_change_rate_percent"))
        if change_rate is None or change_rate == 0:
            continue
        category = _string_value(row.get("category"), "미상")
        amount = int(round(_numeric_value(row.get("total_amount"))))
        previous_amount = int(round(_numeric_value(row.get("prev_month_category_amount"))))
        direction = _change_direction_label(change_rate)
        cards.append(
            UserReportEvidenceCard(
                issue_id=_card_issue_id(report_scope, f"category_change_{category}"),
                period_type=report_scope.report_type,
                period=report_scope.period,
                category=category,
                keywords=(
                    category,
                    "카테고리 전월 대비",
                    direction,
                    "절약 우선순위",
                    "전체 사용자 비교",
                ),
                metric_name="category_prev_month_change_rate_percent",
                metric_value=change_rate,
                metric_unit="percent",
                change_rate_percent=change_rate,
                source_table="category_monthly_trends",
                usable_claim=(
                    f"{report_scope.period} 전체 앱 사용자 {category} 소비는 "
                    f"{_format_number(amount)}원으로 전월 {_format_number(previous_amount)}원 대비 "
                    f"{_format_number(change_rate)}% {direction}했다."
                ),
                feedback_use=(
                    f"개인 월간 피드백에서 사용자의 {category} 지출 증감이 전체 사용자 변화와 "
                    "같은 방향인지 비교해 절약 우선순위를 조정하는 근거로 사용한다."
                ),
                personal_feedback_use_case=(
                    f"사용자의 {category} 지출이 전월 대비 {direction}했을 때, "
                    "개인 변화가 전체 사용자 공통 흐름인지 개인에게만 두드러진 변화인지 판단한다."
                ),
                caution=(
                    "전체 사용자 카테고리 변화율은 개인의 특정 가맹점 지출 원인을 직접 설명하지 않는다."
                ),
            )
        )
    return tuple(cards)


def _build_quarterly_category_change_cards(
    report_scope: UserReportScope,
    category_trends: pd.DataFrame,
) -> tuple[UserReportEvidenceCard, ...]:
    """분기 카테고리별 시작월 대비 종료월 변화 근거 카드를 생성한다."""
    required_columns = {"month", "category", "total_amount"}
    if category_trends.empty or not required_columns.issubset(category_trends.columns):
        return ()

    months = tuple(sorted(str(month) for month in category_trends["month"].dropna().unique()))
    if len(months) < 2:
        return ()

    first_month = months[0]
    last_month = months[-1]
    first_rows = category_trends[category_trends["month"].astype("string") == first_month][
        ["category", "total_amount"]
    ].rename(columns={"total_amount": "first_amount"})
    last_rows = category_trends[category_trends["month"].astype("string") == last_month][
        ["category", "total_amount"]
    ].rename(columns={"total_amount": "last_amount"})
    merged_rows = last_rows.merge(first_rows, on="category", how="outer").fillna(0.0)
    merged_rows["change_rate"] = merged_rows.apply(
        lambda row: (
            (_numeric_value(row.get("last_amount")) - _numeric_value(row.get("first_amount")))
            / _numeric_value(row.get("first_amount"))
            * 100
            if _numeric_value(row.get("first_amount"))
            else 0.0
        ),
        axis=1,
    )
    merged_rows["absolute_change_rate"] = merged_rows["change_rate"].abs()

    cards: list[UserReportEvidenceCard] = []
    sorted_rows = merged_rows.sort_values("absolute_change_rate", ascending=False)
    for _, row in sorted_rows.iterrows():
        change_rate = _numeric_value(row.get("change_rate"))
        if change_rate == 0:
            continue
        category = _string_value(row.get("category"), "미상")
        first_amount = int(round(_numeric_value(row.get("first_amount"))))
        last_amount = int(round(_numeric_value(row.get("last_amount"))))
        direction = _change_direction_label(change_rate)
        cards.append(
            UserReportEvidenceCard(
                issue_id=_card_issue_id(report_scope, f"category_change_{category}"),
                period_type=report_scope.report_type,
                period=report_scope.period,
                category=category,
                keywords=(
                    category,
                    "카테고리 분기 변화",
                    direction,
                    "절약 우선순위",
                    "전체 사용자 비교",
                ),
                metric_name="category_period_change_rate_percent",
                metric_value=change_rate,
                metric_unit="percent",
                change_rate_percent=change_rate,
                source_table="category_monthly_trends",
                usable_claim=(
                    f"{report_scope.period} 전체 앱 사용자 {category} 소비는 {first_month} "
                    f"{_format_number(first_amount)}원에서 {last_month} "
                    f"{_format_number(last_amount)}원으로 {_format_number(change_rate)}% {direction}했다."
                ),
                feedback_use=(
                    f"개인 월간 피드백에서 {category} 지출 변화가 분기 전체 흐름과 이어지는지 "
                    "판단하는 보조 근거로 사용한다."
                ),
                personal_feedback_use_case=(
                    f"사용자의 {category} 지출 증가가 한 달짜리 예외인지 분기 흐름과 맞물린 변화인지 "
                    "비교할 때 사용한다."
                ),
                caution="분기 변화는 월별 피크를 평균화할 수 있으므로 개인 월간 원인 판단에는 보조로만 쓴다.",
            )
        )
    return tuple(cards)


def _build_category_change_cards(
    report_scope: UserReportScope,
    category_trends: pd.DataFrame,
) -> tuple[UserReportEvidenceCard, ...]:
    """보고서 스코프에 맞는 카테고리 변화 근거 카드를 생성한다."""
    if report_scope.report_type == "monthly":
        return _build_monthly_category_change_cards(report_scope, category_trends)
    return _build_quarterly_category_change_cards(report_scope, category_trends)


def _build_payment_behavior_card(
    report_scope: UserReportScope,
    monthly_trends: pd.DataFrame,
    user_metrics: pd.DataFrame,
) -> UserReportEvidenceCard | None:
    """온라인·할부·야간·마찰없는 결제 행동 근거 카드를 지표에서 결정적으로 만든다."""
    if monthly_trends.empty:
        return None

    if report_scope.report_type == "monthly":
        row = _series_last_row(monthly_trends.sort_values("month"))
        if row is None:
            return None
        online_ratio = _numeric_value(row.get("online_ratio_percent"))
        installment_ratio = _numeric_value(row.get("installment_ratio_percent"))
        late_night_ratio = _numeric_value(row.get("late_night_ratio_percent"))
    else:
        online_ratio = _mean_numeric_column(monthly_trends, "online_ratio_percent")
        installment_ratio = _mean_numeric_column(monthly_trends, "installment_ratio_percent")
        late_night_ratio = _mean_numeric_column(monthly_trends, "late_night_ratio_percent")
    frictionless_ratio = _mean_numeric_column(user_metrics, "frictionless_spending_ratio_percent")

    return UserReportEvidenceCard(
        issue_id=_card_issue_id(report_scope, "payment_behavior"),
        period_type=report_scope.report_type,
        period=report_scope.period,
        category="결제 행동",
        keywords=(
            "온라인 결제",
            "할부 결제",
            "야간 소비",
            "마찰없는 결제",
            "결제 행동",
            "현금흐름 부담",
        ),
        metric_name="payment_behavior_ratio_percent",
        metric_value=(
            f"online={_format_plain_number(online_ratio)}, "
            f"installment={_format_plain_number(installment_ratio)}, "
            f"late_night={_format_plain_number(late_night_ratio)}, "
            f"frictionless={_format_plain_number(frictionless_ratio)}"
        ),
        metric_unit="percent",
        change_rate_percent=None,
        source_table="monthly_trends/user_monthly_metrics",
        usable_claim=(
            f"{report_scope.period} 전체 앱 사용자 소비에서 온라인 결제 비중은 "
            f"{_format_number(online_ratio)}%, 할부 결제 비중은 "
            f"{_format_number(installment_ratio)}%, 야간 소비 비중은 "
            f"{_format_number(late_night_ratio)}%, 사용자-월 평균 마찰없는 결제 비중은 "
            f"{_format_number(frictionless_ratio)}%이다."
        ),
        feedback_use=(
            "개인 월간 피드백에서 온라인·할부·야간·마찰없는 결제가 높을 때 결제 편의성, "
            "미래 현금흐름 부담, 결제 전 확인 장치 필요성을 설명하는 보조 근거로 사용한다."
        ),
        personal_feedback_use_case=(
            "사용자의 할부 비중, 온라인 비중, 야간 소비, 마찰없는 결제 비중 중 하나가 높을 때 "
            "전체 사용자 결제 행동과 비교해 절약 행동을 금액 축소보다 결제 방식 관리로 잡을지 판단한다."
        ),
        caution="결제 방식 비중만으로 개인의 충동구매나 재무 위험을 단정하지 않는다.",
    )


def _build_budget_risk_card(
    report_scope: UserReportScope,
    user_metrics: pd.DataFrame,
) -> UserReportEvidenceCard | None:
    """목표 사용률 임계값별 초과 사용자군 분포 카드를 지표에서 결정적으로 만든다."""
    if user_metrics.empty or "monthly_budget_usage_rate_percent" not in user_metrics.columns:
        return None

    usage = pd.to_numeric(user_metrics["monthly_budget_usage_rate_percent"], errors="coerce")
    threshold_counts = {
        threshold: int((usage >= threshold).sum()) for threshold in (100, 150, 200, 300)
    }
    metric_value = ", ".join(
        f"over_{threshold}={count}" for threshold, count in threshold_counts.items()
    )
    claim_counts = ", ".join(
        f"{threshold}% 이상 {count}건" for threshold, count in threshold_counts.items()
    )
    return UserReportEvidenceCard(
        issue_id=_card_issue_id(report_scope, "budget_usage_risk"),
        period_type=report_scope.report_type,
        period=report_scope.period,
        category="예산 사용률 위험 신호",
        keywords=("목표 사용률", "예산 초과", "위험 신호", "사용자 월"),
        metric_name="budget_usage_threshold_distribution",
        metric_value=metric_value,
        metric_unit="user_months",
        change_rate_percent=None,
        source_table="user_monthly_metrics",
        usable_claim=(
            f"{report_scope.period} 전체 앱 사용자-월의 목표 사용률 초과 분포는 {claim_counts}이다."
        ),
        feedback_use=(
            "개인 월간 피드백에서 목표 소비 한도를 크게 넘긴 사용자의 상황을 전체 사용자군의 "
            "예산 초과 분포와 비교하는 보조 근거로 사용한다."
        ),
        personal_feedback_use_case=(
            "사용자의 목표 사용률이 100%, 150%, 200%, 300% 임계값 중 어디에 걸리는지 확인해 "
            "일반 초과, 주의 초과, 집중 관리, 즉시 개입 수준을 구분한다."
        ),
        caution="목표 사용률은 사용자별 목표 한도 설정에 영향을 받으므로 과소비로 바로 단정하지 않는다.",
    )


def _build_user_report_evidence_cards(
    report_scope: UserReportScope,
    scoped_tables: dict[str, pd.DataFrame],
) -> tuple[UserReportEvidenceCard, ...]:
    """스코프별 지표 테이블에서 RAG용 근거 카드 묶음을 결정적으로 생성한다."""
    candidate_cards: list[UserReportEvidenceCard | None] = [
        _build_total_amount_card(report_scope, scoped_tables["monthly_trends"]),
        _build_top_category_card(
            report_scope,
            scoped_tables["monthly_trends"],
            scoped_tables["category_monthly_trends"],
        ),
    ]
    cards: list[UserReportEvidenceCard] = [card for card in candidate_cards if card is not None]
    cards.extend(
        _build_category_ratio_cards(report_scope, scoped_tables["category_monthly_trends"])
    )
    cards.extend(
        _build_category_change_cards(report_scope, scoped_tables["category_monthly_trends"])
    )
    final_candidate_cards = [
        _build_payment_behavior_card(
            report_scope,
            scoped_tables["monthly_trends"],
            scoped_tables["user_monthly_metrics"],
        ),
        _build_budget_risk_card(report_scope, scoped_tables["user_monthly_metrics"]),
    ]
    cards.extend(card for card in final_candidate_cards if card is not None)
    return tuple(cards)


def _period_bounds(report_scope: UserReportScope) -> tuple[str, str]:
    """보고서 기간 라벨에서 시작일과 종료일을 분리한다."""
    if " ~ " not in report_scope.period_label:
        return report_scope.period_label, report_scope.period_label
    start, end = report_scope.period_label.split(" ~ ", maxsplit=1)
    return start, end


def _build_rag_frontmatter(
    *,
    metrics_dir: Path,
    report_scope: UserReportScope,
    report_date: str,
) -> str:
    """Markdown RAG 문서 상단의 YAML front matter를 만든다."""
    period_start, period_end = _period_bounds(report_scope)
    lines = [
        "---",
        "document_kind: user_report",
        f"report_type: {report_scope.report_type}",
        f"report_period: {report_scope.period}",
        f"period_start: {period_start}",
        f"period_end: {period_end}",
        "included_months:",
    ]
    lines.extend(f"  - {month}" for month in report_scope.months)
    lines.extend(
        [
            f"source_metrics_dir: {metrics_dir}",
            f"generated_at: {report_date}",
            "---",
        ]
    )
    return "\n".join(lines)


def _build_rag_metadata_section(
    *,
    metrics_dir: Path,
    report_scope: UserReportScope,
    evidence_cards: tuple[UserReportEvidenceCard, ...],
) -> str:
    """RAG 검색 필터와 청크 식별에 필요한 문서 메타데이터 섹션을 만든다."""
    keywords = sorted({keyword for card in evidence_cards for keyword in card.keywords})
    lines = [
        "## RAG 문서 메타데이터",
        "- document_kind: user_report",
        f"- report_type: {report_scope.report_type}",
        f"- report_period: {report_scope.period}",
        f"- period_label: {report_scope.period_label}",
        f"- included_months: {', '.join(report_scope.months)}",
        f"- source_metrics_dir: {metrics_dir}",
        f"- evidence_card_count: {len(evidence_cards)}",
        f"- keywords: {', '.join(keywords)}",
    ]
    return "\n".join(lines)


def _build_rag_search_summary(evidence_cards: tuple[UserReportEvidenceCard, ...]) -> str:
    """근거 카드의 핵심 claim만 검색용 짧은 요약 bullet로 만든다."""
    if not evidence_cards:
        return "## 검색용 요약\n- 생성 가능한 근거 카드가 없습니다."
    lines = ["## 검색용 요약"]
    lines.extend(f"- {card.usable_claim}" for card in evidence_cards)
    return "\n".join(lines)


def _build_rag_usage_section(evidence_cards: tuple[UserReportEvidenceCard, ...]) -> str:
    """월간 피드백 생성기가 근거를 사용할 때 지킬 제한을 문서화한다."""
    if not evidence_cards:
        return (
            "## 피드백 사용 제한\n- 근거 카드가 없으므로 개인 피드백 보강 근거로 사용하지 않는다."
        )

    lines = ["## 개인 피드백 사용법"]
    for card in evidence_cards:
        lines.append(f"- {card.issue_id}: {card.personal_feedback_use_case}")
    lines.append("")
    lines.append("## 피드백 사용 제한")
    for card in evidence_cards:
        lines.append(f"- {card.issue_id}: {card.caution}")
    return "\n".join(lines)


def _build_user_report_rag_markdown(
    *,
    metrics_dir: Path,
    report_scope: UserReportScope,
    report_date: str,
) -> str:
    """지표 CSV에서 LLM 없이 RAG용 사용자 보고서 Markdown을 결정적으로 생성한다."""
    tables = read_metric_tables(metrics_dir)
    scoped_tables = _filter_metric_tables_for_scope(tables, report_scope)
    evidence_cards = _build_user_report_evidence_cards(report_scope, scoped_tables)
    card_blocks = [_card_block(card) for card in evidence_cards]
    return "\n\n".join(
        [
            _build_rag_frontmatter(
                metrics_dir=metrics_dir,
                report_scope=report_scope,
                report_date=report_date,
            ),
            "# 사용자 소비 RAG 근거 문서",
            _build_rag_metadata_section(
                metrics_dir=metrics_dir,
                report_scope=report_scope,
                evidence_cards=evidence_cards,
            ),
            _build_rag_search_summary(evidence_cards),
            "## RAG 근거 카드\n" + "\n\n".join(card_blocks),
            _build_rag_usage_section(evidence_cards),
        ]
    )


def _scope_context(report_scope: UserReportScope | None) -> str | None:
    """RAG 보고서가 검색될 때 활용할 기간 메타데이터와 근거 카드 작성 계약을 만든다."""
    if report_scope is None:
        return None

    keywords = ", ".join(
        [
            report_scope.report_type,
            report_scope.period,
            "사용자 소비 동향",
            "월간 소비",
            "카테고리 변화",
            "결제 행동",
            "위험 신호",
        ]
    )
    months = ", ".join(report_scope.months)
    return f"""
# RAG 문서 메타데이터
- report_type: {report_scope.report_type}
- report_period: {report_scope.period}
- period_label: {report_scope.period_label}
- included_months: {months}
- keywords: {keywords}

## RAG 근거 카드 작성 계약
- 각 카드는 `issue_id`, `period_type`, `period`, `category`, `keywords`, `usable_claim`, `feedback_use`, `personal_feedback_use_case`, `caution` 필드를 포함한다.
- `usable_claim`은 아래 지표 요약에서 확인되는 수치와 추세만 근거로 작성한다.
- `feedback_use`는 월간 피드백에서 문장 보강, 비교 근거, 주의 신호 설명에 사용할 수 있는 형태로 작성한다.
- `personal_feedback_use_case`는 어떤 개인 소비 상황에서 이 근거를 사용할지 명확히 작성한다.
- `caution`에는 전체 사용자 동향을 개인에게 과도하게 일반화하지 말라는 제한을 남긴다.
""".strip()


def _monthly_context(monthly_trends: pd.DataFrame) -> str:
    """월별 소비 총액과 결제 행동 핵심 컬럼을 컨텍스트로 만든다."""
    if monthly_trends.empty:
        return "## 월별 핵심 지표\n(데이터 없음)"
    columns = [
        column
        for column in [
            "month",
            "active_user_count",
            "approved_transaction_count",
            "total_amount",
            "prev_month_change_rate_percent",
            "average_amount_per_active_user",
            "budget_usage_rate_percent",
            "online_ratio_percent",
            "installment_ratio_percent",
            "late_night_ratio_percent",
            "top_category",
            "top_category_ratio_percent",
        ]
        if column in monthly_trends.columns
    ]
    return "## 월별 핵심 지표\n" + _frame_to_csv_block(monthly_trends[columns], max_rows=24)


def _category_context(category_trends: pd.DataFrame) -> str:
    """카테고리 합산 순위와 기간 시작·종료 변화 요약을 만든다."""
    if category_trends.empty:
        return "## 카테고리 합산 및 변화\n(데이터 없음)"

    total_rows = (
        category_trends.groupby("category", as_index=False)
        .agg(
            total_amount=("total_amount", "sum"),
            average_ratio_percent=("monthly_ratio_percent", "mean"),
            active_user_average=("active_user_count", "mean"),
            transaction_count=("transaction_count", "sum"),
        )
        .sort_values("total_amount", ascending=False)
    )

    change_block = ""
    if {"month", "category", "total_amount", "monthly_ratio_percent"}.issubset(
        category_trends.columns
    ):
        first_month = str(category_trends["month"].min())
        last_month = str(category_trends["month"].max())
        first = category_trends[category_trends["month"] == first_month][
            ["category", "total_amount", "monthly_ratio_percent"]
        ].rename(
            columns={
                "total_amount": "start_amount",
                "monthly_ratio_percent": "start_ratio_percent",
            }
        )
        last = category_trends[category_trends["month"] == last_month][
            ["category", "total_amount", "monthly_ratio_percent"]
        ].rename(
            columns={
                "total_amount": "end_amount",
                "monthly_ratio_percent": "end_ratio_percent",
            }
        )
        changes = last.merge(first, on="category", how="left")
        changes["amount_change"] = changes["end_amount"] - changes["start_amount"]
        changes["ratio_point_change"] = (
            changes["end_ratio_percent"] - changes["start_ratio_percent"]
        )
        change_block = "\n\n### 시작월 대비 종료월 변화\n" + _frame_to_csv_block(
            changes.sort_values("amount_change", ascending=False),
            max_rows=12,
        )

    return "## 카테고리 합산\n" + _frame_to_csv_block(total_rows, max_rows=12) + change_block


def _segment_context(segment_trends: pd.DataFrame) -> str:
    """연령대·성별 세그먼트별 소비 차이를 컨텍스트로 만든다."""
    if segment_trends.empty:
        return "## 세그먼트별 지표\n(데이터 없음)"
    segment = (
        segment_trends.groupby(["age_group", "gender"], as_index=False)
        .agg(
            total_amount=("total_amount", "sum"),
            average_amount_per_active_user=("average_amount_per_active_user", "mean"),
            active_user_average=("active_user_count", "mean"),
            budget_usage_rate_percent=("budget_usage_rate_percent", "mean"),
        )
        .sort_values("average_amount_per_active_user", ascending=False)
    )
    return "## 세그먼트별 지표\n" + _frame_to_csv_block(segment, max_rows=12)


def _user_risk_context(user_metrics: pd.DataFrame) -> str:
    """사용자-월 목표 초과와 결제 행동 위험군 요약을 만든다."""
    if user_metrics.empty:
        return "## 사용자 위험군\n(데이터 없음)"

    usage = user_metrics["monthly_budget_usage_rate_percent"]
    rows: list[dict[str, object]] = []
    for threshold in [100, 150, 200, 300]:
        filtered = user_metrics[usage >= threshold]
        rows.append(
            {
                "condition": f"목표 사용률 {threshold}% 이상",
                "user_month_count": int(len(filtered)),
                "unique_user_count": int(filtered["user_id"].nunique()),
            }
        )
    risk_summary = pd.DataFrame(rows)
    top_columns = [
        column
        for column in [
            "user_id",
            "month",
            "monthly_total_amount",
            "monthly_budget_usage_rate_percent",
            "top_category",
            "installment_ratio_percent",
            "frictionless_spending_ratio_percent",
        ]
        if column in user_metrics.columns
    ]
    top_users = user_metrics.sort_values(
        "monthly_budget_usage_rate_percent",
        ascending=False,
    )[top_columns]

    return (
        "## 사용자 위험군\n"
        + _frame_to_csv_block(risk_summary, max_rows=8)
        + "\n\n### 목표 사용률 상위 사용자-월\n"
        + _frame_to_csv_block(top_users, max_rows=15)
    )


def _peak_context(weekly_trends: pd.DataFrame, daily_trends: pd.DataFrame) -> str:
    """주간·일간 소비 피크 구간을 컨텍스트로 만든다."""
    weekly_columns = [
        column
        for column in [
            "week_start",
            "week_end",
            "total_amount",
            "weekend_spending_ratio_percent",
            "weekend_overspending_index",
        ]
        if column in weekly_trends.columns
    ]
    daily_columns = [
        column
        for column in ["date", "total_amount", "online_ratio_percent", "late_night_ratio_percent"]
        if column in daily_trends.columns
    ]
    weekly_top = (
        weekly_trends.sort_values("total_amount", ascending=False)[weekly_columns]
        if not weekly_trends.empty and weekly_columns
        else pd.DataFrame()
    )
    daily_top = (
        daily_trends.sort_values("total_amount", ascending=False)[daily_columns]
        if not daily_trends.empty and daily_columns
        else pd.DataFrame()
    )
    return (
        "## 주간 피크\n"
        + _frame_to_csv_block(weekly_top, max_rows=8)
        + "\n\n## 일간 피크\n"
        + _frame_to_csv_block(daily_top, max_rows=8)
    )


def _llm_metric_context(llm_metrics: pd.DataFrame) -> str:
    """LLM용 long-form 지표를 보조 컨텍스트로 압축한다."""
    if llm_metrics.empty:
        return "## LLM Long-form 지표\n(데이터 없음)"
    columns = [
        column
        for column in [
            "period_type",
            "period",
            "metric_name",
            "metric_value",
            "metric_unit",
            "change_rate_percent",
            "trend_label",
            "context",
        ]
        if column in llm_metrics.columns
    ]
    return "## LLM Long-form 지표\n" + _frame_to_csv_block(llm_metrics[columns], max_rows=40)


def build_metric_context(
    metrics_dir: Path,
    *,
    report_scope: UserReportScope | None = None,
) -> str:
    """지표 CSV 묶음을 LangChain 보고서 생성용 한국어 컨텍스트로 구성한다."""
    tables = read_metric_tables(metrics_dir)
    scoped_tables = _filter_metric_tables_for_scope(tables, report_scope)
    period_label = (
        report_scope.period_label
        if report_scope is not None
        else infer_period_label(metrics_dir, scoped_tables)
    )
    scope_context = _scope_context(report_scope)
    blocks = [scope_context] if scope_context is not None else []
    blocks.extend(
        [
            f"# 지표 기간\n{period_label}",
            _monthly_context(scoped_tables["monthly_trends"]),
            _category_context(scoped_tables["category_monthly_trends"]),
            _segment_context(scoped_tables["segment_monthly_trends"]),
            _user_risk_context(scoped_tables["user_monthly_metrics"]),
            _peak_context(scoped_tables["weekly_trends"], scoped_tables["daily_trends"]),
            _llm_metric_context(scoped_tables["llm_trend_metrics"]),
        ]
    )
    return "\n\n".join(blocks)


def _chart_links(metrics_dir: Path, report_dir: Path) -> str:
    """보고서 Markdown에서 사용할 수 있는 차트 상대 경로 목록을 만든다."""
    charts_dir = metrics_dir / "charts"
    if not charts_dir.exists():
        return "(차트 파일 없음)"
    chart_paths = sorted(charts_dir.glob("*.png"))
    if not chart_paths:
        return "(차트 파일 없음)"
    lines: list[str] = []
    for chart_path in chart_paths:
        relative_path = os.path.relpath(chart_path, start=report_dir)
        lines.append(f"- {chart_path.stem}: {relative_path}")
    return "\n".join(lines)


def build_user_trend_report_prompt() -> ChatPromptTemplate:
    """사용자 소비 동향 보고서 작성을 위한 한국어 LangChain 프롬프트를 만든다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Catcher 서비스의 사용자 소비 데이터 분석가다. "
                "제공된 지표 CSV 요약만 근거로 한국어 Markdown 보고서를 작성한다. "
                "없는 수치를 만들지 말고, 추론은 지표에서 자연스럽게 도출되는 범위로 제한한다.",
            ),
            (
                "human",
                """
아래 지표 요약을 바탕으로 사용자 소비 동향 보고서를 작성해줘.

작성 기준:
- Markdown 문서만 출력한다.
- 제목은 반드시 `{report_title}`로 시작한다.
- 작성일, 지표 기간, 사용 지표 디렉터리를 포함한다.
- 핵심 요약, 월별 동향, 카테고리 구조, 결제 행동, 세그먼트별 관찰, 사용자 위험군, 주간/일간 피크, LLM 활용 포인트, 권장 액션을 포함한다.
- RAG 검색 품질을 위해 문서 상단에 report_type, report_period, period_label, keywords 메타데이터를 사람이 읽을 수 있는 목록으로 포함한다.
- `## RAG 근거 카드` 섹션을 반드시 포함하고, 각 카드는 issue_id, period_type, period, category, keywords, usable_claim, feedback_use, personal_feedback_use_case, caution 필드를 가진다.
- RAG 근거 카드는 최종 월간 피드백의 보조 근거로 쓰일 수 있게 카테고리 변화, 결제 행동, 위험 신호, 전체 사용자 동향 비교를 중심으로 작성한다.
- 숫자는 가능한 한 원문 지표의 값을 사용하고, 금액·비율·건수를 구체적으로 쓴다.
- 차트가 필요하면 아래 차트 상대 경로를 Markdown 이미지 링크로 사용한다.
- 보고서 문체는 단정하고 분석적으로 작성한다.

작성일: {report_date}
지표 디렉터리: {metrics_dir}
차트 상대 경로:
{chart_links}

지표 요약:
{metric_context}
""".strip(),
            ),
        ]
    )


def build_user_trend_report_chain(
    settings: Settings | None = None,
    *,
    llm: BaseChatModel | None = None,
    temperature: float = 0.2,
):
    """프롬프트, 채팅 모델, 문자열 파서를 연결한 보고서 생성 체인을 만든다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    return build_user_trend_report_prompt() | chat_model | StrOutputParser()


def _report_output_path(
    *,
    metrics_dir: Path,
    output_root: Path,
    report_filename: str,
    report_scope: UserReportScope | None = None,
) -> Path:
    """지표 기간 디렉터리와 동일한 이름의 보고서 출력 경로를 만든다."""
    if report_scope is not None and report_scope.report_type == "monthly":
        return output_root / DEFAULT_MONTHLY_REPORT_DIRNAME / f"{report_scope.period}.md"
    if report_scope is not None and report_scope.report_type == "quarterly":
        return output_root / DEFAULT_QUARTERLY_REPORT_DIRNAME / f"{report_scope.period}.md"

    output_dir = output_root / metrics_dir.name
    return output_dir / report_filename


def generate_user_trend_report_from_metrics(
    *,
    metrics_dir: Path,
    output_root: Path = DEFAULT_REPORT_ROOT,
    report_filename: str = DEFAULT_REPORT_FILENAME,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    temperature: float = 0.2,
    report_date: str | None = None,
    report_scope: UserReportScope | None = None,
) -> ReportGenerationResult:
    """지표 디렉터리를 읽어 LangChain으로 사용자 동향 Markdown 보고서를 생성한다."""
    output_path = _report_output_path(
        metrics_dir=metrics_dir,
        output_root=output_root,
        report_filename=report_filename,
        report_scope=report_scope,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_context = build_metric_context(metrics_dir, report_scope=report_scope)
    chart_links = _chart_links(metrics_dir, output_path.parent)
    chain = build_user_trend_report_chain(settings=settings, llm=llm, temperature=temperature)
    report_title = report_scope.title if report_scope is not None else "# 사용자 소비 동향 보고서"
    report_markdown = cast(
        str,
        chain.invoke(
            {
                "report_title": report_title,
                "report_date": report_date or date.today().isoformat(),
                "metrics_dir": str(metrics_dir),
                "chart_links": chart_links,
                "metric_context": metric_context,
            }
        ),
    )
    output_path.write_text(report_markdown.strip() + "\n", encoding="utf-8")
    return ReportGenerationResult(
        metrics_dir=metrics_dir,
        output_path=output_path,
        report_markdown=report_markdown,
        report_scope=report_scope,
    )


def generate_user_trend_rag_report_from_metrics(
    *,
    metrics_dir: Path,
    report_scope: UserReportScope,
    output_root: Path = DEFAULT_REPORT_ROOT,
    report_filename: str = DEFAULT_REPORT_FILENAME,
    report_date: str | None = None,
) -> ReportGenerationResult:
    """지표 디렉터리에서 LLM 없이 RAG용 월간·분기 보고서를 생성한다."""
    output_path = _report_output_path(
        metrics_dir=metrics_dir,
        output_root=output_root,
        report_filename=report_filename,
        report_scope=report_scope,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_report_date = report_date or date.today().isoformat()
    report_markdown = _build_user_report_rag_markdown(
        metrics_dir=metrics_dir,
        report_scope=report_scope,
        report_date=effective_report_date,
    )
    output_path.write_text(report_markdown.strip() + "\n", encoding="utf-8")
    return ReportGenerationResult(
        metrics_dir=metrics_dir,
        output_path=output_path,
        report_markdown=report_markdown,
        report_scope=report_scope,
    )


def generate_user_trend_rag_reports_from_metrics(
    *,
    metrics_dir: Path,
    output_root: Path = DEFAULT_REPORT_ROOT,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    temperature: float = 0.2,
    report_date: str | None = None,
    include_monthly: bool = True,
    include_quarterly: bool = True,
) -> ReportBatchGenerationResult:
    """지표 디렉터리에서 RAG용 월간·분기 사용자 소비 보고서를 일괄 생성한다."""
    scopes = build_user_report_scopes(
        metrics_dir,
        include_monthly=include_monthly,
        include_quarterly=include_quarterly,
    )
    reports = tuple(
        generate_user_trend_rag_report_from_metrics(
            metrics_dir=metrics_dir,
            output_root=output_root,
            report_date=report_date,
            report_scope=scope,
        )
        for scope in scopes
    )
    return ReportBatchGenerationResult(metrics_dir=metrics_dir, reports=reports)


def generate_latest_user_trend_report(
    *,
    metrics_dir: Path | None = None,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    output_root: Path = DEFAULT_REPORT_ROOT,
    report_filename: str = DEFAULT_REPORT_FILENAME,
    settings: Settings | None = None,
    temperature: float = 0.2,
) -> ReportGenerationResult:
    """명시 지표 디렉터리 또는 최신 지표 디렉터리로 보고서를 생성한다."""
    resolved_metrics_dir = resolve_metrics_dir(metrics_dir, metrics_root=metrics_root)
    return generate_user_trend_report_from_metrics(
        metrics_dir=resolved_metrics_dir,
        output_root=output_root,
        report_filename=report_filename,
        settings=settings,
        temperature=temperature,
    )


def _parse_args() -> argparse.Namespace:
    """CLI에서 지표 디렉터리와 보고서 출력 옵션을 파싱한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=None)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--report-filename", default=DEFAULT_REPORT_FILENAME)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--report-scope",
        choices=("period", "monthly", "quarterly", "rag", "all"),
        default="period",
        help=(
            "period는 기존 기간 전체 보고서, monthly/quarterly/rag는 RAG용 기간 보고서, "
            "all은 둘 다 생성"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI 진입점으로 LangChain 보고서를 생성하고 저장 경로를 출력한다."""
    args = _parse_args()
    resolved_metrics_dir = resolve_metrics_dir(args.metrics_dir, metrics_root=args.metrics_root)

    if args.report_scope in {"monthly", "quarterly", "rag", "all"}:
        batch_result = generate_user_trend_rag_reports_from_metrics(
            metrics_dir=resolved_metrics_dir,
            output_root=args.output_root,
            temperature=args.temperature,
            include_monthly=args.report_scope in {"monthly", "rag", "all"},
            include_quarterly=args.report_scope in {"quarterly", "rag", "all"},
        )
        print(f"Metrics directory: {batch_result.metrics_dir}")
        for report in batch_result.reports:
            print(f"Saved user trend RAG report to: {report.output_path}")

    if args.report_scope in {"period", "all"}:
        result = generate_user_trend_report_from_metrics(
            metrics_dir=resolved_metrics_dir,
            output_root=args.output_root,
            report_filename=args.report_filename,
            temperature=args.temperature,
        )
        print(f"Metrics directory: {result.metrics_dir}")
        print(f"Saved user trend report to: {result.output_path}")


if __name__ == "__main__":
    main()
