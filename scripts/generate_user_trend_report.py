"""LangChain으로 사용자 동향 지표 CSV를 읽어 Markdown 보고서를 생성한다."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model

DEFAULT_METRICS_ROOT = Path("data/processed/user_trend_metrics")
DEFAULT_REPORT_ROOT = Path("data/raw/markdown/users_report")
DEFAULT_REPORT_FILENAME = "users_trend_report.md"

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
class ReportGenerationResult:
    """생성된 보고서 경로와 사용한 지표 디렉터리를 함께 반환한다."""

    metrics_dir: Path
    output_path: Path
    report_markdown: str


def _format_number(value: float | int) -> str:
    """보고서 컨텍스트에서 읽기 쉽도록 숫자에 천 단위 구분자를 붙인다."""
    return f"{float(value):,.4f}".rstrip("0").rstrip(".")


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


def _frame_to_csv_block(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    """LLM 입력에 넣기 좋은 작은 CSV 블록으로 DataFrame을 직렬화한다."""
    if frame.empty:
        return "(데이터 없음)"
    return frame.head(max_rows).to_csv(index=False).strip()


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


def build_metric_context(metrics_dir: Path) -> str:
    """지표 CSV 묶음을 LangChain 보고서 생성용 한국어 컨텍스트로 구성한다."""
    tables = read_metric_tables(metrics_dir)
    period_label = infer_period_label(metrics_dir, tables)
    blocks = [
        f"# 지표 기간\n{period_label}",
        _monthly_context(tables["monthly_trends"]),
        _category_context(tables["category_monthly_trends"]),
        _segment_context(tables["segment_monthly_trends"]),
        _user_risk_context(tables["user_monthly_metrics"]),
        _peak_context(tables["weekly_trends"], tables["daily_trends"]),
        _llm_metric_context(tables["llm_trend_metrics"]),
    ]
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
- 제목은 반드시 `# 사용자 소비 동향 보고서`로 시작한다.
- 작성일, 지표 기간, 사용 지표 디렉터리를 포함한다.
- 핵심 요약, 월별 동향, 카테고리 구조, 결제 행동, 세그먼트별 관찰, 사용자 위험군, 주간/일간 피크, LLM 활용 포인트, 권장 액션을 포함한다.
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
) -> Path:
    """지표 기간 디렉터리와 동일한 이름의 보고서 출력 경로를 만든다."""
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
) -> ReportGenerationResult:
    """지표 디렉터리를 읽어 LangChain으로 사용자 동향 Markdown 보고서를 생성한다."""
    output_path = _report_output_path(
        metrics_dir=metrics_dir,
        output_root=output_root,
        report_filename=report_filename,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_context = build_metric_context(metrics_dir)
    chart_links = _chart_links(metrics_dir, output_path.parent)
    chain = build_user_trend_report_chain(settings=settings, llm=llm, temperature=temperature)
    report_markdown = cast(
        str,
        chain.invoke(
            {
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
    )


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
    return parser.parse_args()


def main() -> None:
    """CLI 진입점으로 LangChain 보고서를 생성하고 저장 경로를 출력한다."""
    args = _parse_args()
    result = generate_latest_user_trend_report(
        metrics_dir=args.metrics_dir,
        metrics_root=args.metrics_root,
        output_root=args.output_root,
        report_filename=args.report_filename,
        temperature=args.temperature,
    )
    print(f"Metrics directory: {result.metrics_dir}")
    print(f"Saved user trend report to: {result.output_path}")


if __name__ == "__main__":
    main()
