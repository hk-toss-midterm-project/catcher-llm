from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DEV_PAGE_METADATA: dict[str, tuple[str, str]] = {
    "01_chat": ("Chat", "💬"),
    "02_retriever_probe": ("Retriever 테스트", "🔎"),
    "03_document_rag": ("문서 RAG", "📄"),
    "04_daily_analysis": ("일일 소비 분석", "📊"),
    "05_daily_interpretation": ("일일 소비 해석 체인", "🧭"),
    "06_daily_feedback": ("일일 피드백", "📣"),
    "07_weekly_analysis": ("주간 소비 분석", "🗓️"),
    "08_weekly_interpretation": ("주간 소비 해석 체인", "🧭"),
    "09_weekly_feedback": ("주간 피드백", "🧾"),
    "10_monthly_analysis": ("월간 소비 분석", "📈"),
    "11_monthly_interpretation": ("월간 소비 해석 체인", "🧭"),
    "12_monthly_feedback": ("월간 피드백", "🧾"),
    "13_daily_report_rim": ("일일 리포트", "📝"),
    "14_weekly_report_rim": ("주간 보고서", "🗓️"),
    "15_monthly_report_rim": ("월간 보고서", "📈"),
    "16_user_trend_report": ("사용자 동향 보고서 생성", "📑"),
    "17_daily_feedback_timing": ("일일 피드백 소요 시간", "⏱️"),
    "18_daily_interpretation_compare": ("일일 해석 방식 비교", "🧪"),
    "19_daily_feedback_unified": ("일일 통합 피드백", "📣"),
    "20_model_comparison": ("GPT 모델 성능 비교", "🔬"),
    "21_ragas": ("RAGAS 평가", "🧪"),
    "22_monthly_rag_evaluation": ("월간 피드백 RAG 평가", "🧪"),
}


@dataclass(frozen=True)
class DevPageSpec:
    path: Path
    title: str
    icon: str
    default: bool = False


def get_repo_root() -> Path:
    """현재 모듈 파일 위치를 기준으로 저장소 루트 경로를 계산한다."""
    return Path(__file__).resolve().parents[3]


def _get_dev_pages_dir() -> Path:
    """개발용 Streamlit 페이지 스크립트가 모여 있는 디렉터리 경로를 반환한다."""
    return get_repo_root() / "dev_pages"


def _prettify_title(stem: str) -> str:
    """파일 stem에서 접두 번호를 제거하고 화면 표시용 제목으로 다듬는다."""
    stem = stem.split("_", maxsplit=1)[-1]
    return stem.replace("_", " ").title()


def get_dev_page_specs() -> list[DevPageSpec]:
    """개발용 페이지 파일들을 읽어 사이드바 구성에 쓸 명세 목록으로 변환한다."""
    specs: list[DevPageSpec] = []

    for index, path in enumerate(sorted(_get_dev_pages_dir().glob("*.py"))):
        title, icon = _DEV_PAGE_METADATA.get(
            path.stem,
            (_prettify_title(path.stem), "🧪"),
        )
        specs.append(
            DevPageSpec(
                path=path,
                title=title,
                icon=icon,
                default=index == 0,
            )
        )

    return specs
