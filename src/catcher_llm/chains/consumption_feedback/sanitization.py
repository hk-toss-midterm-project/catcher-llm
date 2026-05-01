from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from catcher_llm.schemas.consumption_feedback.analysis_outputs import (
    CauseAnalysisResult,
    InterventionTarget,
    PatternAnalysisResult,
    ProblemAnalysisResult,
)
from catcher_llm.schemas.consumption_feedback.base import EvidenceItem, SpendingFinding

_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;:]*[A-Za-z]")
_PATH_TOKEN_PATTERN = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")
_DECREASE_WORDS = ("감소", "급감", "줄")
_LONG_TERM_WORDS = ("지속", "장기")
_TREND_PATH_HINTS = (
    "recent_4week",
    "same_week_last_month",
    "recent_3month",
    "trend",
    "weekly_comparisons",
    "monthly_comparisons",
)
_FIXED_COST_WORDS = ("납부", "관리비", "아파트관리비", "전기", "한국전력", "고정비")
_FIELD_CAUSE_NAMES = {
    "habitual_causes",
    "reward_causes",
    "stress_causes",
    "convenience_causes",
    "small_accumulation_causes",
    "one_off_high_spending_causes",
    "fixed_cost_timing_causes",
    "period_concentration_causes",
    "money_leaks",
    "saving_blockers",
    "fixed_cost_issues",
    "variable_cost_issues",
    "short_term_problem_spending",
    "long_term_problem_spending",
}
_CATEGORY_SUMMARY_INDEX_PATTERN = re.compile(r"category_summary\[(\d+)\]")
_HIGH_SPENDING_ITEM_INDEX_PATTERN = re.compile(r"waste_detection\.high_spending\.items\[(\d+)\]")
_WEEKDAY_LABELS = {
    "월": "월요일",
    "화": "화요일",
    "수": "수요일",
    "목": "목요일",
    "금": "금요일",
    "토": "토요일",
    "일": "일요일",
}
_WEEKDAY_LABELS_BY_INDEX = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")
_CATEGORY_PREFIX = {
    "교육": "education",
    "납부": "fixed_cost",
    "식비": "food",
    "쇼핑": "shopping",
    "여가": "leisure",
    "교통": "transportation",
    "생활": "living",
    "사교": "social",
}


def sanitize_spending_analysis_payload(payload: dict[str, object]) -> dict[str, object]:
    """LLM 해석 결과에서 원본 JSON으로 반증되는 패턴·문제·원인·개입 타겟을 정리한다."""
    raw_json = _load_json_object(payload.get("raw_json"))
    indicator_json = _load_json_object(payload.get("indicator_json"))
    sanitized = dict(payload)
    use_candidate_rebuild = _has_weekly_candidate_inputs(raw_json)

    pattern_result = _coerce_model(payload.get("pattern_result"), PatternAnalysisResult)
    if pattern_result is not None:
        sanitized["pattern_result"] = _sanitize_pattern_result(
            pattern_result,
            raw_json,
            indicator_json,
        )

    problem_result = _coerce_model(payload.get("problem_result"), ProblemAnalysisResult)
    if problem_result is not None:
        sanitized["problem_result"] = (
            _build_weekly_problem_result_from_candidates(raw_json)
            if use_candidate_rebuild
            else _sanitize_problem_result(
                problem_result,
                raw_json,
                indicator_json,
            )
        )

    cause_result = _coerce_model(payload.get("cause_result"), CauseAnalysisResult)
    if cause_result is not None:
        sanitized["cause_result"] = _sanitize_cause_result(
            cause_result,
            raw_json,
            indicator_json,
        )

    return sanitized


def _has_weekly_candidate_inputs(raw_json: Mapping[str, object]) -> bool:
    """주간 원본 JSON이 후보 기반 재구성에 필요한 핵심 필드를 갖는지 확인한다."""
    return (
        _lookup_path(raw_json, "weekly_summary") is not None
        and _lookup_path(raw_json, "category_summary") is not None
        and _lookup_path(raw_json, "waste_detection.high_spending.items") is not None
    )


def _load_json_object(value: object) -> dict[str, object]:
    """문자열 또는 dict 입력을 JSON 객체로 변환하고 터미널 제어 문자를 제거한다."""
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, str):
        return {}

    cleaned_value = _ANSI_ESCAPE_PATTERN.sub("", value)
    try:
        loaded = json.loads(cleaned_value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(key): item for key, item in loaded.items()}


def _load_json_value(value: str) -> object:
    """근거 문자열이 JSON이면 파싱하고 아니면 원문 문자열을 반환한다."""
    cleaned_value = _ANSI_ESCAPE_PATTERN.sub("", value)
    try:
        return json.loads(cleaned_value)
    except json.JSONDecodeError:
        return value


def _coerce_model[ModelT: BaseModel](value: object, model_type: type[ModelT]) -> ModelT | None:
    """dict 또는 Pydantic 모델 값을 지정한 해석 결과 모델로 좁힌다."""
    if isinstance(value, model_type):
        return value
    if isinstance(value, dict):
        try:
            return model_type.model_validate(value)
        except ValueError:
            return None
    return None


def _lookup_path(root: Mapping[str, object], json_path: str) -> object | None:
    """점 표기와 배열 인덱스가 섞인 JSON 경로에서 값을 조회한다."""
    current: object = root
    for segment in json_path.split("."):
        for key, index in _PATH_TOKEN_PATTERN.findall(segment):
            if key:
                if not isinstance(current, Mapping):
                    return None
                current = current.get(key)
                continue
            if index:
                if not isinstance(current, Sequence) or isinstance(current, str):
                    return None
                position = int(index)
                if position >= len(current):
                    return None
                current = current[position]
    return current


def _as_mapping(value: object) -> Mapping[str, object] | None:
    """객체가 문자열 키 매핑이면 매핑으로 반환한다."""
    if isinstance(value, Mapping):
        return value
    return None


def _as_number(value: object) -> float | None:
    """숫자 또는 숫자 문자열을 float 값으로 변환한다."""
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def _format_percent(value: object) -> str:
    """비율 값을 소수 둘째 자리까지 표시하되 불필요한 0은 줄인다."""
    percent = _as_number(value)
    if percent is None:
        return str(value)
    return f"{percent:.2f}".rstrip("0").rstrip(".")


def _resolve_evidence_value(
    evidence: EvidenceItem,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> object | None:
    """근거 경로를 원본 JSON과 지표 JSON에서 찾아보고 없으면 supporting_value를 파싱한다."""
    raw_value = _lookup_path(raw_json, evidence.json_path)
    if raw_value is not None:
        return raw_value
    indicator_value = _lookup_path(indicator_json, evidence.json_path)
    if indicator_value is not None:
        return indicator_value
    return _load_json_value(evidence.supporting_value)


def _finding_text(finding: SpendingFinding) -> str:
    """탐지 결과의 제목, 상세, 근거 설명을 하나의 검색용 문자열로 합친다."""
    evidence_text = " ".join(evidence.reason for evidence in finding.evidences)
    return f"{finding.subcategory} {finding.title} {finding.detail} {evidence_text}"


def _repeat_count_for_evidence(
    evidence: EvidenceItem,
    value: object | None,
    raw_json: Mapping[str, object],
) -> int | None:
    """반복 소비 근거 경로에서 반복 횟수를 추출한다."""
    if "repeat_patterns.top_merchants" in evidence.json_path:
        merchant_value = _as_mapping(value)
        if merchant_value is None:
            merchant_value = _as_mapping(_load_json_value(evidence.supporting_value))
        count = _as_number((merchant_value or {}).get("visit_count"))
        return int(count) if count is not None else None

    for key in ("delivery", "cafe", "convenience", "taxi"):
        if f"repeat_patterns.{key}" not in evidence.json_path:
            continue
        repeat_value = _as_mapping(_lookup_path(raw_json, f"repeat_patterns.{key}"))
        count = _as_number((repeat_value or {}).get("count"))
        return int(count) if count is not None else None

    if "repeat_patterns.consecutive_merchants" in evidence.json_path:
        consecutive_value = _lookup_path(raw_json, "repeat_patterns.consecutive_merchants")
        if isinstance(consecutive_value, Sequence) and not isinstance(consecutive_value, str):
            return 2 if len(consecutive_value) > 0 else 0
    return None


def _has_valid_repetition(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> bool:
    """탐지 결과가 실제 2회 이상 반복 근거를 갖는지 확인한다."""
    counts: list[int] = []
    for evidence in finding.evidences:
        value = _resolve_evidence_value(evidence, raw_json, indicator_json)
        count = _repeat_count_for_evidence(evidence, value, raw_json)
        if count is not None:
            counts.append(count)
    return bool(counts) and max(counts) >= 2


def _is_single_top_merchant_finding(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> bool:
    """탐지 결과의 모든 가맹점 근거가 1회 결제 top_merchants 항목인지 확인한다."""
    counts: list[int] = []
    for evidence in finding.evidences:
        if "repeat_patterns.top_merchants" not in evidence.json_path:
            continue
        value = _resolve_evidence_value(evidence, raw_json, indicator_json)
        count = _repeat_count_for_evidence(evidence, value, raw_json)
        if count is not None:
            counts.append(count)
    return bool(counts) and max(counts) < 2


def _has_impulse_support(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> bool:
    """충동 소비 탐지 결과가 야간·소액 누적 등 충동성 근거를 갖는지 확인한다."""
    if _is_single_top_merchant_finding(finding, raw_json, indicator_json):
        return False

    for evidence in finding.evidences:
        value = _resolve_evidence_value(evidence, raw_json, indicator_json)
        value_map = _as_mapping(value)
        if "late_night" in evidence.json_path:
            total_amount = _as_number((value_map or {}).get("total_amount"))
            if total_amount is None:
                total_amount = _as_number(value)
            return bool(total_amount and total_amount > 0)
        if "micro_spending" in evidence.json_path:
            count = _as_number((value_map or {}).get("count"))
            if count is None:
                count = _as_number(value)
            return bool(count and count >= 3)
    return "충동" not in _finding_text(finding)


def _weekday_label(value: object) -> str:
    """요일 약어 또는 원문 값을 사용자에게 보일 요일명으로 변환한다."""
    weekday = str(value)
    if weekday.endswith("요일"):
        return weekday
    return _WEEKDAY_LABELS.get(weekday, weekday)


def _find_weekday_breakdown(
    raw_json: Mapping[str, object],
    weekday: str,
) -> Mapping[str, object] | None:
    """요일별 breakdown 목록에서 지정한 요일 항목을 찾는다."""
    breakdown_value = _lookup_path(raw_json, "weekday_pattern.weekday_breakdown")
    if not isinstance(breakdown_value, Sequence) or isinstance(breakdown_value, str):
        return None

    for item in breakdown_value:
        item_map = _as_mapping(item)
        if item_map is None:
            continue
        if str(item_map.get("weekday") or "") == weekday:
            return item_map
    return None


def _sanitize_contextual_pattern(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
) -> SpendingFinding:
    """피크 요일 패턴에서 요일 집중도와 주중 비중을 혼동한 설명을 정정한다."""
    if not any(
        "weekday_pattern.peak_weekday" in evidence.json_path for evidence in finding.evidences
    ):
        return finding

    peak_weekday = _lookup_path(raw_json, "weekday_pattern.peak_weekday")
    if not isinstance(peak_weekday, str) or not peak_weekday:
        return finding

    breakdown = _find_weekday_breakdown(raw_json, peak_weekday)
    amount = _as_number((breakdown or {}).get("total_amount"))
    concentration = _lookup_path(
        raw_json,
        "weekly_metrics.special_metrics.weekday_concentration_ratio_percent",
    )
    weekday_ratio = _lookup_path(raw_json, "weekly_metrics.weekday_spending_ratio_percent")
    if amount is None or concentration is None or weekday_ratio is None:
        return finding

    weekday_name = _weekday_label(peak_weekday)
    detail = (
        f"{weekday_name} {_format_amount(amount)}원 결제로 분석 기간 최대 소비 요일입니다. "
        f"{weekday_name} 비중 {_format_percent(concentration)}%, "
        f"주중 소비 비중 {_format_percent(weekday_ratio)}%를 구분해 해석해야 합니다."
    )
    return finding.model_copy(update={"detail": detail})


def _sanitize_pattern_result(
    pattern_result: PatternAnalysisResult,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> PatternAnalysisResult:
    """반복·충동 패턴에서 원본 JSON 근거와 맞지 않는 항목을 제거한다."""
    return pattern_result.model_copy(
        update={
            "repeated_consumption": [
                finding
                for finding in pattern_result.repeated_consumption
                if _has_valid_repetition(finding, raw_json, indicator_json)
            ],
            "overspending_windows": [
                _sanitize_overspending_finding(finding, raw_json)
                for finding in pattern_result.overspending_windows
            ],
            "impulse_patterns": [
                finding
                for finding in pattern_result.impulse_patterns
                if _has_impulse_support(finding, raw_json, indicator_json)
            ],
            "contextual_patterns": [
                _sanitize_contextual_pattern(finding, raw_json)
                for finding in pattern_result.contextual_patterns
            ],
        }
    )


def _evidence_has_non_positive_diff(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> bool:
    """탐지 결과 근거가 감소 또는 비증가 카테고리를 가리키는지 확인한다."""
    for evidence in finding.evidences:
        value = _resolve_evidence_value(evidence, raw_json, indicator_json)
        if "diff_amount" in evidence.json_path:
            diff_value = _as_number(value)
            if diff_value is not None and diff_value <= 0:
                return True

        value_map = _as_mapping(value)
        if value_map is None:
            continue
        direction = value_map.get("direction")
        diff_amount = _as_number(value_map.get("diff_amount"))
        if direction == "decrease" or (diff_amount is not None and diff_amount <= 0):
            return True
    return False


def _is_decrease_problem(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> bool:
    """감소한 카테고리를 문제 소비로 잘못 분류했는지 확인한다."""
    text = _finding_text(finding)
    return any(word in text for word in _DECREASE_WORDS) and _evidence_has_non_positive_diff(
        finding,
        raw_json,
        indicator_json,
    )


def _has_trend_evidence(finding: SpendingFinding) -> bool:
    """장기·지속 판단에 필요한 추이 근거 경로가 있는지 확인한다."""
    return any(
        hint in evidence.json_path for evidence in finding.evidences for hint in _TREND_PATH_HINTS
    )


def _is_unsupported_long_term_problem(finding: SpendingFinding) -> bool:
    """추이 근거 없이 지속 증가나 장기 문제로 단정했는지 확인한다."""
    text = _finding_text(finding)
    return any(word in text for word in _LONG_TERM_WORDS) and not _has_trend_evidence(finding)


def _iter_high_spending_items(
    raw_json: Mapping[str, object],
) -> list[tuple[Mapping[str, object], int, float]]:
    """원본 JSON의 고액 결제 항목을 금액과 인덱스가 포함된 목록으로 반환한다."""
    items_value = _lookup_path(raw_json, "waste_detection.high_spending.items")
    if not isinstance(items_value, Sequence) or isinstance(items_value, str):
        return []

    items: list[tuple[Mapping[str, object], int, float]] = []
    for index, item in enumerate(items_value):
        item_map = _as_mapping(item)
        if item_map is None:
            continue
        amount = _as_number(item_map.get("amount")) or 0.0
        items.append((item_map, index, amount))
    return items


def _high_spending_item_for_category(
    raw_json: Mapping[str, object],
    category: str,
) -> tuple[Mapping[str, object], int] | None:
    """카테고리와 연결되는 고액 결제 항목 중 금액이 가장 큰 항목을 찾는다."""
    matched_items: list[tuple[Mapping[str, object], int, float]] = []
    for item_map, index, amount in _iter_high_spending_items(raw_json):
        if str(item_map.get("category") or "") != category:
            continue
        matched_items.append((item_map, index, amount))
    if not matched_items:
        return None

    item, index, _amount = max(matched_items, key=lambda item_result: item_result[2])
    return item, index


def _high_spending_item_for_finding(
    raw_json: Mapping[str, object],
    finding: SpendingFinding,
) -> tuple[Mapping[str, object], int] | None:
    """탐지 결과의 가맹점명 또는 카테고리로 연결되는 고액 결제 항목을 찾는다."""
    text = _finding_text(finding)
    for item, index, _amount in _iter_high_spending_items(raw_json):
        merchant = str(item.get("merchant") or "")
        if merchant and merchant in text:
            return item, index

    return _high_spending_item_for_category(raw_json, finding.subcategory)


def _high_spending_evidence(item: Mapping[str, object], index: int) -> EvidenceItem:
    """고액 결제 항목을 직접 가리키는 evidence 객체를 만든다."""
    merchant = str(item.get("merchant") or "고액 결제")
    amount = _format_amount(item.get("amount"))
    return EvidenceItem(
        json_path=f"waste_detection.high_spending.items[{index}].amount",
        supporting_value=amount,
        reason=f"{merchant} 고액 결제 금액",
    )


def _high_spending_item_for_evidence(
    raw_json: Mapping[str, object],
    evidence: EvidenceItem,
) -> tuple[Mapping[str, object], int] | None:
    """근거 경로가 가리키는 고액 결제 항목을 원본 JSON에서 찾는다."""
    index_match = _HIGH_SPENDING_ITEM_INDEX_PATTERN.search(evidence.json_path)
    if index_match is None:
        return None

    index = int(index_match.group(1))
    value = _lookup_path(raw_json, f"waste_detection.high_spending.items[{index}]")
    item = _as_mapping(value)
    if item is None:
        return None
    return item, index


def _high_spending_items_for_finding(
    raw_json: Mapping[str, object],
    finding: SpendingFinding,
) -> list[tuple[Mapping[str, object], int]]:
    """탐지 결과의 근거·문구와 연결되는 고액 결제 항목 목록을 찾는다."""
    item_results: list[tuple[Mapping[str, object], int]] = []
    seen_indexes: set[int] = set()
    for evidence in finding.evidences:
        item_result = _high_spending_item_for_evidence(raw_json, evidence)
        if item_result is None:
            continue
        item, index = item_result
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        item_results.append((item, index))

    if item_results:
        return item_results

    fallback_item = _high_spending_item_for_finding(raw_json, finding)
    if fallback_item is None:
        return []
    item, index = fallback_item
    return [(item, index)]


def _normalize_high_spending_evidence_paths(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
) -> SpendingFinding:
    """고액 결제 근거 경로를 배열 항목이 아닌 amount 필드 경로로 좁힌다."""
    normalized_evidences: list[EvidenceItem] = []
    for evidence in finding.evidences:
        item_result = _high_spending_item_for_evidence(raw_json, evidence)
        if item_result is None:
            normalized_evidences.append(evidence)
            continue
        item, index = item_result
        normalized_evidences.append(_high_spending_evidence(item, index))
    return finding.model_copy(update={"evidences": normalized_evidences})


def _sanitize_overspending_finding(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
) -> SpendingFinding:
    """고액 소비 탐지 결과의 근거를 실제 high_spending item 필드로 보정한다."""
    item_result = _high_spending_item_for_finding(raw_json, finding)
    if item_result is None:
        return finding

    item, index = item_result
    return finding.model_copy(update={"evidences": [_high_spending_evidence(item, index)]})


def _category_entry_for_name(
    raw_json: Mapping[str, object],
    category: str,
) -> tuple[Mapping[str, object], int] | None:
    """category_summary에서 지정한 카테고리 항목과 인덱스를 찾는다."""
    categories_value = _lookup_path(raw_json, "category_summary")
    if not isinstance(categories_value, Sequence) or isinstance(categories_value, str):
        return None

    for index, item in enumerate(categories_value):
        item_map = _as_mapping(item)
        if item_map is None:
            continue
        if str(item_map.get("category") or "") == category:
            return item_map, index
    return None


def _category_diff_evidence(
    category_entry: Mapping[str, object],
    index: int,
) -> EvidenceItem | None:
    """카테고리 증감 문제에 직접 연결되는 diff_amount evidence를 만든다."""
    diff_amount = _as_number(category_entry.get("diff_amount"))
    if diff_amount is None:
        return None

    category = str(category_entry.get("category") or "카테고리")
    return EvidenceItem(
        json_path=f"category_summary[{index}].diff_amount",
        supporting_value=_format_amount(diff_amount),
        reason=f"{category} 카테고리 전주 대비 증감액",
    )


def _sanitize_problem_finding(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
) -> SpendingFinding:
    """카테고리 문제 결과의 근거를 전체 지표가 아닌 카테고리 판단 필드로 보정한다."""
    if finding.subcategory not in _CATEGORY_PREFIX:
        return finding

    if not any(
        evidence.json_path.startswith(("weekly_summary", "weekly_comparisons"))
        for evidence in finding.evidences
    ):
        return finding

    category_result = _category_entry_for_name(raw_json, finding.subcategory)
    if category_result is None:
        return finding

    category_entry, index = category_result
    evidence = _category_diff_evidence(category_entry, index)
    if evidence is None:
        return finding
    return finding.model_copy(update={"evidences": [evidence]})


def _finding_category_entries(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """탐지 결과 근거에서 category_summary 항목들을 추출한다."""
    entries: list[Mapping[str, object]] = []
    for evidence in finding.evidences:
        value = _resolve_evidence_value(evidence, raw_json, indicator_json)
        value_map = _as_mapping(value)
        if value_map is not None and value_map.get("category") is not None:
            entries.append(value_map)
    return entries


def _is_low_signal_new_category_problem(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> bool:
    """전주 0원에서 새로 생긴 소액성 변동비를 문제 소비로 단정했는지 확인한다."""
    for entry in _finding_category_entries(finding, raw_json, indicator_json):
        category = str(entry.get("category") or "")
        if category in {"교육", "납부"}:
            continue

        previous_amount = _as_number(entry.get("prev_week_amount"))
        diff_rate = _as_number(entry.get("diff_rate_percent"))
        ratio = _as_number(entry.get("ratio_percent"))
        if (
            previous_amount == 0
            and diff_rate == 0
            and (ratio is None or ratio < 10)
            and _high_spending_item_for_category(raw_json, category) is None
        ):
            return True
    return False


def _sanitize_problem_findings(
    findings: list[SpendingFinding],
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> list[SpendingFinding]:
    """문제 소비 목록에서 감소 카테고리와 추이 없는 장기 단정을 제거한다."""
    sanitized_findings: list[SpendingFinding] = []
    for finding in findings:
        sanitized_finding = _sanitize_problem_finding(finding, raw_json)
        if _is_decrease_problem(sanitized_finding, raw_json, indicator_json):
            continue
        if _is_unsupported_long_term_problem(sanitized_finding):
            continue
        if _is_low_signal_new_category_problem(sanitized_finding, raw_json, indicator_json):
            continue
        sanitized_findings.append(sanitized_finding)
    return sanitized_findings


def _sanitize_problem_result(
    problem_result: ProblemAnalysisResult,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> ProblemAnalysisResult:
    """문제 소비 결과 전체에 공통 정제 규칙을 적용한다."""
    return problem_result.model_copy(
        update={
            "money_leaks": _sanitize_problem_findings(
                problem_result.money_leaks,
                raw_json,
                indicator_json,
            ),
            "saving_blockers": _sanitize_problem_findings(
                problem_result.saving_blockers,
                raw_json,
                indicator_json,
            ),
            "fixed_cost_issues": _sanitize_problem_findings(
                problem_result.fixed_cost_issues,
                raw_json,
                indicator_json,
            ),
            "variable_cost_issues": _sanitize_problem_findings(
                problem_result.variable_cost_issues,
                raw_json,
                indicator_json,
            ),
            "short_term_problem_spending": _sanitize_problem_findings(
                problem_result.short_term_problem_spending,
                raw_json,
                indicator_json,
            ),
            "long_term_problem_spending": _sanitize_problem_findings(
                problem_result.long_term_problem_spending,
                raw_json,
                indicator_json,
            ),
        }
    )


def _build_weekly_problem_result_from_candidates(
    raw_json: Mapping[str, object],
) -> ProblemAnalysisResult:
    """주간 원본 JSON의 허용 후보만 사용해 문제 소비 결과를 재구성한다."""
    return ProblemAnalysisResult(
        money_leaks=_build_money_leak_candidates(raw_json),
        saving_blockers=_build_saving_blocker_candidates(raw_json),
        fixed_cost_issues=_build_fixed_cost_issue_candidates(raw_json),
        variable_cost_issues=[],
        short_term_problem_spending=[],
        long_term_problem_spending=[],
    )


def _build_money_leak_candidates(raw_json: Mapping[str, object]) -> list[SpendingFinding]:
    """비고정비 고액 결제 중 가장 큰 항목을 새는 돈 후보로 만든다."""
    variable_items = [
        item_result
        for item_result in _iter_high_spending_items(raw_json)
        if not _is_fixed_cost_item(item_result[0])
    ]
    if not variable_items:
        return []

    item, index, _amount = max(variable_items, key=lambda item_result: item_result[2])
    merchant = str(item.get("merchant") or "고액 결제")
    category = str(item.get("category") or "기타")
    amount = _format_amount(item.get("amount"))
    return [
        SpendingFinding(
            subcategory=category,
            title=f"{merchant} 고액 결제",
            detail=f"{merchant}에서 {amount}원 단일 고액 결제가 확인되었습니다.",
            confidence="high",
            evidences=[_high_spending_evidence(item, index)],
        )
    ]


def _positive_category_candidates(
    raw_json: Mapping[str, object],
) -> list[tuple[Mapping[str, object], int, float]]:
    """전주 대비 증가한 카테고리 항목을 증감액 기준으로 반환한다."""
    category_value = _lookup_path(raw_json, "category_summary")
    if not isinstance(category_value, Sequence) or isinstance(category_value, str):
        return []

    candidates: list[tuple[Mapping[str, object], int, float]] = []
    for index, item in enumerate(category_value):
        item_map = _as_mapping(item)
        if item_map is None:
            continue
        diff_amount = _as_number(item_map.get("diff_amount"))
        if diff_amount is None or diff_amount <= 0:
            continue
        candidates.append((item_map, index, diff_amount))
    return candidates


def _build_saving_blocker_candidates(raw_json: Mapping[str, object]) -> list[SpendingFinding]:
    """증가 폭이 가장 큰 비고정 카테고리를 절약 방해 후보로 만든다."""
    positive_categories = [
        item_result
        for item_result in _positive_category_candidates(raw_json)
        if str(item_result[0].get("category") or "") != "납부"
    ]
    if not positive_categories:
        return []

    category_entry, index, _diff_amount = max(
        positive_categories,
        key=lambda item_result: item_result[2],
    )
    category = str(category_entry.get("category") or "카테고리")
    total_amount = _format_amount(category_entry.get("total_amount"))
    diff_amount = _format_amount(category_entry.get("diff_amount"))
    evidence = _category_diff_evidence(category_entry, index)
    if evidence is None:
        return []

    return [
        SpendingFinding(
            subcategory=category,
            title=f"{category} 지출 증가",
            detail=f"{category} 지출이 {total_amount}원으로 전주 대비 {diff_amount}원 증가했습니다.",
            confidence="high",
            evidences=[evidence],
        )
    ]


def _build_fixed_cost_issue_candidates(raw_json: Mapping[str, object]) -> list[SpendingFinding]:
    """고액 납부 항목을 고정비 점검 후보로 묶어 만든다."""
    fixed_items = [
        item_result
        for item_result in _iter_high_spending_items(raw_json)
        if _is_fixed_cost_item(item_result[0])
    ]
    if not fixed_items:
        return []

    sorted_items = sorted(fixed_items, key=lambda item_result: item_result[2], reverse=True)[:2]
    evidences = [_high_spending_evidence(item, index) for item, index, _amount in sorted_items]
    item_descriptions = [
        f"{str(item.get('merchant') or '고정비')} {_format_amount(item.get('amount'))}원"
        for item, _index, _amount in sorted_items
    ]
    return [
        SpendingFinding(
            subcategory="납부",
            title="고정비 점검 대상",
            detail=f"{', '.join(item_descriptions)} 고액 납부 결제가 확인되었습니다.",
            confidence="high",
            evidences=evidences,
        )
    ]


def _build_weekly_cause_result_from_candidates(
    raw_json: Mapping[str, object],
) -> CauseAnalysisResult:
    """주간 원본 JSON의 객관 후보만 사용해 원인 결과와 개입 타겟을 재구성한다."""
    return CauseAnalysisResult(
        habitual_causes=[],
        reward_causes=[],
        stress_causes=[],
        convenience_causes=[],
        small_accumulation_causes=[],
        intervention_targets=_build_intervention_targets_from_candidates(raw_json),
    )


def _build_intervention_targets_from_candidates(
    raw_json: Mapping[str, object],
) -> list[InterventionTarget]:
    """원본 고액 결제와 배달 결제 후보만 사용해 RAG 개입 타겟을 만든다."""
    targets = [
        *_auto_high_spending_targets(raw_json),
        *_auto_delivery_targets(raw_json),
    ]
    return _dedupe_intervention_targets(targets)


def _has_reward_support(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> bool:
    """보상성 원인으로 볼 만한 여가·사교성 근거가 있는지 확인한다."""
    for evidence in finding.evidences:
        value = _resolve_evidence_value(evidence, raw_json, indicator_json)
        value_map = _as_mapping(value)
        category = str((value_map or {}).get("category") or (value_map or {}).get("main_category"))
        if category in {"여가", "사교"} and not _is_single_top_merchant_finding(
            finding,
            raw_json,
            indicator_json,
        ):
            return True
    return False


def _is_fixed_cost_finding(finding: SpendingFinding) -> bool:
    """납부·관리비 등 고정비 성격의 원인 항목인지 확인한다."""
    text = _finding_text(finding)
    return any(word in text for word in _FIXED_COST_WORDS)


def _is_fixed_cost_item(item: Mapping[str, object]) -> bool:
    """고액 결제 항목이 납부·관리비 등 고정비 성격인지 확인한다."""
    category = str(item.get("category") or "")
    merchant = str(item.get("merchant") or "")
    return category == "납부" or any(word in merchant for word in _FIXED_COST_WORDS)


def _dedupe_high_spending_items(
    item_results: list[tuple[Mapping[str, object], int]],
) -> list[tuple[Mapping[str, object], int]]:
    """고액 결제 항목 목록에서 같은 인덱스의 중복을 제거한다."""
    deduped: list[tuple[Mapping[str, object], int]] = []
    seen_indexes: set[int] = set()
    for item, index in item_results:
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        deduped.append((item, index))
    return deduped


def _date_part_from_used_at(value: object) -> str | None:
    """결제 시각 문자열에서 YYYY-MM-DD 날짜 부분을 추출한다."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    return value[:10]


def _time_part_from_used_at(value: object) -> str | None:
    """결제 시각 문자열에서 HH:MM 시각 부분을 추출한다."""
    if not isinstance(value, str) or len(value) < 16:
        return None
    return value[11:16]


def _weekday_label_for_date(
    raw_json: Mapping[str, object],
    date_text: str | None,
) -> str | None:
    """날짜 문자열을 요일명으로 바꾸되 주간 피크 요일 값이 있으면 우선 사용한다."""
    if date_text is None:
        return None

    max_day_date = str(_lookup_path(raw_json, "weekly_summary.max_day_date") or "")
    peak_weekday = _lookup_path(raw_json, "weekday_pattern.peak_weekday")
    if date_text == max_day_date and peak_weekday is not None:
        return _weekday_label(peak_weekday)

    try:
        parsed_date = dt.date.fromisoformat(date_text)
    except ValueError:
        return None
    return _WEEKDAY_LABELS_BY_INDEX[parsed_date.weekday()]


def _evidence_for_numeric_path(
    raw_json: Mapping[str, object],
    json_path: str,
    reason: str,
    *,
    percent: bool = False,
) -> EvidenceItem | None:
    """원본 JSON의 숫자 경로를 evidence 객체로 변환한다."""
    value = _lookup_path(raw_json, json_path)
    if value is None:
        return None
    supporting_value = _format_percent(value) if percent else _format_amount(value)
    return EvidenceItem(
        json_path=json_path,
        supporting_value=supporting_value,
        reason=reason,
    )


def _dedupe_evidences(evidences: list[EvidenceItem]) -> list[EvidenceItem]:
    """근거 목록에서 같은 JSON 경로의 중복을 제거한다."""
    deduped: list[EvidenceItem] = []
    seen_paths: set[str] = set()
    for evidence in evidences:
        if evidence.json_path in seen_paths:
            continue
        seen_paths.add(evidence.json_path)
        deduped.append(evidence)
    return deduped


def _sanitize_one_off_high_spending_causes(
    findings: list[SpendingFinding],
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> list[SpendingFinding]:
    """단일 고액 원인에서 고정비 항목을 제거하고 근거 경로를 amount 필드로 좁힌다."""
    sanitized_findings: list[SpendingFinding] = []
    seen_paths: set[str] = set()
    for finding in findings:
        if not _has_supported_cause_direction(finding, raw_json, indicator_json):
            continue

        item_results = _high_spending_items_for_finding(raw_json, finding)
        variable_items = [
            (item, index) for item, index in item_results if not _is_fixed_cost_item(item)
        ]
        if item_results and not variable_items:
            continue
        if not item_results and _is_fixed_cost_finding(finding):
            continue

        normalized = _normalize_high_spending_evidence_paths(finding, raw_json)
        if variable_items:
            normalized = normalized.model_copy(
                update={
                    "evidences": [
                        _high_spending_evidence(item, index) for item, index in variable_items
                    ]
                }
            )

        dedupe_key = "|".join(evidence.json_path for evidence in normalized.evidences)
        if dedupe_key in seen_paths:
            continue
        seen_paths.add(dedupe_key)
        sanitized_findings.append(normalized)
    return sanitized_findings


def _fixed_cost_item_results_from_findings(
    findings: list[SpendingFinding],
    raw_json: Mapping[str, object],
) -> list[tuple[Mapping[str, object], int]]:
    """원인 항목들에서 고정비 고액 결제 항목을 모은다."""
    item_results: list[tuple[Mapping[str, object], int]] = []
    for finding in findings:
        for item, index in _high_spending_items_for_finding(raw_json, finding):
            if _is_fixed_cost_item(item):
                item_results.append((item, index))
    return _dedupe_high_spending_items(item_results)


def _build_fixed_cost_timing_cause(
    raw_json: Mapping[str, object],
    item_results: list[tuple[Mapping[str, object], int]],
) -> SpendingFinding:
    """고정비 고액 결제 항목들을 하나의 납부 타이밍 원인으로 묶는다."""
    item_descriptions = [
        f"{str(item.get('merchant') or '고정비')} {_format_amount(item.get('amount'))}원"
        for item, _index in item_results
    ]
    date_values = {
        date_text
        for item, _index in item_results
        if (date_text := _date_part_from_used_at(item.get("used_at"))) is not None
    }
    shared_date = next(iter(date_values)) if len(date_values) == 1 else None
    weekday = _weekday_label_for_date(raw_json, shared_date)
    if weekday is not None and len(item_results) > 1:
        timing_phrase = f"{weekday}에 함께 반영되어"
    elif weekday is not None:
        timing_phrase = f"{weekday}에 반영되어"
    else:
        timing_phrase = "같은 주에 반영되어"

    weekend_ratio = _lookup_path(raw_json, "weekly_metrics.weekend_spending_ratio_percent")
    ratio_phrase = ""
    if weekend_ratio is not None:
        ratio_phrase = (
            f" 주말 소비 비중 {_format_percent(weekend_ratio)}% 해석에도 함께 반영됩니다."
        )

    return SpendingFinding(
        subcategory="납부",
        title="고정비 납부 타이밍 집중",
        detail=(
            f"{', '.join(item_descriptions)} 납부가 {timing_phrase} 이번 주 총액을 키운 "
            f"고정비 납부 타이밍 후보입니다.{ratio_phrase}"
        ),
        confidence="high",
        evidences=[_high_spending_evidence(item, index) for item, index in item_results],
    )


def _sanitize_fixed_cost_timing_causes(
    findings: list[SpendingFinding],
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> list[SpendingFinding]:
    """고정비 납부 원인을 중복 없이 묶고 일반론적 설명을 원본 맥락으로 바꾼다."""
    supported_findings = [
        finding
        for finding in findings
        if _has_supported_cause_direction(finding, raw_json, indicator_json)
    ]
    if not supported_findings:
        return []

    fixed_item_results = _fixed_cost_item_results_from_findings(supported_findings, raw_json)
    if fixed_item_results:
        return [_build_fixed_cost_timing_cause(raw_json, fixed_item_results)]

    return [
        _normalize_high_spending_evidence_paths(finding, raw_json)
        for finding in supported_findings
        if _is_fixed_cost_finding(finding)
    ]


def _largest_high_spending_item_on_date(
    raw_json: Mapping[str, object],
    date_text: str | None,
) -> tuple[Mapping[str, object], int] | None:
    """지정한 날짜의 비고정비 고액 결제 중 가장 큰 항목을 찾는다."""
    matched_items: list[tuple[Mapping[str, object], int, float]] = []
    for item, index, amount in _iter_high_spending_items(raw_json):
        if _is_fixed_cost_item(item):
            continue
        item_date = _date_part_from_used_at(item.get("used_at"))
        if date_text is not None and item_date != date_text:
            continue
        matched_items.append((item, index, amount))

    if not matched_items:
        return None
    item, index, _amount = max(matched_items, key=lambda item_result: item_result[2])
    return item, index


def _build_period_concentration_cause(
    raw_json: Mapping[str, object],
    fallback_finding: SpendingFinding,
) -> SpendingFinding | None:
    """요일·기간 집중 원인을 피크 요일 금액과 핵심 고액 결제 맥락으로 재작성한다."""
    peak_weekday_value = _lookup_path(raw_json, "weekday_pattern.peak_weekday")
    if peak_weekday_value is None:
        return None

    peak_weekday = str(peak_weekday_value)
    peak_weekday_label = _weekday_label(peak_weekday)
    peak_breakdown = _find_weekday_breakdown(raw_json, peak_weekday)
    peak_amount = _lookup_path(raw_json, "weekly_summary.max_day_amount")
    if peak_breakdown is not None:
        peak_amount = peak_breakdown.get("total_amount", peak_amount)
    if peak_amount is None:
        return None

    concentration = _lookup_path(
        raw_json,
        "weekly_metrics.special_metrics.weekday_concentration_ratio_percent",
    )
    weekend_ratio = _lookup_path(raw_json, "weekly_metrics.weekend_spending_ratio_percent")
    max_day_date = str(_lookup_path(raw_json, "weekly_summary.max_day_date") or "")
    high_item_result = _largest_high_spending_item_on_date(raw_json, max_day_date or None)

    metric_phrases: list[str] = []
    if concentration is not None:
        metric_phrases.append(f"요일 편중도 {_format_percent(concentration)}%")
    if weekend_ratio is not None:
        metric_phrases.append(f"주말 소비 비중 {_format_percent(weekend_ratio)}%")
    metric_text = ", ".join(metric_phrases)

    detail = f"{peak_weekday_label} {_format_amount(peak_amount)}원 결제가 발생해"
    if metric_text:
        detail = f"{detail} {metric_text}로 소비가 특정 기간에 집중된 후보입니다."
    else:
        detail = f"{detail} 소비가 특정 기간에 집중된 후보입니다."

    evidences: list[EvidenceItem] = []
    max_day_evidence = _evidence_for_numeric_path(
        raw_json,
        "weekly_summary.max_day_amount",
        "최대 소비일 금액",
    )
    if max_day_evidence is not None:
        evidences.append(max_day_evidence)
    concentration_evidence = _evidence_for_numeric_path(
        raw_json,
        "weekly_metrics.special_metrics.weekday_concentration_ratio_percent",
        "최대 소비 요일 집중도",
        percent=True,
    )
    if concentration_evidence is not None:
        evidences.append(concentration_evidence)
    weekend_evidence = _evidence_for_numeric_path(
        raw_json,
        "weekly_metrics.weekend_spending_ratio_percent",
        "주말 소비 비중",
        percent=True,
    )
    if weekend_evidence is not None:
        evidences.append(weekend_evidence)

    title = f"{peak_weekday_label} 소비 집중"
    if high_item_result is not None:
        item, index = high_item_result
        merchant = str(item.get("merchant") or "고액 결제")
        amount = _format_amount(item.get("amount"))
        used_time = _time_part_from_used_at(item.get("used_at"))
        time_phrase = f" {used_time}" if used_time is not None else ""
        detail = (
            f"{detail} 특히 {merchant} {amount}원 결제가 "
            f"{peak_weekday_label}{time_phrase}에 발생해 집중도를 크게 설명합니다."
        )
        evidences.append(_high_spending_evidence(item, index))
        title = f"{peak_weekday_label} 단일 고액 결제 중심의 주말 편중"

    return fallback_finding.model_copy(
        update={
            "title": title,
            "detail": detail,
            "evidences": _dedupe_evidences(evidences or fallback_finding.evidences),
        }
    )


def _sanitize_period_concentration_causes(
    findings: list[SpendingFinding],
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> list[SpendingFinding]:
    """기간 집중 원인을 실제 피크 요일·비중·고액 결제 맥락으로 정규화한다."""
    supported_findings = [
        finding
        for finding in findings
        if _has_supported_cause_direction(finding, raw_json, indicator_json)
    ]
    if not supported_findings:
        return []

    contextual_cause = _build_period_concentration_cause(raw_json, supported_findings[0])
    if contextual_cause is not None:
        return [contextual_cause]
    return [
        _normalize_high_spending_evidence_paths(finding, raw_json) for finding in supported_findings
    ]


def _sanitize_cause_result(
    cause_result: CauseAnalysisResult,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> CauseAnalysisResult:
    """원인 결과에서 근거 없는 습관·보상·스트레스 단정과 타겟 오류를 정리한다."""
    return cause_result.model_copy(
        update={
            "habitual_causes": [
                finding
                for finding in cause_result.habitual_causes
                if _has_valid_repetition(finding, raw_json, indicator_json)
                and _has_supported_cause_direction(finding, raw_json, indicator_json)
            ],
            "reward_causes": [
                finding
                for finding in cause_result.reward_causes
                if _has_reward_support(finding, raw_json, indicator_json)
                and _has_supported_cause_direction(finding, raw_json, indicator_json)
            ],
            "stress_causes": [
                finding
                for finding in cause_result.stress_causes
                if not _is_fixed_cost_finding(finding)
                and _has_supported_cause_direction(finding, raw_json, indicator_json)
            ],
            "convenience_causes": [
                finding
                for finding in cause_result.convenience_causes
                if _has_supported_cause_direction(finding, raw_json, indicator_json)
            ],
            "small_accumulation_causes": [
                finding
                for finding in cause_result.small_accumulation_causes
                if _has_supported_cause_direction(finding, raw_json, indicator_json)
            ],
            "one_off_high_spending_causes": _sanitize_one_off_high_spending_causes(
                cause_result.one_off_high_spending_causes,
                raw_json,
                indicator_json,
            ),
            "fixed_cost_timing_causes": _sanitize_fixed_cost_timing_causes(
                cause_result.fixed_cost_timing_causes,
                raw_json,
                indicator_json,
            ),
            "period_concentration_causes": _sanitize_period_concentration_causes(
                cause_result.period_concentration_causes,
                raw_json,
                indicator_json,
            ),
            "intervention_targets": _sanitize_intervention_targets(
                cause_result.intervention_targets,
                raw_json,
            ),
        }
    )


def _has_supported_cause_direction(
    finding: SpendingFinding,
    raw_json: Mapping[str, object],
    indicator_json: Mapping[str, object],
) -> bool:
    """원인 해석 항목이 감소 또는 추이 없는 장기 단정으로 반증되지 않는지 확인한다."""
    if _is_decrease_problem(finding, raw_json, indicator_json):
        return False
    return not _is_unsupported_long_term_problem(finding)


def _find_high_spending_item(
    raw_json: Mapping[str, object],
    target: InterventionTarget,
) -> tuple[Mapping[str, object], int] | None:
    """개입 타겟과 연결되는 고액 결제 항목과 인덱스를 찾는다."""
    items = _iter_high_spending_items(raw_json)
    if not items:
        return None

    index_match = _HIGH_SPENDING_ITEM_INDEX_PATTERN.search(target.target_json_path)
    if index_match is not None:
        target_index = int(index_match.group(1))
        for item_map, index, _amount in items:
            if index == target_index:
                return item_map, index

    direct_value = _lookup_path(raw_json, target.target_json_path)
    direct_map = _as_mapping(direct_value)
    direct_category = str((direct_map or {}).get("category") or "")
    if direct_category:
        category_item = _high_spending_item_for_category(raw_json, direct_category)
        if category_item is not None:
            return category_item

    for item_map, index, _amount in items:
        if direct_map is item_map or direct_map == item_map:
            return item_map, index
        merchant = str(item_map.get("merchant") or "")
        if merchant and merchant in target.title:
            return item_map, index
    return None


def _category_entry_for_target(
    raw_json: Mapping[str, object],
    target: InterventionTarget,
) -> Mapping[str, object] | None:
    """개입 타겟 경로가 category_summary 항목이면 해당 카테고리 항목을 반환한다."""
    match = _CATEGORY_SUMMARY_INDEX_PATTERN.search(target.target_json_path)
    if match is None:
        return None

    value = _lookup_path(raw_json, f"category_summary[{match.group(1)}]")
    return _as_mapping(value)


def _format_amount(value: object) -> str:
    """금액 값을 정수 문자열로 표현한다."""
    amount = _as_number(value)
    if amount is None:
        return str(value)
    return str(int(amount))


def _high_spending_query_hint(category: str, merchant: str) -> str:
    """고액 결제 항목의 RAG 검색 질의 힌트를 생성한다."""
    if category == "쇼핑" and "하이마트" in merchant:
        return "가전 쇼핑 고액 결제 예산 점검"
    return f"{category} 고액 결제 예산 점검"


def _normalize_high_spending_item(
    item: Mapping[str, object],
    index: int,
) -> InterventionTarget:
    """고액 결제 항목 하나를 RAG 검색용 개입 타겟으로 정규화한다."""
    merchant = str(item.get("merchant") or "고액 결제")
    category = str(item.get("category") or "")
    amount = _format_amount(item.get("amount"))

    if _is_fixed_cost_item(item):
        return InterventionTarget(
            target_type="fixed_cost_review",
            title=f"{merchant} 고정비 점검 타겟",
            linked_cause=f"{merchant} {amount}원 납부 지출",
            target_json_path=f"waste_detection.high_spending.items[{index}].amount",
            reason=f"고정비 성격의 {merchant} 결제가 주간 지출에 크게 반영되어 RAG 검색 후보로 넘김",
            query_hint=f"{merchant} 고정비 점검",
        )

    prefix = _CATEGORY_PREFIX.get(category, "category")
    return InterventionTarget(
        target_type=f"{prefix}_high_spending_review",
        title=f"{merchant} {category} 고액 단일 결제 점검 타겟",
        linked_cause=f"{merchant} {amount}원 단일 고액 결제",
        target_json_path=f"waste_detection.high_spending.items[{index}].amount",
        reason=f"{category} 지출 급증의 주요 원인이 {merchant} 단일 고액 결제인지 확인하기 위해 RAG 검색 후보로 넘김",
        query_hint=_high_spending_query_hint(category, merchant),
    )


def _normalize_high_spending_target(
    raw_json: Mapping[str, object],
    target: InterventionTarget,
) -> InterventionTarget | None:
    """고액 결제 개입 타겟을 구체적인 검색 후보 형태로 정규화한다."""
    item_result = _find_high_spending_item(raw_json, target)
    if item_result is None:
        return None

    item, index = item_result
    return _normalize_high_spending_item(item, index)


def _normalize_delivery_target(
    raw_json: Mapping[str, object],
    target: InterventionTarget,
) -> InterventionTarget | None:
    """배달 관련 개입 타겟을 반복 단정 없이 단일 결제 점검 후보로 정규화한다."""
    if "배달" not in target.title and "repeat_patterns.delivery" not in target.target_json_path:
        return None

    delivery = _as_mapping(_lookup_path(raw_json, "repeat_patterns.delivery"))
    if delivery is None:
        return None
    count = int(_as_number(delivery.get("count")) or 0)
    amount = _format_amount(delivery.get("total_amount"))
    if count <= 0 or amount == "0":
        return None

    target_type = "delivery_frequency_review" if count >= 2 else "delivery_single_order_review"
    title = "배달 반복 결제 점검 타겟" if count >= 2 else "배달 단일 결제 점검 타겟"
    return InterventionTarget(
        target_type=target_type,
        title=title,
        linked_cause=f"배달 {count}회 결제 {amount}원이 확인됨",
        target_json_path="repeat_patterns.delivery.total_amount",
        reason="배달 결제 금액이 확인되어 RAG 검색 후보로 넘김",
        query_hint="배달 결제 절약 점검",
    )


def _is_period_concentration_target(target: InterventionTarget) -> bool:
    """개입 타겟이 요일·기간 집중 원인과 연결되는지 확인한다."""
    target_type = target.target_type.lower()
    text = f"{target.title} {target.linked_cause} {target.target_json_path}"
    return (
        target.linked_cause == "period_concentration_causes"
        or target_type
        in {"spending_pattern", "period_concentration", "period_concentration_review"}
        or "weekday_pattern.peak_weekday" in target.target_json_path
        or "요일" in text
        and "집중" in text
    )


def _normalize_period_concentration_target(
    raw_json: Mapping[str, object],
    target: InterventionTarget,
) -> InterventionTarget | None:
    """요일·기간 집중 개입 타겟을 RAG 검색에 쓸 판단 필드 중심 후보로 정규화한다."""
    if not _is_period_concentration_target(target):
        return None

    concentration_path = "weekly_metrics.special_metrics.weekday_concentration_ratio_percent"
    concentration = _lookup_path(raw_json, concentration_path)
    peak_weekday = _lookup_path(raw_json, "weekday_pattern.peak_weekday")
    if concentration is None or peak_weekday is None:
        return None

    peak_weekday_label = _weekday_label(peak_weekday)
    max_day_date = str(_lookup_path(raw_json, "weekly_summary.max_day_date") or "")
    high_item_result = _largest_high_spending_item_on_date(raw_json, max_day_date or None)

    linked_cause = f"{peak_weekday_label} 소비 집중"
    title = f"{peak_weekday_label} 소비 집중 점검 타겟"
    reason = (
        f"{peak_weekday_label} 소비 집중도 {_format_percent(concentration)}%가 확인되어 "
        "기간 편중 해석과 RAG 검색 후보로 넘김"
    )
    query_hint = f"{peak_weekday_label} 소비 집중 점검"
    if high_item_result is not None:
        item, _index = high_item_result
        merchant = str(item.get("merchant") or "고액 결제")
        amount = _format_amount(item.get("amount"))
        title = f"{peak_weekday_label} 고액 결제 집중 점검 타겟"
        linked_cause = f"{peak_weekday_label} 단일 고액 결제 중심의 주말 편중"
        reason = (
            f"{peak_weekday_label} 소비 집중도 {_format_percent(concentration)}%이고 "
            f"{merchant} {amount}원 결제가 같은 날 발생해 기간 편중 해석에 사용됨"
        )
        query_hint = f"{peak_weekday_label} 고액 결제 집중 점검"

    return InterventionTarget(
        target_type="period_concentration_review",
        title=title,
        linked_cause=linked_cause,
        target_json_path=concentration_path,
        reason=reason,
        query_hint=query_hint,
    )


def _should_drop_target(
    target: InterventionTarget,
    raw_json: Mapping[str, object],
) -> bool:
    """정규화할 수 없고 근거가 약한 개입 타겟을 제거할지 판단한다."""
    target_type = target.target_type.lower()
    if "impulse" in target_type:
        return True
    if "habitual" in target_type and "배달" not in target.title:
        return True
    if target.linked_cause in _FIELD_CAUSE_NAMES:
        return True

    category_entry = _category_entry_for_target(raw_json, target)
    if category_entry is not None:
        diff_amount = _as_number(category_entry.get("diff_amount"))
        if diff_amount is not None and diff_amount <= 0:
            return True
        return True

    path_value = _lookup_path(raw_json, target.target_json_path)
    path_map = _as_mapping(path_value)
    visit_count = _as_number((path_map or {}).get("visit_count"))
    return visit_count is not None and visit_count < 2


def _is_weak_aggregate_target(target: InterventionTarget) -> bool:
    """핵심 고액 결제 타겟이 있을 때 중복되는 집계성 개입 타겟인지 확인한다."""
    aggregate_path_prefixes = (
        "weekly_comparisons.",
        "weekly_metrics.",
        "waste_detection.late_night.",
    )
    aggregate_target_types = {
        "shopping_increase",
        "weekend_spending",
        "late_night_spending",
    }
    return target.target_type in aggregate_target_types or target.target_json_path.startswith(
        aggregate_path_prefixes
    )


def _sanitize_intervention_targets(
    targets: list[InterventionTarget],
    raw_json: Mapping[str, object],
) -> list[InterventionTarget]:
    """개입 타겟 후보를 RAG 검색에 적합한 구체적 후보로 정규화하고 중복을 제거한다."""
    auto_targets = [
        *_auto_high_spending_targets(raw_json),
        *_auto_delivery_targets(raw_json),
    ]
    candidate_targets = [
        *auto_targets,
        *targets,
    ]
    sanitized_targets: list[InterventionTarget] = []
    for target in candidate_targets:
        normalized_target = _normalize_high_spending_target(raw_json, target)
        if normalized_target is None:
            normalized_target = _normalize_delivery_target(raw_json, target)
        if normalized_target is None:
            normalized_target = _normalize_period_concentration_target(raw_json, target)
        if normalized_target is None:
            if auto_targets and _is_weak_aggregate_target(target):
                continue
            if _should_drop_target(target, raw_json):
                continue
            normalized_target = target

        sanitized_targets.append(normalized_target)
    return _dedupe_intervention_targets(sanitized_targets)


def _dedupe_intervention_targets(
    targets: list[InterventionTarget],
) -> list[InterventionTarget]:
    """개입 타겟 목록에서 유형과 경로가 같은 중복 항목을 제거한다."""
    deduped_targets: list[InterventionTarget] = []
    seen_keys: set[tuple[str, str]] = set()
    for target in targets:
        dedupe_key = (target.target_type, target.target_json_path)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped_targets.append(target)
    return deduped_targets


def _auto_high_spending_targets(raw_json: Mapping[str, object]) -> list[InterventionTarget]:
    """LLM이 놓친 핵심 비고정비·고정비 고액 결제 타겟을 자동으로 보강한다."""
    high_spending_items = _iter_high_spending_items(raw_json)
    if not high_spending_items:
        return []

    variable_candidates = [
        item_result
        for item_result in high_spending_items
        if not _is_fixed_cost_item(item_result[0])
    ]
    fixed_candidates = [
        item_result for item_result in high_spending_items if _is_fixed_cost_item(item_result[0])
    ]

    selected_items: list[tuple[Mapping[str, object], int, float]] = []
    if variable_candidates:
        selected_items.append(max(variable_candidates, key=lambda item_result: item_result[2]))
    if fixed_candidates:
        selected_items.extend(
            sorted(fixed_candidates, key=lambda item_result: item_result[2], reverse=True)[:2]
        )

    return [_normalize_high_spending_item(item, index) for item, index, _amount in selected_items]


def _auto_delivery_targets(raw_json: Mapping[str, object]) -> list[InterventionTarget]:
    """배달 결제 정보가 있으면 반복 단정 없이 배달 점검 타겟을 만든다."""
    delivery = _as_mapping(_lookup_path(raw_json, "repeat_patterns.delivery"))
    if delivery is None:
        return []

    count = int(_as_number(delivery.get("count")) or 0)
    amount = _format_amount(delivery.get("total_amount"))
    if count <= 0 or amount == "0":
        return []

    target_type = "delivery_frequency_review" if count >= 2 else "delivery_single_order_review"
    title = "배달 반복 결제 점검 타겟" if count >= 2 else "배달 단일 결제 점검 타겟"
    return [
        InterventionTarget(
            target_type=target_type,
            title=title,
            linked_cause=f"배달 {count}회 결제 {amount}원이 확인됨",
            target_json_path="repeat_patterns.delivery.total_amount",
            reason="배달 결제 금액이 확인되어 RAG 검색 후보로 넘김",
            query_hint="배달 결제 절약 점검",
        )
    ]


__all__ = ["sanitize_spending_analysis_payload"]
