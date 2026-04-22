from __future__ import annotations

import re
from collections.abc import Sequence

from langsmith.evaluation import EvaluationResult


def normalize_text(value: str) -> str:
    lowered = value.lower()
    alnum_only = re.sub(r"[^0-9a-z가-힣\s/_.-]+", " ", lowered)
    return " ".join(alnum_only.split())


def _extract_context_text(outputs: dict) -> str:
    contexts = outputs.get("contexts", [])
    if not isinstance(contexts, Sequence):
        return ""

    parts: list[str] = []
    for item in contexts:
        if isinstance(item, dict):
            parts.append(str(item.get("content", "")))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def answer_exact_match(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> EvaluationResult:
    expected = normalize_text(reference_outputs.get("answer", ""))
    actual = normalize_text(outputs.get("answer", ""))
    return EvaluationResult(
        key="answer_exact_match",
        score=bool(expected and actual and expected == actual),
    )


def answer_contains_reference(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> EvaluationResult:
    expected = normalize_text(reference_outputs.get("answer", ""))
    actual = normalize_text(outputs.get("answer", ""))
    return EvaluationResult(
        key="answer_contains_reference",
        score=bool(expected and actual and expected in actual),
    )


def retrieved_context_supports_reference(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> EvaluationResult:
    expected = normalize_text(reference_outputs.get("answer", ""))
    context_blob = normalize_text(_extract_context_text(outputs))
    return EvaluationResult(
        key="retrieved_context_supports_reference",
        score=bool(expected and context_blob and expected in context_blob),
    )


def has_retrieved_sources(
    inputs: dict,
    outputs: dict,
) -> EvaluationResult:
    sources = outputs.get("sources", [])
    return EvaluationResult(
        key="has_retrieved_sources",
        score=bool(isinstance(sources, Sequence) and len(sources) > 0),
    )


RAG_EVALUATORS = [
    answer_exact_match,
    answer_contains_reference,
    retrieved_context_supports_reference,
    has_retrieved_sources,
]
