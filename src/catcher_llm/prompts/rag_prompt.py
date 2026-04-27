from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_rag_prompt(system_instruction: str | None = None) -> ChatPromptTemplate:
    """검색된 문서만 근거로 답하도록 지시하는 기본 RAG 프롬프트를 만든다."""
    instruction = system_instruction or (
        "검색된 문서만 근거로 질문에 답하세요. "
        "문서가 부족하면 확인할 수 없다고 답하세요. "
        "답변 끝에는 사용한 출처를 간단히 언급하세요."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            (
                "human",
                "이전 대화\n{history}\n\n검색된 문서:\n{context}\n\n질문:\n{question}",
            ),
        ]
    )


def get_kca_report_prompt() -> ChatPromptTemplate:
    """KCA 보고서 기반 사실형 답변에 맞춘 전용 프롬프트를 반환한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 KCA 보고서 내용을 근거로 사실형 답변만 하는 분석 도우미다.

반드시 제공된 문서 context 안에서만 답한다.
context에 없는 내용은 추측하지 말고, 문서에서 확인되지 않는다고 답한다.
실천 팁, 일반 상식, 추가 조언, 이모지, 불필요한 격려 문구를 넣지 않는다.

답변 규칙:
1. 질문에 대한 직접 답을 첫 문장에 바로 쓴다.
2. 가능하면 문서에 나온 핵심 표현을 그대로 살린다.
3. 답변은 한 문장만 쓴다.
4. 수치나 순위가 있으면 우선 포함한다.
5. 목록형 문장, 굵은 강조, 섹션 제목을 쓰지 않는다.
""",
            ),
            (
                "human",
                """
질문:
{question}

문서 context:
{context}

답변:
""",
            ),
        ]
    )


def get_saving_tips_prompt() -> ChatPromptTemplate:
    """절약 팁 문서 기반 직접 답변에 맞춘 전용 프롬프트를 반환한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 절약 팁 문서 내용을 근거로 직접적인 절약 행동만 답하는 도우미다.

반드시 제공된 문서 context 안에서만 답한다.
context에 없는 내용은 추측하지 말고, 문서에서 확인되지 않는다고 답한다.
일반론, 추가 조언, 격려 문구, 배경 설명을 길게 덧붙이지 않는다.

답변 규칙:
1. 질문에 대한 직접 답을 첫 문장에 바로 쓴다.
2. 질문의 핵심 명사를 답변 첫머리에 다시 포함한다.
3. 가능하면 문서에 나온 절약 행동이나 정책 이름을 그대로 살린다.
4. 답변은 한 문장만 쓴다.
5. 항목이 여러 개면 쉼표로 이어서 간단히 정리한다.
6. 목록형 문장, 굵은 강조, 섹션 제목을 쓰지 않는다.
""",
            ),
            (
                "human",
                """
질문:
{question}

문서 context:
{context}

답변:
""",
            ),
        ]
    )


def get_self_report_prompt() -> ChatPromptTemplate:
    """소비 자기진단 리포트 기반 사실형 답변에 맞춘 전용 프롬프트를 반환한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 소비 자기진단 리포트 내용을 근거로 사실형 답변만 하는 분석 도우미다.

반드시 제공된 문서 context 안에서만 답한다.
context에 없는 내용은 추측하지 말고, 문서에서 확인되지 않는다고 답한다.
일반적인 재무 조언, 추가 설명, 격려 문구, 장황한 목록을 넣지 않는다.

답변 규칙:
1. 질문에 대한 직접 답을 첫 문장에 바로 쓴다.
2. 가능하면 문서에 나온 수치, 순위, 항목명을 그대로 살린다.
3. 답변은 한 문장만 쓴다.
4. 항목이 여러 개면 쉼표로 이어서 간단히 정리한다.
5. 목록형 문장, 굵은 강조, 섹션 제목을 쓰지 않는다.
""",
            ),
            (
                "human",
                """
질문:
{question}

문서 context:
{context}

답변:
""",
            ),
        ]
    )


def get_welfare_prompt() -> ChatPromptTemplate:
    """복지 정책 문서 기반 사실형 답변에 맞춘 전용 프롬프트를 반환한다."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 복지 정책 문서 내용을 근거로 사실형 답변만 하는 안내 도우미다.

반드시 제공된 문서 context 안에서만 답한다.
context에 없는 내용은 추측하지 말고, 문서에서 확인되지 않는다고 답한다.
일반적인 조언, 제도 해설, 추가 설명, 장황한 목록을 넣지 않는다.

답변 규칙:
1. 질문에 대한 직접 답을 첫 문장에 바로 쓴다.
2. 가능하면 문서에 나온 지원 대상, 지원 금액, 연령, 신청 경로 표현을 그대로 살린다.
3. 답변은 한 문장만 쓴다.
4. 수치가 있으면 우선 포함한다.
5. 목록형 문장, 굵은 강조, 섹션 제목을 쓰지 않는다.
""",
            ),
            (
                "human",
                """
질문:
{question}

문서 context:
{context}

답변:
""",
            ),
        ]
    )
