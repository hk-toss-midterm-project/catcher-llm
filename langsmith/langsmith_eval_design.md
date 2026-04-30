# Catcher LLM — LangSmith 성능평가 체계 설계

> 작성일: 2026-04-30  
> 대상: Catcher LLM (소비습관 교정 서비스)  
> 평가 플랫폼: LangSmith  

---

## A. 평가 전략 요약

### 핵심 철학

Catcher LLM은 "소비 데이터 → 행동 교정 피드백 → 절약 실천"으로 이어지는 **행동 변화 서비스**다. 따라서 평가 기준도 단순한 "정답 여부"가 아니라 **"이 출력이 실제 행동 변화를 유도하는가"** 에 초점을 맞춰야 한다.

일반 QA 서비스 평가와 결정적으로 다른 3가지 원칙은 다음과 같다.

**원칙 1 — 데이터 근거성 우선**  
소비 데이터에 없는 수치·패턴을 만들어내는 순간, 서비스 신뢰가 붕괴된다. groundedness 항목은 어떤 태스크에서도 4점 미만이면 전체 실패로 처리한다.

**원칙 2 — 기간 역할 분리**  
일간·주간·월간은 서로 다른 역할을 가진다. 일간 피드백이 월간 수준의 구조 분석을 하거나, 월간 피드백이 "내일 할 일 1가지"로 끝나면 기능 목적 자체를 벗어난 실패다. period fit은 별도 evaluator로 독립 측정한다.

**원칙 3 — RAG는 문서 경계 안에서만**  
정책 추천은 검색된 문서에 없는 내용을 절대 생성해선 안 된다. 일반 상식으로 문서를 보완하는 것 자체가 hallucination으로 분류된다.

### 평가 운영 구조

```
[LangSmith]
    ├── Dataset (태스크별 5개)
    │     ├── daily_feedback_dataset
    │     ├── weekly_feedback_dataset
    │     ├── monthly_feedback_dataset
    │     ├── action_suggestion_dataset
    │     └── policy_rag_dataset
    │
    ├── Evaluators
    │     ├── Heuristic Evaluators  (규칙 기반, 빠름)
    │     ├── LLM-as-a-Judge        (rubric 기반, 정밀)
    │     └── Binary Evaluators     (hallucination 필터)
    │
    └── Experiments
          ├── prompt_ab_test
          ├── model_comparison
          └── retrieval_config_comparison
```

---

## B. 태스크별 평가 항목 표

### B-1. 평가 태스크를 5개로 나누는 이유

| 태스크 | 독립 평가가 필요한 이유 |
|---|---|
| 일간 피드백 | "오늘 문제 + 내일 행동 1개"라는 초단기 임팩트 구조. 월간과 섞이면 간결성 기준이 흐려짐 |
| 주간 피드백 | 반복 패턴 발견과 규칙 제안이 핵심. 단순 합산이 아닌 패턴 추출 능력을 별도 측정해야 함 |
| 월간 피드백 | 소비 구조 분석 + 다음 달 목표 설정. 정보 밀도와 전략적 깊이를 측정해야 함 |
| 액션 제안 | 피드백과 달리 "즉시 실행 가능성"이 가장 중요. 분석보다 처방 품질을 측정 |
| 정책 RAG | 문서 근거 충실성이라는 완전히 다른 품질 축이 추가됨. 다른 태스크 기준과 혼합 불가 |

### B-2. 태스크별 핵심 평가 항목

| 평가 항목 | 정의 | 일간 | 주간 | 월간 | 액션 | RAG |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **groundedness** | 출력 내 수치·패턴이 모두 입력 데이터에서 도출된 것인가 | ✅ | ✅ | ✅ | ✅ | ✅ |
| **actionability** | 사용자가 당장 실행할 수 있는 구체적 행동을 담고 있는가 | ✅ | ✅ | ✅ | ✅ | ✅ |
| **personalization** | 이 사용자의 소비 데이터에 특화된 내용인가 (일반론 배제) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **tone_fit** | 친근하되 비난하지 않는 잔소리형 톤을 유지하는가 | ✅ | ✅ | ✅ | ✅ | — |
| **period_fit** | 일간/주간/월간 각 기간의 역할에 맞는 내용과 깊이인가 | ✅ | ✅ | ✅ | — | — |
| **conciseness** | 일간 피드백이 지나치게 길거나 장황하지 않은가 | ✅ | — | — | ✅ | — |
| **pattern_detection** | 반복 소비 습관을 정확히 포착하고 설명하는가 | — | ✅ | ✅ | — | — |
| **strategy_depth** | 다음 주/월 전략 제안이 표면적이지 않고 구체적인가 | — | ✅ | ✅ | — | — |
| **action_specificity** | "절약하세요" 수준이 아닌 카테고리·행동 특정 제안인가 | — | — | — | ✅ | — |
| **faithfulness** | 검색된 문서 범위 안에서만 정보를 생성하는가 | — | — | — | — | ✅ |
| **relevance** | 추천 정책이 사용자 소비 상황과 실제로 연관되는가 | — | — | — | — | ✅ |
| **hallucination_risk** | 문서에 없는 정보를 사실처럼 생성하는 위험이 있는가 | — | — | — | — | ✅ |

### B-3. 항목별 세부 정의 (Catcher LLM 특화)

**groundedness (근거성)**  
- 4점: 출력의 모든 수치, 카테고리, 패턴이 `structured_metrics` 필드에서 직접 확인됨  
- 3점: 대부분 근거가 있으나 1개 정도 추론이 포함됨  
- 2점: 수치 2개 이상이 입력에 없거나 과장됨  
- 1점: 근거 없는 수치/패턴이 출력의 핵심을 구성함  

**actionability (실행가능성)**  
- 4점: 오늘/이번 주 바로 실행할 수 있는 행동을 구체적 카테고리·방법으로 제시  
- 3점: 행동 제안은 있으나 다소 추상적 ("배달 줄이기")  
- 2점: "절약이 필요합니다" 수준의 일반 권고만 있음  
- 1점: 행동 제안이 없거나 불가능한 수준  

**personalization (개인화)**  
- 4점: 이 사용자의 특정 거래내역·패턴을 명시적으로 언급  
- 3점: 카테고리는 맞지만 일반 수준의 설명  
- 2점: 이 사용자 아닌 누구에게도 해당되는 일반론  
- 1점: 소비 데이터를 무시한 generic 답변  

**period_fit (기간 적합성)**  
- 4점: 기간별 역할(일간=내일 행동 1개, 주간=패턴+규칙, 월간=구조+목표)을 완전히 충족  
- 3점: 역할에 근접하나 일부 이탈  
- 2점: 기간 목적의 절반 이하만 충족  
- 1점: 기간 역할과 무관한 내용  

**faithfulness (RAG 충실성)**  
- 4점: 출력의 모든 정보가 `retrieved_context`에서 확인 가능  
- 3점: 대부분 문서 근거가 있으나 1개 일반론이 추가됨  
- 2점: 문서 기반과 일반 지식이 혼합됨  
- 1점: 검색 문서를 사실상 무시하고 일반 지식으로 대체함  

---

## C. Dataset 설계안

### C-1. 일간 피드백 Dataset

**필드 구조:**

```json
{
  "id": "daily_001",
  "task_type": "daily_feedback",
  "analysis_period": {
    "date": "2024-03-15",
    "day_of_week": "금요일"
  },
  "user_profile": {
    "age_group": "20대 후반",
    "income_tier": "중간",
    "saving_goal": "월 20만원 절약"
  },
  "structured_metrics": {
    "total_spent_today": 47500,
    "transactions": [
      {"category": "카페", "amount": 6500, "merchant": "스타벅스", "time": "08:30"},
      {"category": "배달", "amount": 18900, "merchant": "배달의민족", "time": "12:15"},
      {"category": "편의점", "amount": 4200, "merchant": "CU", "time": "15:00"},
      {"category": "배달", "amount": 17900, "merchant": "쿠팡이츠", "time": "19:30"}
    ],
    "worst_spend": {"category": "배달", "total": 36800, "ratio": 0.77},
    "daily_budget": 30000,
    "budget_exceeded_by": 17500
  },
  "user_request": "오늘 소비 피드백 줘",
  "reference_output": "오늘 47,500원 중 배달에만 36,800원 썼어요 (77%). 예산도 17,500원 초과했고요. 내일 점심은 편의점 도시락으로 대체해보는 건 어때요? 한 끼당 4,000원대면 오늘 배달비의 1/4이에요.",
  "metadata": {
    "difficulty": "medium",
    "failure_risk": ["근거없는_수치생성", "일반론_조언"],
    "annotator": "human_v1"
  }
}
```

### C-2. 주간 피드백 Dataset

**필드 구조:**

```json
{
  "id": "weekly_001",
  "task_type": "weekly_feedback",
  "analysis_period": {
    "week": "2024-W11",
    "start_date": "2024-03-11",
    "end_date": "2024-03-17",
    "prev_week": "2024-W10"
  },
  "user_profile": {
    "age_group": "30대 초반",
    "income_tier": "중간",
    "saving_goal": "월 30만원 절약"
  },
  "structured_metrics": {
    "this_week": {
      "total": 312000,
      "by_category": {
        "카페": 34000,
        "배달": 89000,
        "편의점": 21000,
        "택시": 42000,
        "구독": 18900
      }
    },
    "prev_week": {
      "total": 285000,
      "by_category": {
        "카페": 31000,
        "배달": 76000,
        "편의점": 18500,
        "택시": 38000,
        "구독": 18900
      }
    },
    "repeated_patterns": [
      {"pattern": "화요일_목요일_카페_출근전", "frequency": 2, "avg_amount": 6200},
      {"pattern": "주중_점심_배달", "frequency": 4, "avg_amount": 15800},
      {"pattern": "주말_택시_귀가", "frequency": 2, "avg_amount": 14500}
    ],
    "week_over_week_change": {
      "total_change": 27000,
      "total_change_pct": 9.5,
      "biggest_increase": {"category": "배달", "change": 13000}
    }
  },
  "user_request": "이번 주 소비 피드백 줘",
  "reference_output": "이번 주 배달비가 89,000원으로 전주보다 13,000원(+17%) 올랐어요. 화·목 출근길 카페 패턴도 고정됐고, 주말 택시도 2회 반복됐고요. 다음 주는 딱 하나만 바꿔볼게요. 주중 점심 배달을 주 2회로 제한하고, 나머지는 편의점·도시락으로 대체하면 약 30,000원 줄일 수 있어요.",
  "metadata": {
    "difficulty": "hard",
    "failure_risk": ["기간구분실패", "패턴발견누락", "전주비교누락"],
    "annotator": "human_v1"
  }
}
```

### C-3. 월간 피드백 Dataset

**필드 구조:**

```json
{
  "id": "monthly_001",
  "task_type": "monthly_feedback",
  "analysis_period": {
    "month": "2024-03",
    "prev_month": "2024-02"
  },
  "user_profile": {
    "age_group": "20대 후반",
    "income_tier": "중간",
    "monthly_income": 2800000,
    "saving_goal": "월 30만원 이상 저축"
  },
  "structured_metrics": {
    "this_month": {
      "total": 1340000,
      "by_category": {
        "식비_배달": 320000,
        "카페": 89000,
        "교통_택시": 78000,
        "편의점": 65000,
        "구독": 54000,
        "의류": 130000,
        "기타": 604000
      },
      "income_ratio": 0.479
    },
    "prev_month": {
      "total": 1180000,
      "income_ratio": 0.421
    },
    "top_category": "식비_배달",
    "easiest_reduction_category": "구독",
    "easiest_reduction_reason": "사용 빈도 낮고 중복 가능성 높음",
    "reduction_potential": {
      "구독": 20000,
      "카페": 30000,
      "편의점": 20000
    }
  },
  "user_request": "이번 달 소비 분석해줘",
  "reference_output": "3월 총 지출 1,340,000원 — 수입의 47.9%를 썼어요. 전월(42.1%)보다 5.8%p 높아졌고요. 식비/배달이 320,000원으로 1위이지만 줄이기 가장 쉬운 건 구독(54,000원)이에요. 사용 빈도 낮은 구독 1~2개를 정리하는 것부터 시작해서, 4월 목표는 지출 1,200,000원 이하로 잡아봐요. 먼저 구독 목록 정리부터.",
  "metadata": {
    "difficulty": "hard",
    "failure_risk": ["기간구분실패", "근거없는목표설정", "개인화부족"],
    "annotator": "human_v1"
  }
}
```

### C-4. 액션 제안 Dataset

**필드 구조:**

```json
{
  "id": "action_001",
  "task_type": "action_suggestion",
  "problem_type": "배달",
  "user_profile": {
    "age_group": "20대 후반",
    "saving_goal": "배달비 월 5만원 줄이기"
  },
  "structured_metrics": {
    "delivery_monthly": 280000,
    "delivery_frequency": 18,
    "avg_per_order": 15556,
    "delivery_peak_days": ["화요일", "목요일", "일요일"],
    "delivery_peak_time": "점심(12-13시), 저녁(19-20시)",
    "current_subscriptions": ["쿠팡로켓와우", "배달의민족클럽"]
  },
  "user_request": "배달비 줄이는 방법 알려줘",
  "reference_output": "이달 배달 18회, 280,000원이에요. 화·목 점심이 가장 많아요. 당장 할 수 있는 것 3가지: ① 화·목 점심은 직장 근처 식당 1곳 고정 (1회 약 8,000원 절약) ② 배민클럽 해지 (월 9,900원, 혜택 대비 주문 빈도 낮음) ③ 일요일 저녁은 밀키트 하나 사두기 (주문 충동 차단). 이 3개만 해도 월 5~6만원 줄 수 있어요.",
  "metadata": {
    "difficulty": "medium",
    "failure_risk": ["추상적_조언", "개인화부족"],
    "annotator": "human_v1"
  }
}
```

### C-5. 정책 RAG Dataset

**필드 구조:**

```json
{
  "id": "rag_001",
  "task_type": "policy_rag",
  "user_profile": {
    "age_group": "20대 중반",
    "income_tier": "저소득",
    "monthly_income": 1800000,
    "residence": "서울",
    "household_type": "1인가구"
  },
  "structured_metrics": {
    "monthly_total": 1650000,
    "income_ratio": 0.917,
    "problematic_categories": ["식비", "주거비"],
    "saving_amount_last_3m": 0
  },
  "user_request": "나 같은 상황에서 받을 수 있는 정부 지원 있어?",
  "retrieved_context": [
    {
      "doc_id": "policy_001",
      "source": "2024_청년_주거지원_가이드.pdf",
      "content": "청년 월세 지원사업: 만 19~34세, 1인 가구, 월소득 중위 60% 이하(2024년 기준 약 1,335,000원), 임차보증금 5천만원 이하, 월세 70만원 이하인 경우 월 최대 20만원 지원. 신청: 복지로(www.bokjiro.go.kr)"
    },
    {
      "doc_id": "policy_002",
      "source": "2024_청년내일저축계좌_안내.pdf",
      "content": "청년내일저축계좌: 만 19~34세, 근로 중인 청년, 기준 중위소득 100% 이하, 월 10만원 저축 시 정부 매칭 최대 30만원. 3년 만기 시 1,440만원 적립 가능. 신청: 주민센터 또는 복지로"
    }
  ],
  "reference_output": "소득이 수입의 90%를 넘기고 있어서 구조적인 지원이 필요한 상황이에요. 문서 기준으로 2가지를 확인해보세요. ① 청년 월세 지원: 월 최대 20만원 지원인데, 소득 기준(중위 60%, 약 133만원)을 먼저 확인해야 해요. 지금 180만원이면 해당 안 될 수 있어요. ② 청년내일저축계좌: 중위 100% 이하면 가능하고, 월 10만원 저축 시 정부가 최대 30만원 매칭해줘요. 둘 다 복지로(bokjiro.go.kr)나 주민센터에서 신청해요.",
  "metadata": {
    "difficulty": "hard",
    "failure_risk": ["hallucination", "문서없는정책추가", "소득기준오류"],
    "annotator": "human_v1"
  }
}
```

---

## D. Evaluator 설계안

### D-1. Evaluator 유형별 역할

| Evaluator 유형 | 속도 | 정밀도 | 주요 사용 태스크 |
|---|---|---|---|
| Heuristic (규칙 기반) | 빠름 | 낮음 | 길이 검사, 필수 키워드 포함 여부 |
| Binary | 빠름 | 중간 | hallucination 1차 필터 |
| Score-based (LLM Judge) | 중간 | 높음 | groundedness, actionability, personalization |
| Rubric-based (LLM Judge) | 느림 | 최고 | faithfulness, 종합 품질 |

### D-2. 구체적 Evaluator 목록

**Heuristic Evaluators**

```python
# evaluator_heuristic.py

def check_length_daily(run, example):
    """일간 피드백은 300자 이하여야 함"""
    output = run.outputs.get("output", "")
    passed = len(output) <= 300
    return {"key": "length_ok_daily", "score": 1 if passed else 0}

def check_no_generic_phrases(run, example):
    """'절약이 필요합니다' 같은 금지 표현 필터"""
    forbidden = ["절약이 필요합니다", "소비를 줄여야", "돈을 아껴야", "재정 관리가 중요"]
    output = run.outputs.get("output", "")
    has_forbidden = any(phrase in output for phrase in forbidden)
    return {"key": "no_generic_advice", "score": 0 if has_forbidden else 1}

def check_contains_amount(run, example):
    """구체적 금액 언급 여부 (숫자+원 패턴)"""
    import re
    output = run.outputs.get("output", "")
    has_amount = bool(re.search(r'\d{1,3}(,\d{3})*원', output))
    return {"key": "contains_specific_amount", "score": 1 if has_amount else 0}

def check_action_verb(run, example):
    """액션 제안에 구체적 동사 포함 여부"""
    action_verbs = ["해지", "줄이기", "대체", "고정", "제한", "사두기", "취소"]
    output = run.outputs.get("output", "")
    has_action = any(v in output for v in action_verbs)
    return {"key": "has_action_verb", "score": 1 if has_action else 0}
```

**Binary Evaluator (Hallucination 1차 필터)**

```python
# evaluator_binary_hallucination.py

HALLUCINATION_FILTER_PROMPT = """
다음은 소비 데이터 입력과 AI 출력입니다.
출력에 입력 데이터에 없는 수치나 카테고리가 포함되어 있으면 1, 없으면 0을 반환하세요.

[입력 데이터]
{structured_metrics}

[AI 출력]
{model_output}

JSON으로만 응답: {{"hallucination_detected": 0 또는 1, "reason": "..."}}
"""

def binary_hallucination_check(run, example):
    # LLM 호출로 빠른 1차 필터
    result = call_llm(HALLUCINATION_FILTER_PROMPT.format(
        structured_metrics=example.inputs["structured_metrics"],
        model_output=run.outputs["output"]
    ))
    detected = result["hallucination_detected"]
    return {
        "key": "hallucination_free",
        "score": 1 - detected,  # 0이면 clean, 1이면 오염
        "comment": result["reason"]
    }
```

**Score-based LLM-as-a-Judge Evaluator**

```python
# evaluator_llm_judge.py

def run_judge(prompt_template, run, example, key):
    filled = prompt_template.format(
        user_profile=example.inputs.get("user_profile", {}),
        analysis_period=example.inputs.get("analysis_period", {}),
        structured_metrics=example.inputs.get("structured_metrics", {}),
        user_request=example.inputs.get("user_request", ""),
        model_output=run.outputs.get("output", ""),
        reference_output=example.outputs.get("reference_output", "")
    )
    result = call_llm(filled)
    return {"key": key, "score": result["score"], "comment": result["reason"]}
```

---

## E. LangSmith용 Judge Prompt

### E-1. 소비 피드백 평가용 Judge Prompt

```
당신은 소비습관 교정 서비스 "Catcher LLM"의 AI 출력을 평가하는 전문 심사관입니다.
이 서비스는 단순 가계부가 아니라 사용자의 소비 행동을 변화시키는 것이 목표입니다.
따라서 "맞는 말"이 아니라 "이 사람이 실제로 행동을 바꿀 수 있는 출력인가"를 기준으로 평가하세요.

---

[평가 대상 태스크]
태스크 유형: {task_type}  (daily / weekly / monthly 중 하나)
분석 기간: {analysis_period}

[사용자 프로필]
{user_profile}

[입력 소비 데이터]
{structured_metrics}

[사용자 요청]
{user_request}

[AI 출력 (평가 대상)]
{model_output}

[참조 출력 (참고용, 절대적 정답 아님)]
{reference_output}

---

[평가 기준] 각 항목을 1~4점으로 채점하고 이유를 작성하세요.

## 1. groundedness (근거성) — 배점: 4점
AI 출력의 수치, 카테고리, 패턴이 모두 [입력 소비 데이터]에 근거하는가?
- 4점: 모든 수치·패턴이 입력 데이터에서 확인 가능
- 3점: 대부분 근거 있으나 소소한 추론 1개 포함
- 2점: 2개 이상의 수치·패턴이 입력에 없거나 과장됨
- 1점: 핵심 내용이 입력 데이터와 불일치하거나 근거 없음
점수: [  ] / 이유: [          ]

## 2. actionability (실행가능성) — 배점: 4점
사용자가 오늘/이번 주 당장 실행할 수 있는 구체적 행동을 제시하는가?
- 4점: 카테고리·방법·시기가 명확한 실행 가능한 행동 제시
- 3점: 행동 제안은 있으나 다소 추상적 (예: "배달 줄이기")
- 2점: "절약하세요" 수준의 권고만 있음
- 1점: 행동 제안 없음 또는 불가능한 제안
점수: [  ] / 이유: [          ]

## 3. personalization (개인화) — 배점: 4점
이 특정 사용자의 소비 패턴에 특화된 내용인가, 아니면 누구에게나 해당되는 일반론인가?
- 4점: 이 사용자의 특정 거래내역·패턴을 명시적으로 언급
- 3점: 카테고리는 맞지만 일반 수준의 설명
- 2점: 이 사용자가 아닌 일반 절약 팁 수준
- 1점: 소비 데이터를 거의 반영하지 않은 generic 답변
점수: [  ] / 이유: [          ]

## 4. tone_fit (톤 적합성) — 배점: 4점
친근하고 직접적이되, 비난하거나 과도하게 부정적이지 않은가?
- 4점: 임팩트 있는 잔소리형이지만 공격적이지 않음. 응원하는 느낌
- 3점: 적절한 톤이나 일부 딱딱하거나 다소 형식적
- 2점: 비난하는 표현 또는 과도하게 부드러워서 임팩트 없음
- 1점: 모욕적이거나 반대로 아무 자극도 없는 무의미한 위로
점수: [  ] / 이유: [          ]

## 5. period_fit (기간 역할 적합성) — 배점: 4점
태스크 유형({task_type})의 목적에 맞는 내용과 깊이인가?
[daily 기준] 오늘 문제 1개 포착 + 내일 바로 할 행동 1개 제안
[weekly 기준] 반복 패턴 발견 + 전주 비교 + 다음 주 규칙 제안
[monthly 기준] 소비 구조 분석 + 줄이기 쉬운 카테고리 + 다음 달 목표 설정
- 4점: 기간 목적을 완전히 충족
- 3점: 기간 목적의 70% 이상 충족
- 2점: 기간 목적의 절반 이하 충족, 또는 다른 기간의 역할과 혼용
- 1점: 기간 목적과 완전히 다른 내용
점수: [  ] / 이유: [          ]

---

[종합 평가]
총점: [  ] / 20점

전체 코멘트 (2~3문장):
[          ]

---

[JSON 출력 형식]
반드시 아래 JSON 형식으로만 응답하세요:

{
  "groundedness": {"score": 0, "reason": ""},
  "actionability": {"score": 0, "reason": ""},
  "personalization": {"score": 0, "reason": ""},
  "tone_fit": {"score": 0, "reason": ""},
  "period_fit": {"score": 0, "reason": ""},
  "total_score": 0,
  "overall_comment": "",
  "pass": true
}

pass 기준: groundedness >= 4 AND actionability >= 3 AND personalization >= 3 AND total_score >= 14
```

---

### E-2. 문서 RAG 평가용 Judge Prompt

```
당신은 소비습관 교정 서비스 "Catcher LLM"의 정책 추천 RAG 출력을 평가하는 전문 심사관입니다.
이 평가에서 가장 중요한 원칙은 단 하나입니다:
"AI 출력의 모든 정보는 [검색된 문서]에서 찾을 수 있어야 한다."
문서에 없는 정보를 그럴듯하게 덧붙이는 것 자체가 실패입니다. 상식적으로 맞는 말이라도 문서 외 정보면 감점입니다.

---

[사용자 프로필]
{user_profile}

[입력 소비 데이터]
{structured_metrics}

[사용자 요청]
{user_request}

[검색된 정책 문서]
{retrieved_context}

[AI 출력 (평가 대상)]
{model_output}

[참조 출력 (참고용)]
{reference_output}

---

[평가 기준] 각 항목을 1~4점으로 채점하세요.

## 1. faithfulness (문서 충실성) — 배점: 4점
AI 출력의 모든 정보가 [검색된 정책 문서]에서 확인 가능한가?
- 4점: 출력의 모든 수치·조건·내용이 문서에서 직접 확인 가능
- 3점: 대부분 문서 기반이나, 문서에 없는 일반론 1개 추가됨
- 2점: 문서 기반 정보와 일반 지식이 혼재하여 구분이 어려움
- 1점: 문서를 사실상 무시하고 일반 금융/복지 지식으로 대체함
점수: [  ] / 이유: [          ]

## 2. hallucination_risk (환각 위험) — 배점: 4점
문서에 없는 정책명, 금액, 조건, URL, 기관명을 생성했는가? (역채점: 없을수록 높은 점수)
- 4점: 환각 없음. 모든 정보가 문서에서 검증됨
- 3점: 사소한 표현 변형이 있으나 정보 왜곡은 없음
- 2점: 문서 조건과 다른 수치 또는 조건이 1~2개 포함됨
- 1점: 존재하지 않는 정책명·금액·조건이 사실처럼 서술됨
점수: [  ] / 이유: [          ]

## 3. relevance (사용자 상황 관련성) — 배점: 4점
추천된 정책이 이 사용자의 소비 상황, 소득 수준, 연령에 실제로 연관되는가?
- 4점: 사용자 소비 데이터와 프로필에 직접 연결되는 이유를 명시
- 3점: 정책이 적절하나 왜 이 사용자에게 해당되는지 설명이 부족
- 2점: 소비 상황과 무관한 일반 정책 나열
- 1점: 사용자 프로필 조건에 맞지 않는 정책을 추천함
점수: [  ] / 이유: [          ]

## 4. actionability (실행가능성) — 배점: 4점
정책 신청 방법, 조건 확인 방법을 문서 기반으로 안내하는가?
- 4점: 신청 경로·URL·조건 확인 방법까지 문서 기반으로 안내
- 3점: 신청 방법 안내가 있으나 일부 모호함
- 2점: 정책만 언급하고 어떻게 신청하는지 안내 없음
- 1점: 신청 정보 없음, 또는 잘못된 신청 방법 안내
점수: [  ] / 이유: [          ]

## 5. transparency (한계 투명성) — 배점: 4점
사용자 조건이 정책 수혜 기준에 해당 안 될 수 있을 때 솔직하게 고지하는가?
- 4점: 조건 불일치 가능성을 명시하고 확인을 권유함
- 3점: 불확실성을 언급하나 다소 모호함
- 2점: 조건 미충족 가능성이 명확한데도 "신청하세요"로만 끝냄
- 1점: 맞지 않는 조건임에도 확실하게 해당된다고 오안내
점수: [  ] / 이유: [          ]

---

[종합 평가]
총점: [  ] / 20점

전체 코멘트 (2~3문장):
[          ]

---

[JSON 출력 형식]
반드시 아래 JSON 형식으로만 응답하세요:

{
  "faithfulness": {"score": 0, "reason": ""},
  "hallucination_risk": {"score": 0, "reason": ""},
  "relevance": {"score": 0, "reason": ""},
  "actionability": {"score": 0, "reason": ""},
  "transparency": {"score": 0, "reason": ""},
  "total_score": 0,
  "overall_comment": "",
  "pass": true
}

pass 기준: faithfulness >= 4 AND hallucination_risk >= 4 AND total_score >= 14
(faithfulness와 hallucination_risk 중 하나라도 4 미만이면 자동 FAIL)
```

---

## F. Pass/Fail 기준

### F-1. 일간 피드백

| 평가 항목 | Pass 기준 | 실패 시 처리 |
|---|---|---|
| groundedness | **4점 이상** (필수) | 자동 FAIL — 이유 불문 |
| actionability | 3점 이상 | 경고 태그, 재평가 대상 |
| personalization | 3점 이상 | 경고 태그 |
| tone_fit | 3점 이상 | 경고 태그 |
| period_fit | 3점 이상 | 경고 태그 |
| **총점** | **14점 이상 / 20점** | 미달 시 FAIL |
| no_generic_advice (heuristic) | 1 (pass) | 자동 FAIL |

### F-2. 주간 피드백

| 평가 항목 | Pass 기준 | 실패 시 처리 |
|---|---|---|
| groundedness | **4점 이상** (필수) | 자동 FAIL |
| pattern_detection | 3점 이상 | 경고 태그 |
| actionability | 3점 이상 | 경고 태그 |
| personalization | 3점 이상 | 경고 태그 |
| period_fit | **4점 이상** (필수) | 자동 FAIL — 주간 역할 이탈은 치명적 |
| **총점** | **14점 이상 / 20점** | 미달 시 FAIL |

### F-3. 월간 피드백

| 평가 항목 | Pass 기준 | 실패 시 처리 |
|---|---|---|
| groundedness | **4점 이상** (필수) | 자동 FAIL |
| strategy_depth | 3점 이상 | 경고 태그 |
| personalization | 3점 이상 | 경고 태그 |
| period_fit | **4점 이상** (필수) | 자동 FAIL |
| **총점** | **15점 이상 / 20점** | 미달 시 FAIL (월간은 기준 엄격) |

### F-4. 액션 제안

| 평가 항목 | Pass 기준 | 실패 시 처리 |
|---|---|---|
| groundedness | **4점 이상** (필수) | 자동 FAIL |
| action_specificity | **4점 이상** (필수) | 자동 FAIL — 액션이 애매하면 서비스 존재 이유 없음 |
| personalization | 3점 이상 | 경고 태그 |
| no_generic_advice (heuristic) | 1 | 자동 FAIL |
| **총점** | **14점 이상 / 20점** | 미달 시 FAIL |

### F-5. 정책 RAG

| 평가 항목 | Pass 기준 | 실패 시 처리 |
|---|---|---|
| faithfulness | **4점 이상** (필수) | 자동 FAIL — 문서 외 정보는 신뢰 붕괴 |
| hallucination_risk | **4점 이상** (필수) | 자동 FAIL |
| relevance | 3점 이상 | 경고 태그 |
| actionability | 3점 이상 | 경고 태그 |
| transparency | 3점 이상 | 경고 태그 |
| **총점** | **14점 이상 / 20점** | 미달 시 FAIL |

### F-6. 전체 서비스 Pass 기준

```
전체 Pass = 5개 태스크 PASS율 >= 80%
           AND groundedness 평균 >= 3.8
           AND hallucination_risk (RAG) 평균 >= 3.8
```

---

## G. 실패 유형 Taxonomy

### G-1. 분류 체계

| ID | 실패 유형 | 태스크 | 심각도 | 설명 | 개선 방향 |
|---|---|---|---|---|---|
| F-01 | **근거 없는 수치 생성** | 전체 | 🔴 Critical | "이번 달 카페비 12만원" 등 입력 데이터에 없는 금액 생성 | groundedness 프롬프트 강화, 입력 데이터 포맷 명시화 |
| F-02 | **일반론 조언** | 전체 | 🟠 High | "절약하는 습관을 기르세요"처럼 소비 데이터를 반영 안 함 | personalization 지시 강화, few-shot에 나쁜 예시 추가 |
| F-03 | **기간 구분 실패** | 일/주/월 | 🟠 High | 일간 피드백에서 "이번 달 목표"를 설정하거나, 월간에서 "내일 행동 1개"로 끝냄 | 시스템 프롬프트에 기간별 출력 구조 명시 |
| F-04 | **문서 외 정보 추가 (RAG)** | RAG | 🔴 Critical | 검색된 문서에 없는 정책명, 금액, 조건을 사실처럼 추가 | context-only 지시 강화, 문서 인용 강제 |
| F-05 | **개인화 부족** | 전체 | 🟠 High | 이 사용자의 특정 소비 패턴 대신 일반적인 유형 설명 | 사용자 거래내역 강조, 개인화 few-shot 추가 |
| F-06 | **과도하게 비난하는 말투** | 일/주/월 | 🟡 Medium | "이렇게 쓰면 안 되죠", "낭비예요" 같은 표현 | tone instruction 강화, forbidden phrase 필터 추가 |
| F-07 | **너무 긴 일간 피드백** | 일간 | 🟡 Medium | 일간 피드백이 500자 이상으로 월간 수준의 정보 밀도 | 출력 길이 제한 명시, 구조 템플릿 적용 |
| F-08 | **패턴 발견 누락** | 주간/월간 | 🟠 High | 반복 소비 패턴이 있는데도 단순 합산만 제공 | 패턴 분석 단계를 chain-of-thought로 분리 |
| F-09 | **소득 기준 오판 (RAG)** | RAG | 🔴 Critical | 문서의 소득 기준과 사용자 소득을 비교하지 않고 무조건 추천 | 조건 매칭 로직을 RAG 파이프라인에 명시 |
| F-10 | **전주 비교 누락** | 주간 | 🟡 Medium | 이번 주 데이터만 분석하고 전주 대비 변화를 언급하지 않음 | 주간 프롬프트에 비교 분석 필수 지시 |
| F-11 | **실행 불가능한 제안** | 액션 | 🟠 High | "소비 패턴을 바꾸세요" 같이 즉시 실행 불가능한 제안 | 액션 제안 템플릿에 카테고리·시기·방법 3요소 필수화 |
| F-12 | **구독 중복 감지 실패** | 월간/액션 | 🟡 Medium | 동일 기능 구독 서비스를 나란히 보유 중인데도 언급하지 않음 | 구독 카테고리 분석 로직 추가 |

### G-2. 실패 유형별 LangSmith 태그

```python
FAILURE_TAGS = {
    "F-01": "hallucination_numeric",
    "F-02": "generic_advice",
    "F-03": "period_mismatch",
    "F-04": "rag_out_of_context",
    "F-05": "low_personalization",
    "F-06": "blaming_tone",
    "F-07": "length_overflow_daily",
    "F-08": "pattern_missed",
    "F-09": "eligibility_mismatch_rag",
    "F-10": "prev_week_comparison_missing",
    "F-11": "unactionable_suggestion",
    "F-12": "subscription_overlap_missed"
}
```

---

## H. 실제 운영 권장안

### H-1. Phase별 운영 로드맵

```
Phase 1 — 기반 구축 (Week 1~2)
├── 태스크별 dataset 각 20개 이상 구축 (인간 레이블링)
├── Heuristic evaluator 배포 및 검증
└── Judge prompt 초안 검증 (인간 평가와 상관계수 측정)

Phase 2 — 기준선 수립 (Week 3~4)
├── 현재 프롬프트로 baseline 실험 실행
├── 태스크별 pass/fail 기준 미세 조정
└── 실패 패턴 1차 분류 및 taxonomy 정제

Phase 3 — 반복 개선 (Month 2~)
├── Prompt A/B 테스트
├── 모델 비교 실험
├── RAG 설정 비교
└── 실패 사례 재평가 루프
```

### H-2. Prompt A/B 테스트 설계

```python
# LangSmith experiment 설정 예시
from langsmith import Client
from langchain.smith import RunEvalConfig

client = Client()

# Experiment A: 현재 프롬프트
experiment_a = client.run_on_dataset(
    dataset_name="daily_feedback_dataset",
    llm_or_chain_factory=lambda: build_chain(prompt="prompt_v1"),
    evaluation=RunEvalConfig(
        evaluators=["groundedness_judge", "actionability_judge"],
        custom_evaluators=[check_no_generic_phrases, check_contains_amount]
    ),
    experiment_prefix="daily_v1"
)

# Experiment B: 개선된 프롬프트
experiment_b = client.run_on_dataset(
    dataset_name="daily_feedback_dataset",
    llm_or_chain_factory=lambda: build_chain(prompt="prompt_v2"),
    evaluation=RunEvalConfig(
        evaluators=["groundedness_judge", "actionability_judge"],
        custom_evaluators=[check_no_generic_phrases, check_contains_amount]
    ),
    experiment_prefix="daily_v2"
)
```

**A/B 테스트 판정 기준:**

```
승리 조건:
- groundedness 평균 개선 >= 0.2점
- actionability 평균 개선 >= 0.2점
- PASS율 개선 >= 5%p
- tone_fit 하락 없음 (기존 대비 -0.1 이내)
```

### H-3. 모델 비교 실험 매트릭스

| 실험 변수 | 비교 대상 | 판정 지표 |
|---|---|---|
| 모델 크기 | GPT-4o vs Claude Sonnet vs 소형 모델 | groundedness + 비용/토큰 |
| 모델 세대 | 현재 vs 이전 버전 | 전체 pass율 |
| Temperature | 0.0 / 0.3 / 0.7 | tone_fit + actionability 분산 |

### H-4. RAG 설정 비교 실험

```python
RAG_EXPERIMENTS = [
    {"chunk_size": 512,  "overlap": 50,  "top_k": 3, "name": "rag_v1_small"},
    {"chunk_size": 1024, "overlap": 100, "top_k": 5, "name": "rag_v2_medium"},
    {"chunk_size": 2048, "overlap": 200, "top_k": 3, "name": "rag_v3_large"},
]

# 판정 지표: faithfulness + hallucination_risk + relevance
# 트레이드오프: chunk_size 크면 faithfulness↑ but relevance↓ (너무 많은 정보 포함)
```

### H-5. 실패 사례 재평가 루프

```
1. 매주 FAIL 사례를 LangSmith에서 필터링
2. 실패 유형 taxonomy에 따라 태그 부착
3. 태그별 빈도 집계 → 가장 많은 실패 유형 우선 개선
4. 개선된 프롬프트로 해당 실패 사례만 재실험
5. 개선이 확인되면 전체 dataset으로 확대 실험
```

```python
# 실패 사례 필터링 쿼리 (LangSmith SDK)
failed_runs = client.list_runs(
    project_name="catcher_llm_eval",
    filter='and(eq(feedback_key, "pass"), eq(feedback_score, 0))',
    limit=100
)

for run in failed_runs:
    # 실패 유형 태그 부착
    client.create_feedback(
        run_id=run.id,
        key="failure_type",
        value=classify_failure(run)
    )
```

### H-6. 기간별 성능 모니터링

```python
# 월간 성능 추이 추적 권장 지표
TRACKING_METRICS = {
    "daily":   ["groundedness_avg", "actionability_avg", "pass_rate"],
    "weekly":  ["groundedness_avg", "pattern_detection_avg", "pass_rate"],
    "monthly": ["groundedness_avg", "strategy_depth_avg", "pass_rate"],
    "action":  ["action_specificity_avg", "personalization_avg", "pass_rate"],
    "rag":     ["faithfulness_avg", "hallucination_risk_avg", "pass_rate"]
}

# 알람 기준: 전주 대비 pass_rate -5%p 이상 하락 시 Slack 알림
```

### H-7. Dataset 확장 전략

초기 dataset은 **인간이 설계한 시나리오** 기반으로 시작하고, 이후 **실제 서비스 로그**에서 엣지 케이스를 추가하는 방식으로 운영하길 권장한다.

```
초기 (Month 1): 인간 설계 시나리오 × 20개/태스크
Month 2: 서비스 로그 기반 엣지 케이스 추가 × 10개/태스크
Month 3~: 실패 사례 자동 수집 → 레이블링 → dataset 편입 루프
```

엣지 케이스 우선 수집 대상은 다음과 같다. 소비 데이터가 극단적으로 많거나 적은 경우, 카테고리가 1~2개에 집중된 경우, 소득 대비 소비 비율이 90% 이상인 경우, 정책 소득 기준 경계에 있는 사용자의 RAG 요청이 이에 해당한다.

---

*이 문서는 Catcher LLM v1.0 기준으로 작성되었습니다. LangSmith 실험 결과에 따라 pass/fail 기준 및 evaluator 가중치를 주기적으로 업데이트하세요.*
