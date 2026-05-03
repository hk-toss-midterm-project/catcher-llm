# Catcher LLM

Streamlit UI와 LangChain 애플리케이션 로직을 분리해서, 화면 계층은 얇게 유지하고 핵심 워크플로우는 `src/catcher_llm`에 모아둔 프로젝트다.

## 프로젝트 구조

```text
.
├── app.py                         # Streamlit 진입점
├── pages/                         # Streamlit 멀티 페이지 화면
│   ├── 01_chat.py                 # 채팅 UI
│   ├── 02_docs.py                 # 로컬 문서 적재 UI
│   └── 03_admin.py                # 런타임/설정 점검 화면
├── src/catcher_llm/
│   ├── chains/                    # 라우팅, 채팅, 요약, RAG 체인
│   ├── config/                    # 앱 설정과 로깅
│   ├── evaluation/                # LangSmith 데이터셋/평가 헬퍼
│   ├── llm/                       # 모델 및 콜백 헬퍼
│   ├── prompts/                   # 프롬프트 템플릿
│   ├── retrievers/                # 로더와 벡터스토어 검색
│   ├── schemas/                   # 타입 기반 스키마
│   ├── services/                  # 채팅, RAG, 적재 서비스 계층
│   ├── ui/                        # 공통 Streamlit 컴포넌트와 세션 상태
│   └── utils/                     # 소규모 유틸리티
├── scripts/
│   ├── ingest_docs.py             # CLI 문서 적재
│   ├── eval_chain.py              # CLI 채팅 체인 스모크 테스트
│   ├── create_rag_eval_dataset.py # LangSmith 데이터셋 생성
│   └── run_rag_eval.py            # LangSmith RAG 평가 실행
├── tests/                         # 단위 테스트
├── data/
│   ├── raw/                       # 적재 대상 원본 문서
│   ├── processed/                 # 매니페스트 등 파생 산출물
│   ├── sqlite/                    # 로컬 앱/세션 SQLite 파일
│   ├── vectordb/                  # 로컬 벡터스토어 파일
│   └── evals/                     # 평가 예제 데이터
├── pyproject.toml                 # 의존성과 ruff 설정
├── AGENTS.md                      # 협업 및 바이브코딩 규칙
└── .env.example                   # 로컬 환경 변수 템플릿
```

## 주요 흐름

- `app.py`는 앱 셸, 로깅, 사이드바를 초기화한다.
- `pages/01_chat.py`는 채팅 입력을 처리하고 chat service로 요청을 전달한다.
- `pages/02_docs.py`는 `data/raw`의 파일을 탐색하고 로컬 적재를 실행한다.
- `pages/03_admin.py`는 런타임 설정과 로컬 초기화 동작을 보여준다.
- `src/catcher_llm/services/`는 UI와 CLI 스크립트가 함께 사용하는 서비스 계층이다.
- `src/catcher_llm/evaluation/`는 LangSmith 데이터셋 로딩, 평가기, 평가 실행 로직을 포함한다.

## 실행

```bash
uv sync
uv run streamlit run app.py
```

Streamlit 앱을 실행한 뒤 `Chat` 페이지부터 시작하면 된다. 로컬 원본 문서를 확인하고 적재하려면 `Docs` 페이지를 사용한다.

## 환경 변수

`.env.example`을 `.env`로 복사한 뒤 필요한 값을 설정한다.

실제 모델 호출에 필수:

- OpenAI 채팅 또는 OpenAI 임베딩을 쓰는 경우 `OPENAI_API_KEY`
- Claude/Anthropic 채팅을 쓰는 경우 `ANTHROPIC_API_KEY`
- Ollama를 쓰는 경우 로컬 Ollama 서버와 필요한 모델 pull

자주 조정하는 선택 옵션:

- `LLM_PROVIDER`: `openai`, `anthropic` 또는 `ollama` (`claude`, `local` 별칭 지원)
- `OPENAI_MODEL`
- `ANTHROPIC_MODEL`
- `OLLAMA_MODEL`
- `OLLAMA_BASE_URL`
- `EMBEDDING_PROVIDER`: `upstage`, `openai` 또는 `ollama` (`solar`, `local` 별칭 지원)
- `OPENAI_EMBEDDING_MODEL`
- `OLLAMA_EMBEDDING_MODEL`
- `UPSTAGE_API_KEY`
- `UPSTAGE_EMBEDDING_MODEL`

모델 temperature는 전역 환경변수 대신 `generate_reply`, `generate_rag_reply`, `build_*_chain` 호출 옵션으로 기능별로 지정한다.

RAG의 `chunk_size`, `chunk_overlap`, `top_k`는 전역 환경변수보다 코퍼스/호출 단위 인자로 전달하는 방식을 기본으로 사용한다.

LangSmith 평가 관련 설정:

- `LANGSMITH_API_KEY`
- `LANGSMITH_ENDPOINT`
- `LANGSMITH_PROJECT`
- `LANGSMITH_DATASET`
- `LANGSMITH_EXPERIMENT_PREFIX`
- `LANGSMITH_TRACING`

## RAG 작업 흐름

1. 원본 문서를 `data/raw`에 넣는다.
2. `Docs` 페이지에서 적재를 실행하거나 CLI를 사용한다.

```bash
uv run python scripts/ingest_docs.py
```

1. `Chat` 페이지에서 질문한다. 프롬프트에 `docs`, `document`, `문서`, `검색`, `rag`가 포함되면 검색 기반 흐름으로 라우팅된다.

## 평가

기본 평가 데이터셋은 `data/evals/rag_examples.json`에 있다.

```bash
uv run python scripts/create_rag_eval_dataset.py
uv run python scripts/run_rag_eval.py
```

CLI에서 채팅 체인을 간단히 스모크 테스트할 수도 있다.

```bash
uv run python scripts/eval_chain.py "이 프로젝트 구조를 짧게 설명해줘."
```

## 개발

린트와 테스트 도구가 필요하면 dev 의존성 그룹까지 함께 설치한다.

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format .

uv run streamlit run dev_app.py
```

이 저장소의 작업 규칙은 `AGENTS.md`에 정리돼 있다.
