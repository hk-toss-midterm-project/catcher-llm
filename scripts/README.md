# Scripts

프로젝트 루트에서 아래 명령으로 실행한다.

```bash
uv run python scripts/<script_name>.py
```

`.env`는 자동으로 로드되며, 기본 데이터 경로는 아래 설정을 따른다.

- 원본 문서: `data/raw`
- 처리 결과: `data/processed`
- 평가 예제: `data/evals`

## 필수 환경변수

- `OPENAI_API_KEY`
  - `eval_chain.py`, RAG 생성, LangSmith 평가 실행에 필요하다.
- `LANGSMITH_API_KEY`
  - `create_rag_eval_dataset.py`, `run_rag_eval.py`에 필요하다.

선택적으로 아래 환경변수를 조정할 수 있다.

- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `LANGSMITH_DATASET`
- `LANGSMITH_EXPERIMENT_PREFIX`

## 스크립트 목록

### `ingest_docs.py`

`data/raw` 아래의 지원 문서를 전부 읽고 청킹한 뒤, 적재 매니페스트를 생성한다.

```bash
uv run python scripts/ingest_docs.py
```

출력 항목:

- 적재한 파일 수
- 생성된 청크 수
- 매니페스트 경로

### `ingest_source_doc.py`

지정한 문서 하나만 읽고 청킹한 뒤, 파일별 적재 매니페스트를 생성한다.

```bash
uv run python scripts/ingest_source_doc.py data/raw/pdf/saving_tips/4.pdf
```

인자:

- `source`: 적재할 문서 경로

출력 항목:

- 적재한 파일 수
- 생성된 청크 수
- 매니페스트 경로

### `eval_chain.py`

프롬프트를 한 번 실행해서 현재 라우팅 결과와 응답을 확인한다.

```bash
uv run python scripts/eval_chain.py
uv run python scripts/eval_chain.py "문서 검색해서 알려줘"
```

출력 항목:

- 선택된 라우트(`chat`, `rag`, `summary`)
- 생성된 응답 본문

### `create_rag_eval_dataset.py`

LangSmith에서 RAG 평가용 데이터셋을 조회하거나 생성하고, 비어 있으면 예제를 업로드한다.

기본 예제 파일은 `data/evals/rag_examples.json`이다.

```bash
uv run python scripts/create_rag_eval_dataset.py
uv run python scripts/create_rag_eval_dataset.py --dataset-name my-rag-dataset
uv run python scripts/create_rag_eval_dataset.py --examples-path data/evals/rag_examples.json
```

옵션:

- `--dataset-name`: 사용할 LangSmith 데이터셋 이름
- `--examples-path`: 업로드할 평가 예제 JSON 경로

### `run_rag_eval.py`

LangSmith에 RAG 평가 실행을 제출한다.

```bash
uv run python scripts/run_rag_eval.py
uv run python scripts/run_rag_eval.py --dataset-name my-rag-dataset
uv run python scripts/run_rag_eval.py --experiment-prefix local-test
uv run python scripts/run_rag_eval.py --max-concurrency 2
```

옵션:

- `--dataset-name`: 사용할 데이터셋 이름
- `--examples-path`: 평가 예제 JSON 경로
- `--experiment-prefix`: LangSmith 실험 이름 접두사
- `--max-concurrency`: 평가 동시 실행 수

## 참고

- 현재 `ingest_docs.py`와 `ingest_source_doc.py`는 문서 로드, 청킹, 매니페스트 저장까지 수행한다.
- 벡터스토어를 파일로 저장하는 스크립트는 아직 없다.
- 실제 `FAISS` 생성은 애플리케이션에서 `build_local_vectorstore()` 또는 RAG 서비스가 호출될 때 이뤄진다.
