# ============================================================
# Catcher LLM — LangSmith 작업 단위별 Git Commit 스크립트
# PowerShell에서 실행: .\git_commits_langsmith.ps1
# 전제: feat/langsmith 브랜치에 있어야 함
# ============================================================

Set-Location $PSScriptRoot

# ── 사전 확인 ──────────────────────────────────────────────
Write-Host ""
Write-Host "=== 현재 브랜치 ===" -ForegroundColor Cyan
git branch --show-current

Write-Host ""
Write-Host "=== lock 파일 제거 (있으면) ===" -ForegroundColor Yellow
if (Test-Path ".git\index.lock") {
    Remove-Item ".git\index.lock" -Force
    Write-Host "index.lock 제거 완료" -ForegroundColor Green
} else {
    Write-Host "lock 파일 없음 (정상)" -ForegroundColor Green
}


# ════════════════════════════════════════════════════════════
# COMMIT 1 — 평가 체계 설계 문서
# ════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "================================================" -ForegroundColor Magenta
Write-Host "COMMIT 1: LangSmith 성능평가 체계 설계 문서" -ForegroundColor Magenta
Write-Host "================================================" -ForegroundColor Magenta

git add langsmith_eval_design.md

git commit -m "docs: LangSmith 성능평가 체계 설계 문서 추가

Catcher LLM 프로젝트의 LangSmith 기반 오프라인 평가 체계 전체 설계안.

포함 내용:
- 평가 전략 3원칙 (근거성 우선 / 기간 역할 분리 / RAG 문서 경계)
- 태스크 5종 평가 항목 (일간/주간/월간/액션/정책RAG)
- LangSmith dataset 필드 구조 및 태스크별 예시 row
- Evaluator 4종 설계 (heuristic/binary/score-based/rubric)
- 소비 피드백 judge prompt + 문서 RAG judge prompt (완성형)
- 태스크별 pass/fail 기준 (groundedness 4점 필수 등)
- 실패 유형 taxonomy F-01~F-12 + 개선 방향
- Phase별 실험 운영 권장안 (A/B테스트 / 모델비교 / RAG설정비교)"

Write-Host "✅ Commit 1 완료" -ForegroundColor Green


# ════════════════════════════════════════════════════════════
# COMMIT 2 — RAG eval dataset (data/evals)
# ════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "================================================" -ForegroundColor Magenta
Write-Host "COMMIT 2: RAG 평가 Dataset 초안" -ForegroundColor Magenta
Write-Host "================================================" -ForegroundColor Magenta

git add data/evals/rag_examples.json

git commit -m "feat(eval): RAG 평가 dataset 초안 추가 (복지 정책 QA 5개)

LangSmith catcher-llm-rag-eval 데이터셋의 로컬 소스 파일.

포함된 QA 쌍 (복지 정책 문서 기반):
- 여성청소년 생리용품 지원 월 지원금
- 임신 사전건강관리 지원사업 여성 최대 지원금
- 저소득 청소년부모 아동양육비 월 지원금
- 국·공립유치원 교육비 월 지원금
- 청년내일저축계좌 정부 매칭 한도

각 row 구조: inputs(question, chunk_size, chunk_overlap, top_k) + outputs(answer)
LangSmith evaluate()의 reference_outputs로 사용됨"

Write-Host "✅ Commit 2 완료" -ForegroundColor Green


# ════════════════════════════════════════════════════════════
# PUSH
# ════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "================================================" -ForegroundColor Magenta
Write-Host "PUSH → origin/feat/langsmith" -ForegroundColor Magenta
Write-Host "================================================" -ForegroundColor Magenta

git push origin feat/langsmith

Write-Host ""
Write-Host "✅ 전체 완료!" -ForegroundColor Green
Write-Host ""

Write-Host "=== 최근 커밋 확인 ===" -ForegroundColor Cyan
git log --oneline -5
