"""
session 테이블에 feedback_message 컬럼을 추가하는 마이그레이션 스크립트.

실행 방법:
    python migrate_add_feedback_message.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# 프로젝트 루트 기준으로 DB 경로 자동 탐색
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DB_PATH = _SCRIPT_DIR / "data" / "sqlite" / "app.sqlite3"


def migrate(db_path: Path = _DEFAULT_DB_PATH) -> None:
    if not db_path.exists():
        print(f"[ERROR] DB 파일을 찾을 수 없습니다: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # 현재 컬럼 목록 확인
    cur.execute("PRAGMA table_info(session)")
    existing_cols = {row[1] for row in cur.fetchall()}
    print(f"현재 session 테이블 컬럼: {sorted(existing_cols)}")

    if "feedback_message" in existing_cols:
        print("feedback_message 컬럼이 이미 존재합니다. 마이그레이션을 건너뜁니다.")
    else:
        cur.execute("ALTER TABLE session ADD COLUMN feedback_message TEXT")
        conn.commit()
        print("✅ feedback_message 컬럼 추가 완료")

    # 결과 확인
    cur.execute("PRAGMA table_info(session)")
    final_cols = [row[1] for row in cur.fetchall()]
    print(f"마이그레이션 후 컬럼: {final_cols}")

    conn.close()


if __name__ == "__main__":
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_DB_PATH
    print(f"대상 DB: {db_path}")
    migrate(db_path)
