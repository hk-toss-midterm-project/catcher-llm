from __future__ import annotations

import os

RAW_USERS_CSV_FILENAME_ENV = "RAW_USERS_CSV_FILENAME"
RAW_TRANSACTIONS_CSV_FILENAME_ENV = "RAW_TRANSACTIONS_CSV_FILENAME"

DEFAULT_RAW_USERS_CSV_FILENAME = "users_v4.csv"
DEFAULT_RAW_TRANSACTIONS_CSV_FILENAME = "transactions_v4.csv"

RAW_USERS_CSV_FILENAME = DEFAULT_RAW_USERS_CSV_FILENAME
RAW_TRANSACTIONS_CSV_FILENAME = DEFAULT_RAW_TRANSACTIONS_CSV_FILENAME


def get_raw_users_csv_filename() -> str:
    """사용자 raw CSV 파일명을 환경변수 또는 기본값에서 읽어 반환한다."""
    return os.getenv(RAW_USERS_CSV_FILENAME_ENV, RAW_USERS_CSV_FILENAME)


def get_raw_transactions_csv_filename() -> str:
    """거래 raw CSV 파일명을 환경변수 또는 기본값에서 읽어 반환한다."""
    return os.getenv(RAW_TRANSACTIONS_CSV_FILENAME_ENV, RAW_TRANSACTIONS_CSV_FILENAME)
