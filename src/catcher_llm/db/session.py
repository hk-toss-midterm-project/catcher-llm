from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.db.models import Base

_ENGINE_CACHE: dict[str, Engine] = {}


def get_engine(settings: Settings | None = None) -> Engine:
    """설정된 SQLite 경로에 맞는 SQLAlchemy 엔진을 생성하거나 캐시에서 재사용한다."""
    config = settings or get_settings()
    db_path = config.sqlite_db_path.resolve()
    cache_key = str(db_path)
    engine = _ENGINE_CACHE.get(cache_key)
    if engine is None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        _ENGINE_CACHE[cache_key] = engine
    return engine


def dispose_engine(settings: Settings | None = None) -> None:
    """설정된 SQLite 경로에 연결된 엔진을 캐시에서 제거하고 종료한다."""
    config = settings or get_settings()
    cache_key = str(config.sqlite_db_path.resolve())
    engine = _ENGINE_CACHE.pop(cache_key, None)
    if engine is not None:
        engine.dispose()


def create_database_tables(settings: Settings | None = None) -> None:
    """등록된 SQLAlchemy 모델 기준으로 필요한 데이터베이스 테이블을 생성한다."""
    Base.metadata.create_all(get_engine(settings))


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """세션을 열고 성공 시 커밋, 실패 시 롤백한 뒤 안전하게 종료한다."""
    session = Session(get_engine(settings), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
