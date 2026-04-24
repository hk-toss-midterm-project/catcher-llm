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


def create_database_tables(settings: Settings | None = None) -> None:
    Base.metadata.create_all(get_engine(settings))


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    session = Session(get_engine(settings), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
