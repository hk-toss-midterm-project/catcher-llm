from __future__ import annotations

from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from catcher_llm.chains.chat_chain import build_session_chat_chain
from catcher_llm.config.settings import Settings
from catcher_llm.services.session_history_service import (
    build_user_session_id,
    clear_session_chat_messages,
    get_sqlite_chat_message_history,
    load_session_chat_messages,
)


def _make_settings(tmp_path: Path) -> Settings:
    """세션 SQLite 파일이 임시 디렉터리에 생성되도록 테스트 설정을 만든다."""
    data_dir = tmp_path / "data"
    return Settings(
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        processed_data_dir=data_dir / "processed",
        vectorstore_dir=data_dir / "vectordb",
        eval_data_dir=data_dir / "evals",
        sqlite_db_path=data_dir / "sqlite" / "app.sqlite3",
    )


def test_build_user_session_id_includes_logged_in_user_id() -> None:
    """로그인한 사용자 ID가 LangChain 세션 ID에 안정적으로 반영되는지 검증한다."""
    assert build_user_session_id(7) == "user:7:chat"
    assert build_user_session_id(7, scope="report") == "user:7:report"


def test_sqlite_chat_message_history_persists_messages(tmp_path: Path) -> None:
    """LangChain 메시지 히스토리가 SQLite 파일에 저장되고 다시 조회되는지 검증한다."""
    settings = _make_settings(tmp_path)
    session_id = build_user_session_id(1)

    history = get_sqlite_chat_message_history(session_id, settings=settings)
    history.add_user_message("안녕")
    history.add_ai_message("반가워요")

    restored_messages = load_session_chat_messages(session_id, settings=settings)

    assert settings.session_sqlite_db_path.exists()
    assert [message.role for message in restored_messages] == ["user", "assistant"]
    assert [message.content for message in restored_messages] == ["안녕", "반가워요"]


def test_clear_session_chat_messages_removes_persisted_messages(tmp_path: Path) -> None:
    """세션 초기화 요청이 SQLite에 저장된 대화 메시지를 삭제하는지 검증한다."""
    settings = _make_settings(tmp_path)
    session_id = build_user_session_id(1)
    history = get_sqlite_chat_message_history(session_id, settings=settings)
    history.add_user_message("초기화 전 질문")

    clear_session_chat_messages(session_id, settings=settings)

    assert load_session_chat_messages(session_id, settings=settings) == []


def test_build_session_chat_chain_writes_turns_to_sqlite(tmp_path: Path) -> None:
    """세션 채팅 체인 호출이 사용자 입력과 AI 응답을 SQLite 히스토리에 저장하는지 검증한다."""
    settings = _make_settings(tmp_path)
    session_id = build_user_session_id(2)
    chain = build_session_chat_chain(
        settings=settings,
        llm=FakeListChatModel(responses=["첫 응답"]),
    )

    result = chain.invoke(
        {"input": "오늘 소비를 줄이는 방법은?"},
        config={"configurable": {"session_id": session_id}},
    )
    restored_messages = load_session_chat_messages(session_id, settings=settings)

    assert result == "첫 응답"
    assert [message.role for message in restored_messages] == ["user", "assistant"]
    assert restored_messages[0].content == "오늘 소비를 줄이는 방법은?"
    assert restored_messages[1].content == "첫 응답"
