from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "CompetitionCreateInput": (
        "catcher_llm.services.group_competition_service",
        "CompetitionCreateInput",
    ),
    "CompetitionCreateResult": (
        "catcher_llm.services.group_competition_service",
        "CompetitionCreateResult",
    ),
    "GroupCreateInput": ("catcher_llm.services.group_competition_service", "GroupCreateInput"),
    "GroupCreateResult": ("catcher_llm.services.group_competition_service", "GroupCreateResult"),
    "MerchantInference": (
        "catcher_llm.services.transaction_upload_service",
        "MerchantInference",
    ),
    "MerchantInferenceBatch": (
        "catcher_llm.services.transaction_upload_service",
        "MerchantInferenceBatch",
    ),
    "TransactionShareInput": (
        "catcher_llm.services.group_competition_service",
        "TransactionShareInput",
    ),
    "TransactionShareResult": (
        "catcher_llm.services.group_competition_service",
        "TransactionShareResult",
    ),
    "TransactionUploadResult": (
        "catcher_llm.services.transaction_upload_service",
        "TransactionUploadResult",
    ),
    "UserRegistrationInput": (
        "catcher_llm.services.user_data_service",
        "UserRegistrationInput",
    ),
    "UserRegistrationResult": (
        "catcher_llm.services.user_data_service",
        "UserRegistrationResult",
    ),
    "authenticate_user": ("catcher_llm.services.user_data_service", "authenticate_user"),
    "build_daily_consumption_analysis_json": (
        "catcher_llm.services.consumption_feedback.daily_analysis",
        "build_daily_consumption_analysis_json",
    ),
    "create_competition": ("catcher_llm.services.group_competition_service", "create_competition"),
    "create_group": ("catcher_llm.services.group_competition_service", "create_group"),
    "discover_source_files": ("catcher_llm.services.ingestion_service", "discover_source_files"),
    "ensure_user_database": ("catcher_llm.services.user_data_service", "ensure_user_database"),
    "extract_spending_indicators": (
        "catcher_llm.services.consumption_feedback.interpretation",
        "extract_spending_indicators",
    ),
    "generate_catcher_consumption_benchmark_rag_reply": (
        "catcher_llm.services.rag",
        "generate_catcher_consumption_benchmark_rag_reply",
    ),
    "generate_daily_feedback": (
        "catcher_llm.services.consumption_feedback.daily_feedback",
        "generate_daily_feedback",
    ),
    "generate_kca_report_rag_reply": (
        "catcher_llm.services.rag",
        "generate_kca_report_rag_reply",
    ),
    "generate_rag_reply": ("catcher_llm.services.rag_service", "generate_rag_reply"),
    "generate_reply": ("catcher_llm.services.chat_service", "generate_reply"),
    "generate_saving_tips_rag_reply": (
        "catcher_llm.services.rag",
        "generate_saving_tips_rag_reply",
    ),
    "generate_welfare_rag_reply": (
        "catcher_llm.services.rag",
        "generate_welfare_rag_reply",
    ),
    "get_group_feed": ("catcher_llm.services.group_competition_service", "get_group_feed"),
    "get_group_leaderboard": (
        "catcher_llm.services.group_competition_service",
        "get_group_leaderboard",
    ),
    "get_group_leaderboard_for_group": (
        "catcher_llm.services.group_competition_service",
        "get_group_leaderboard_for_group",
    ),
    "get_group_member_feedback_status": (
        "catcher_llm.services.group_competition_service",
        "get_group_member_feedback_status",
    ),
    "get_user_registration_columns": (
        "catcher_llm.services.user_data_service",
        "get_user_registration_columns",
    ),
    "get_user_transactions": ("catcher_llm.services.user_data_service", "get_user_transactions"),
    "infer_merchant_transaction_fields": (
        "catcher_llm.services.transaction_upload_service",
        "infer_merchant_transaction_fields",
    ),
    "ingest_local_documents": ("catcher_llm.services.ingestion_service", "ingest_local_documents"),
    "invoke_retriever_question": ("catcher_llm.services.test_service", "invoke_retriever_question"),
    "list_group_competitions": (
        "catcher_llm.services.group_competition_service",
        "list_group_competitions",
    ),
    "list_user_groups": ("catcher_llm.services.group_competition_service", "list_user_groups"),
    "list_user_memories": ("catcher_llm.services.user_data_service", "list_user_memories"),
    "load_user_spending_data": (
        "catcher_llm.services.consumption_feedback.interpretation",
        "load_user_spending_data",
    ),
    "make_spending_analysis_input": (
        "catcher_llm.services.consumption_feedback.interpretation",
        "make_spending_analysis_input",
    ),
    "parse_user_spending_data": (
        "catcher_llm.services.consumption_feedback.interpretation",
        "parse_user_spending_data",
    ),
    "rag_target": ("catcher_llm.services.rag_service", "rag_target"),
    "register_user": ("catcher_llm.services.user_data_service", "register_user"),
    "resolve_transaction_upload_column_mapping": (
        "catcher_llm.services.transaction_upload_service",
        "resolve_transaction_upload_column_mapping",
    ),
    "save_user_memory": ("catcher_llm.services.user_data_service", "save_user_memory"),
    "share_transaction_to_group": (
        "catcher_llm.services.group_competition_service",
        "share_transaction_to_group",
    ),
    "upload_transactions_dataframe": (
        "catcher_llm.services.transaction_upload_service",
        "upload_transactions_dataframe",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    """요청된 서비스 심볼을 실제 접근 시점에 지연 로딩한다."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    """패키지 자동완성과 탐색을 위해 공개 심볼 목록을 반환한다."""
    return sorted(globals().keys() | _EXPORTS.keys())
