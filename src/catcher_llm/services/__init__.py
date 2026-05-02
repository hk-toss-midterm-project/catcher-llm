from catcher_llm.services.chat_service import generate_reply
from catcher_llm.services.consumption_feedback.daily_analysis import (
    build_daily_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.daily_feedback import (
    generate_daily_feedback,
)
from catcher_llm.services.consumption_feedback.interpretation import (
    extract_spending_indicators,
    load_user_spending_data,
    make_spending_analysis_input,
    parse_user_spending_data,
)
from catcher_llm.services.group_competition_service import (
    CompetitionCreateInput,
    CompetitionCreateResult,
    GroupCreateInput,
    GroupCreateResult,
    TransactionShareInput,
    TransactionShareResult,
    create_competition,
    create_group,
    get_group_feed,
    get_group_leaderboard,
    list_group_competitions,
    list_user_groups,
    share_transaction_to_group,
)
from catcher_llm.services.ingestion_service import discover_source_files, ingest_local_documents
from catcher_llm.services.rag import (
    generate_catcher_consumption_benchmark_rag_reply,
    generate_kca_report_rag_reply,
    generate_saving_tips_rag_reply,
    generate_welfare_rag_reply,
)
from catcher_llm.services.rag_service import generate_rag_reply, rag_target
from catcher_llm.services.test_service import invoke_retriever_question
from catcher_llm.services.transaction_upload_service import (
    MerchantInference,
    MerchantInferenceBatch,
    TransactionUploadResult,
    infer_merchant_transaction_fields,
    resolve_transaction_upload_column_mapping,
    upload_transactions_dataframe,
)
from catcher_llm.services.user_data_service import (
    UserRegistrationInput,
    UserRegistrationResult,
    authenticate_user,
    ensure_user_database,
    get_user_registration_columns,
    get_user_transactions,
    list_user_memories,
    register_user,
    save_user_memory,
)

__all__ = [
    "CompetitionCreateInput",
    "CompetitionCreateResult",
    "GroupCreateInput",
    "GroupCreateResult",
    "MerchantInference",
    "MerchantInferenceBatch",
    "TransactionShareInput",
    "TransactionShareResult",
    "TransactionUploadResult",
    "UserRegistrationInput",
    "UserRegistrationResult",
    "authenticate_user",
    "build_daily_consumption_analysis_json",
    "create_competition",
    "create_group",
    "discover_source_files",
    "ensure_user_database",
    "extract_spending_indicators",
    "generate_catcher_consumption_benchmark_rag_reply",
    "generate_daily_feedback",
    "generate_kca_report_rag_reply",
    "generate_rag_reply",
    "generate_reply",
    "generate_saving_tips_rag_reply",
    "generate_welfare_rag_reply",
    "get_group_feed",
    "get_group_leaderboard",
    "list_group_competitions",
    "list_user_groups",
    "get_user_registration_columns",
    "get_user_transactions",
    "infer_merchant_transaction_fields",
    "ingest_local_documents",
    "invoke_retriever_question",
    "list_user_memories",
    "load_user_spending_data",
    "make_spending_analysis_input",
    "parse_user_spending_data",
    "rag_target",
    "register_user",
    "resolve_transaction_upload_column_mapping",
    "save_user_memory",
    "share_transaction_to_group",
    "upload_transactions_dataframe",
]
