from __future__ import annotations

from catcher_llm.analysis.user_daily_analysis import build_daily_consumption_analysis_from_frames
from catcher_llm.analysis.user_monthly_analysis import (
    build_monthly_consumption_analysis_from_frames,
)
from catcher_llm.analysis.user_weekly_analysis import build_weekly_consumption_analysis_from_frames

__all__ = [
    "build_daily_consumption_analysis_from_frames",
    "build_weekly_consumption_analysis_from_frames",
    "build_monthly_consumption_analysis_from_frames",
]
