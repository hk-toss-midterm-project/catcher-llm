from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "dev_pages/16_user_trend_report.py"),
    run_name="__main__",
)
