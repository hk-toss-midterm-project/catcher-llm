from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def main() -> int:
    os.environ.setdefault("LANGSMITH_TRACING", "false")

    if sys.version_info < (3, 12) and VENV_PYTHON.exists():
        env = os.environ.copy()
        if env.get("CATCHER_HELLO_WORLD_REEXEC") != "1":
            env["CATCHER_HELLO_WORLD_REEXEC"] = "1"
            completed = subprocess.run(
                [str(VENV_PYTHON), str(Path(__file__).resolve())],
                check=False,
                cwd=PROJECT_ROOT,
                env=env,
            )
            return completed.returncode

    sys.path.insert(0, str(SRC_DIR))

    from catcher_llm.services.helloworld import build_hello_world_chain

    result = build_hello_world_chain()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
