"""
PF physics loss validation — thin runner for tests/test_pf_physics_loss.py.

Run:
    python _tmp_pf_physics_validate.py
    python -m pytest tests/test_pf_physics_loss.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def main() -> int:
    try:
        import pytest
    except ImportError:
        print("pytest is required: pip install pytest", file=sys.stderr)
        return 1

    print("=" * 72)
    print("PF physics loss validation (pytest suite)")
    print("=" * 72)
    args = [str(REPO / "tests" / "test_pf_physics_loss.py"), "-v", "--tb=short"]
    return int(pytest.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
