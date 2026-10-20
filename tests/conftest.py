import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic import make_demo  # noqa: E402


@pytest.fixture(scope="session")
def demo():
    return make_demo()
