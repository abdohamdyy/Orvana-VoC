import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orvana.config import settings  # noqa: E402
from orvana.schema import Category  # noqa: E402


class DummyBackend:
    """Deterministic stand-in for a classifier so pipeline/report tests need no model."""
    name = "dummy"

    def classify(self, segments):
        return [(Category.inquiry if "؟" in s or "?" in s else Category.delivery_delay, 0.99) for s in segments]


@pytest.fixture
def dummy_backend():
    return DummyBackend()


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Point state/outputs/incoming at a temp dir so tests never touch real files."""
    for name in ("state_dir", "outputs_dir", "incoming_dir"):
        p = tmp_path / name
        p.mkdir()
        monkeypatch.setattr(settings, name, p)
    monkeypatch.setattr(settings, "file_stable_seconds", 0)
    return settings
