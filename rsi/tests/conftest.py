# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _modelscope_source_unless_overridden(monkeypatch):
    """fetch.yaml defaults to hope. Closed-loop tests keep hub ids unless they opt into hope."""
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'modelscope')
