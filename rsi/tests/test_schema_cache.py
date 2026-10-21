# Copyright (c) ModelScope Contributors. All rights reserved.
"""Schema endpoint must stay cached and not re-ship pruned / duplicate trees."""
import json

from rsi.schemas import clear_public_schema_cache, public_schema

# Uncompressed JSON before cache/trim was ~370KB. Keep a soft ceiling so regressions
# that re-introduce Web-UI advanced accordions or triplicate layout fields fail loudly.
_SCHEMA_JSON_CEILING = 360_000


def test_public_schema_is_cached_and_bounded():
    clear_public_schema_cache()
    first = public_schema()
    second = public_schema()
    assert first is second

    raw = json.dumps(first, ensure_ascii=False).encode()
    assert len(raw) < _SCHEMA_JSON_CEILING

    sft = first['stage_layouts']['sft']
    assert 'fields' not in sft
    for key in ('envs', 'tuner_params', 'optimizer_params', 'task_params', 'extra_params'):
        assert key not in sft['by_id']
    assert 'fields' not in first['llm_export']
    assert 'blocks' not in first['llm_export']
    assert 'sections' in first['llm_export']
    assert 'by_id' in first['llm_export']
