# Copyright (c) ModelScope Contributors. All rights reserved.
"""Schema endpoint must stay cached and not re-ship pruned / duplicate trees."""
import gzip
import json
import threading

from starlette.testclient import TestClient

from rsi.backends import cached_gpu, clear_gpu_cache
from rsi.schemas import clear_public_schema_cache, public_schema, public_schema_wire

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


def test_cached_gpu_ttl_and_stale_on_empty(monkeypatch):
    clear_gpu_cache()
    calls = {'n': 0}
    good = {'available': True, 'devices': ['GPU 0: Fake']}
    empty = {'available': False, 'devices': []}

    def fake_detect():
        calls['n'] += 1
        return good if calls['n'] == 1 else empty

    monkeypatch.setattr('rsi.backends.detect_gpu', fake_detect)
    first = cached_gpu(ttl=60.0)
    second = cached_gpu(ttl=60.0)
    assert first == good
    assert second is first
    assert calls['n'] == 1

    # Expire TTL, then a failed probe must keep the last good devices.
    import rsi.backends as backends
    backends._gpu_cache['at'] = float('-inf')
    stale = cached_gpu(ttl=60.0)
    assert stale == good
    assert calls['n'] == 2


def test_public_schema_wire_is_stable_and_gzipped():
    clear_public_schema_cache()
    raw, gzipped, etag = public_schema_wire()
    again = public_schema_wire()
    assert again == (raw, gzipped, etag)
    assert etag.startswith('"') and etag.endswith('"')
    assert gzip.decompress(gzipped) == raw
    assert json.loads(raw.decode()) is not None
    assert len(raw) < _SCHEMA_JSON_CEILING


def test_schema_endpoint_etag_and_304(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    monkeypatch.setenv('RSI_HOME', str(tmp_path / 'home'))
    from rsi.app import create_app

    with TestClient(create_app(tmp_path / 'home')) as api:
        first = api.get('/api/schema', headers={'Accept-Encoding': 'identity'})
        assert first.status_code == 200
        etag = first.headers.get('etag')
        assert etag
        # We set Vary; GZipMiddleware may append the same token.
        assert 'Accept-Encoding' in (first.headers.get('vary') or '')
        body = first.json()
        assert 'stages' in body

        gzipped = api.get('/api/schema', headers={'Accept-Encoding': 'gzip'})
        assert gzipped.status_code == 200
        assert gzipped.headers.get('content-encoding') == 'gzip'
        assert gzipped.headers.get('etag') == etag
        assert gzipped.json() == body

        cached = api.get('/api/schema', headers={'If-None-Match': etag, 'Accept-Encoding': 'gzip'})
        assert cached.status_code == 304
        assert cached.headers.get('etag') == etag
        assert cached.content == b''


def test_health_uses_presets_not_schema(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    monkeypatch.setenv('RSI_HOME', str(tmp_path / 'home'))
    from rsi.app import create_app
    from rsi.schemas import presets

    calls = {'n': 0}

    def boom():
        calls['n'] += 1
        raise AssertionError('health must not build public_schema')

    monkeypatch.setattr('rsi.schemas.public_schema', boom)
    monkeypatch.setattr('rsi.schemas.public_schema_wire', boom)
    expected = (presets().get('hardware') or {})

    with TestClient(create_app(tmp_path / 'home')) as api:
        health = api.get('/api/health')
    assert health.status_code == 200
    assert health.json()['hardware'] == expected
    assert 'A100' in health.json()['hardware']['gpu']
    assert calls['n'] == 0


def test_schema_build_is_single_flight(monkeypatch):
    clear_public_schema_cache()
    builds = {'n': 0}
    started = threading.Event()
    release = threading.Event()

    def slow_build(device_count):
        builds['n'] += 1
        started.set()
        assert release.wait(timeout=5)
        return {
            'stages': [],
            'stage_order': [],
            'fields': {},
            'stage_layouts': {},
            'presets': {},
            'public_datasets': [],
            'bundled': [],
            'hardware': {},
            'base_model_default': 'x',
            'model_source': 'modelscope',
            'llm_infer': {},
            'llm_export': {},
            'catalog': {},
        }

    monkeypatch.setattr('rsi.schemas._build_public_schema', slow_build)
    monkeypatch.setattr('rsi.schemas.cached_gpu', lambda ttl=30.0: {'available': False, 'devices': []})
    monkeypatch.setattr('rsi.schemas.model_source', lambda: 'modelscope')

    results = []

    def worker():
        results.append(public_schema())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    assert started.wait(timeout=5)
    # Other callers should be blocked on the lock, not spawning extra builds.
    threading.Event().wait(0.05)
    assert builds['n'] == 1
    release.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert builds['n'] == 1
    assert len(results) == 4
    assert all(item is results[0] for item in results)
