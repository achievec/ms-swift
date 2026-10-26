# Copyright (c) ModelScope Contributors. All rights reserved.
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rsi.metrics import (
    GRPO_PLOT,
    SFT_PLOT,
    _downsample,
    find_event_file,
    series_for_stage,
)
from rsi.store import Store


def test_plot_specs_match_webui():
    assert [p['name'] for p in GRPO_PLOT] == [
        'train/loss',
        'train/reward',
        'train/learning_rate',
        'train/completions/mean_length',
        'train/kl',
    ]
    assert [p['name'] for p in SFT_PLOT] == [
        'train/loss',
        'train/acc',
        'train/learning_rate',
        'eval/loss',
        'eval/acc',
    ]


def test_downsample_keeps_ends():
    steps = list(range(1000))
    values = [float(i) for i in steps]
    out_s, out_v = _downsample(steps, values, limit=10)
    assert out_s[0] == 0
    assert out_s[-1] == 999
    assert len(out_s) <= 10
    assert out_v[0] == 0.0
    assert out_v[-1] == 999.0


def test_find_event_file_prefers_runs_subdir(tmp_path):
    root = tmp_path / 'out'
    runs = root / 'runs'
    runs.mkdir(parents=True)
    event = runs / 'events.out.tfevents.test'
    event.write_bytes(b'not-a-real-event')
    found = find_event_file([root])
    assert found == event.resolve()


def test_series_for_stage_empty_placeholders():
    series = series_for_stage('rl', data={})
    assert [item['name'] for item in series] == [p['name'] for p in GRPO_PLOT]
    assert all(item['steps'] == [] and item['values'] == [] for item in series)


def test_series_for_stage_reads_scalars_and_smooth():
    data = {
        'train/loss': [{'step': 1, 'value': 1.0}, {'step': 2, 'value': 0.5}, {'step': 3, 'value': 0.25}],
        'train/reward': [{'step': 1, 'value': 0.1}],
        'train/token_acc': [{'step': 1, 'value': 0.9}],
    }
    rl = series_for_stage('rl', data=data)
    by_name = {item['name']: item for item in rl}
    assert by_name['train/loss']['steps'] == [1, 2, 3]
    assert by_name['train/loss']['values'] == [1.0, 0.5, 0.25]
    assert 'smoothed' in by_name['train/loss']
    assert len(by_name['train/loss']['smoothed']) == 3
    assert by_name['train/reward']['values'] == [0.1]
    assert by_name['train/learning_rate']['values'] == []
    assert by_name['train/kl']['values'] == []

    sft = series_for_stage('sft', data=data)
    sft_by = {item['name']: item for item in sft}
    assert sft_by['train/acc']['tag'] == 'train/token_acc'
    assert sft_by['train/acc']['values'] == [0.9]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    home = tmp_path / 'rsi-home'
    app = create_app(home)
    with TestClient(app) as test_client:
        yield test_client, home


def test_metrics_endpoint_returns_grpo_placeholders(client):
    api, home = client
    created = api.post('/api/runs', json={
        'name': 'metrics',
        'scenario': 'tb charts',
        'base_model': 'Qwen/Qwen2.5-0.5B-Instruct',
    })
    assert created.status_code == 200
    run_id = created.json()['run']['id']
    # Advance to RL so a step exists with output_dir under the run.
    store = Store(home / 'rsi.sqlite')
    with store.transaction() as conn:
        store.insert_step(conn, run_id, 1, 'rl', {
            'output_dir': str(home / 'runs' / run_id / 'round_1' / 'rl'),
        }, 'swift')
    out = home / 'runs' / run_id / 'round_1' / 'rl' / 'runs'
    out.mkdir(parents=True)
    (out / 'events.out.tfevents.fake').write_bytes(b'')

    resp = api.get(f'/api/runs/{run_id}/stages/rl/metrics')
    assert resp.status_code == 200
    body = resp.json()
    assert body['stage'] == 'rl'
    assert [item['name'] for item in body['series']] == [p['name'] for p in GRPO_PLOT]
    # Fake event file is unreadable; still return empty series, not 500.
    assert all(item['values'] == [] for item in body['series'])


def test_metrics_endpoint_rejects_unknown_stage(client):
    api, _home = client
    created = api.post('/api/runs', json={
        'name': 'metrics-sft-eval',
        'scenario': 'no charts',
        'base_model': 'Qwen/Qwen2.5-0.5B-Instruct',
    })
    run_id = created.json()['run']['id']
    assert api.get(f'/api/runs/{run_id}/stages/sft_eval/metrics').status_code == 400
