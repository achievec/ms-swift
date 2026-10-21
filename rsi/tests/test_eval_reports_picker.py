# Copyright (c) ModelScope Contributors. All rights reserved.
"""Flexible same-run eval report picker (GET /api/runs/{id}/eval-reports)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BASE = 'Qwen/Qwen2.5-7B-Instruct'


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    app = create_app(tmp_path / 'rsi-home')
    with TestClient(app) as test_client:
        yield test_client


def _create(api, name='自选对比'):
    return api.post('/api/runs', json={
        'name': name,
        'scenario': '同实验内挑选评测报告做对照',
        'base_model': BASE,
        'model_type': 'qwen2',
        'template': 'qwen2_5',
    }).json()['run']['id']


def _run(api, run_id, stage, params=None):
    return api.post(f'/api/runs/{run_id}/stages/{stage}/run', json={'params': params or {}})


def test_list_eval_reports_after_sft_eval(client):
    api = client
    run_id = _create(api, 'list-sft')
    assert _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200
    assert _run(api, run_id, 'sft_eval', {'eval_base': True}).status_code == 200

    listed = api.get(f'/api/runs/{run_id}/eval-reports')
    assert listed.status_code == 200, listed.text
    reports = listed.json()['reports']
    roles = [row['role'] for row in reports]
    assert 'base' in roles
    assert 'sft' in roles
    for row in reports:
        assert row['id']
        assert row['step_id']
        assert row['round_index'] == 1
        assert row['stage'] == 'sft_eval'
        assert row['label'].startswith('第1轮 · SFT评测/')
        assert row['metrics'].get('accuracy') is not None
        assert row['report_url'].startswith(f'/api/runs/{run_id}/stages/sft_eval/report?')
        assert 'step=' in row['report_url']
        assert row['has_report'] is True
        # Compact metrics only — no HTML blob.
        assert 'report_html' not in row['metrics']

    # Scoped to current round matches the unscoped list for a single-round run.
    scoped = api.get(f'/api/runs/{run_id}/eval-reports?round=1')
    assert scoped.status_code == 200
    assert len(scoped.json()['reports']) == len(reports)

    empty = api.get(f'/api/runs/{run_id}/eval-reports?round=99')
    assert empty.status_code == 200
    assert empty.json()['reports'] == []


def test_list_eval_reports_includes_rl_and_historical(client):
    api = client
    run_id = _create(api, 'list-rl')
    assert _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200
    assert _run(api, run_id, 'sft_eval', {'eval_base': True}).status_code == 200
    assert _run(api, run_id, 'rl', {}).status_code == 200
    assert _run(api, run_id, 'rl_eval', {'eval_base': True}).status_code == 200

    first = api.get(f'/api/runs/{run_id}/eval-reports').json()['reports']
    first_ids = {row['id'] for row in first}
    assert any(row['stage'] == 'rl_eval' and row['role'] == 'rl' for row in first)
    assert any(row['stage'] == 'rl_eval' and row['role'] == 'sft' for row in first)

    # Re-run sft_eval supersedes the prior step; both should remain listable.
    assert _run(api, run_id, 'sft_eval', {}).status_code == 200
    again = api.get(f'/api/runs/{run_id}/eval-reports').json()['reports']
    assert len(again) > len(first)
    historical = [row for row in again if row['status'] == 'superseded']
    assert historical
    assert any('历史#' in row['label'] for row in historical)
    # Prior ids still present (picker can compare history vs current).
    assert first_ids.issubset({row['id'] for row in again})


def test_report_endpoint_accepts_step(client):
    api = client
    run_id = _create(api, 'step-report')
    assert _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200
    assert _run(api, run_id, 'sft_eval', {'eval_base': True}).status_code == 200
    reports = api.get(f'/api/runs/{run_id}/eval-reports').json()['reports']
    base = next(row for row in reports if row['role'] == 'base')
    body = api.get(base['report_url'])
    assert body.status_code == 200, body.text
    assert 'mock eval report' in body.text


def test_eval_reports_not_embedded_in_run_view(client):
    api = client
    run_id = _create(api, 'no-bloat')
    assert _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200
    assert _run(api, run_id, 'sft_eval', {'eval_base': True}).status_code == 200
    view = api.get(f'/api/runs/{run_id}').json()
    assert 'eval-reports' not in view
    assert 'eval_reports' not in view
    # Auto compare remains on the stage result.
    assert len(view['forms']['sft_eval']['result']['compare']) >= 2


def test_ui_has_eval_picker_markup():
    root = Path(__file__).resolve().parents[1] / 'static'
    script = (root / 'app.js').read_text(encoding='utf-8')
    css = (root / 'app.css').read_text(encoding='utf-8')
    html = (root / 'index.html').read_text(encoding='utf-8')
    assert 'function evalPickerMarkup(' in script
    assert 'function openEvalCompareDialog(' in script
    assert 'function loadEvalReports(' in script
    assert 'function runEvalPickerCompare(' in script
    assert 'function buildCompareTableHtml(' in script
    assert 'data-action="eval-picker-open"' in script
    assert '对比' in script
    assert '/eval-reports' in script
    assert 'eval-picker-chip' in css
    assert 'eval-picker-launch' in css
    assert 'id="eval-compare-dialog"' in html
    assert 'data-action="eval-picker-compare"' in html
