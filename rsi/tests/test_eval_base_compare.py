# Copyright (c) ModelScope Contributors. All rights reserved.
"""Base-model dual eval for sft_eval / rl_eval."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rsi.commands import INTERNAL_KEYS, build_argv
from rsi.eval_report import base_eval_output_dir, compact_eval_metrics

BASE = 'Qwen/Qwen2.5-7B-Instruct'


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    app = create_app(tmp_path / 'rsi-home')
    with TestClient(app) as test_client:
        yield test_client


def _create(api, name='基座对比'):
    return api.post('/api/runs', json={
        'name': name,
        'scenario': '评测阶段要同时跑基座',
        'base_model': BASE,
        'model_type': 'qwen2',
        'template': 'qwen2_5',
    }).json()['run']['id']


def _run(api, run_id, stage, params=None):
    return api.post(f'/api/runs/{run_id}/stages/{stage}/run', json={'params': params or {}})


def _preview(api, run_id, stage, params=None):
    return api.post(f'/api/runs/{run_id}/stages/{stage}/preview', json={'params': params or {}}).json()


def test_eval_base_is_internal_and_helpers():
    assert 'eval_base' in INTERNAL_KEYS
    assert '--eval_base' not in ' '.join(build_argv('sft_eval', {
        'model': BASE,
        'eval_dataset': 'gsm8k',
        'eval_base': True,
        'adapters': '/tmp/adapter',
    }))
    root = Path('/tmp/run/round_1/sft_eval/eval_output')
    assert base_eval_output_dir(root) == root / 'base'
    assert base_eval_output_dir(root / 'adapter') == root / 'base'
    assert compact_eval_metrics({'accuracy': 0.5, 'report_html': '/x', 'scores': {'bleu-1': 0.1}}) == {
        'accuracy': 0.5,
        'scores': {'bleu-1': 0.1},
    }


def test_sft_eval_preview_base_has_no_adapters(client):
    api = client
    run_id = _create(api, 'sft-preview')
    assert _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200
    body = _preview(api, run_id, 'sft_eval', {})
    assert body['params'].get('eval_base') is True
    assert body['params'].get('adapters')
    assert '--adapters' in body['command']
    base = body['base_params']
    assert base['model'] == BASE
    assert not base.get('adapters')
    assert str(base['eval_output_dir']).endswith('/eval_output/base')
    assert '--adapters' not in body['base_command']
    assert f'--model {BASE}' in body['base_command']
    assert 'swift eval' in body['base_command']


def test_sft_eval_run_stores_base_compare(client):
    api = client
    run_id = _create(api, 'sft-compare')
    assert _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200
    evaluated = _run(api, run_id, 'sft_eval', {})
    assert evaluated.status_code == 200, evaluated.text
    result = evaluated.json()['result']
    compare = result['compare']
    assert [row['role'] for row in compare] == ['base', 'sft']
    assert compare[0]['label'] == '基座'
    assert compare[1]['label'] == 'SFT'
    assert compare[0]['metrics']['accuracy'] < compare[1]['metrics']['accuracy']
    assert result['metrics']['accuracy'] == compare[1]['metrics']['accuracy']
    assert str(compare[0]['eval_output_dir']).endswith('/eval_output/base')
    assert result.get('base_command')
    assert '--adapters' not in result['base_command']

    off = _run(api, run_id, 'sft_eval', {'eval_base': False})
    assert off.status_code == 200, off.text
    assert 'base_command' not in (off.json()['result'] or {})
    roles = [row['role'] for row in off.json()['result']['compare']]
    assert roles == ['sft']


def test_rl_eval_compare_includes_sft_when_available(client):
    api = client
    run_id = _create(api, 'rl-compare')
    assert _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200
    assert _run(api, run_id, 'sft_eval', {}).status_code == 200
    assert _run(api, run_id, 'rl', {}).status_code == 200
    preview = _preview(api, run_id, 'rl_eval', {})
    assert preview['base_params']['model'] == BASE
    assert not preview['base_params'].get('adapters')
    assert '--adapters' not in preview['base_command']

    evaluated = _run(api, run_id, 'rl_eval', {})
    assert evaluated.status_code == 200, evaluated.text
    compare = evaluated.json()['result']['compare']
    assert [row['role'] for row in compare] == ['base', 'sft', 'rl']
    assert compare[0]['metrics']['accuracy'] < compare[2]['metrics']['accuracy']
    # Prior SFT metrics are reused (no second SFT eval run).
    view = api.get(f'/api/runs/{run_id}').json()
    sft_acc = view['forms']['sft_eval']['result']['metrics']['accuracy']
    assert compare[1]['metrics']['accuracy'] == sft_acc


def test_ui_has_compare_table_markup():
    script = (Path(__file__).resolve().parents[1] / 'static' / 'app.js').read_text(encoding='utf-8')
    assert 'function compareTable(' in script
    assert '基座对比' in script
    assert 'kind=' in script
