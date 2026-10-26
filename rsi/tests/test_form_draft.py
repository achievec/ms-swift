# Copyright (c) ModelScope Contributors. All rights reserved.
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DATASETS = ['AI-ModelScope/alpaca-gpt4-data-zh', 'modelscope/gsm8k']


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    monkeypatch.setenv('RSI_HOME', str(tmp_path / 'home'))
    from rsi.app import create_app
    home = tmp_path / 'home'
    app = create_app(home)
    with TestClient(app) as test_client:
        yield test_client, home


def _create(api, name='草稿'):
    created = api.post('/api/runs', json={
        'name': name,
        'scenario': '刷新后还要看到刚才填的值',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    assert created.status_code == 200, created.text
    return created.json()


def _draft(api, run_id, stage, params, round_index=1):
    response = api.post(f'/api/runs/{run_id}/drafts', json={
        'stage': stage,
        'round': round_index,
        'params': params,
    })
    assert response.status_code == 200, response.text
    return response.json()


def test_new_sft_dataset_is_empty_and_eval_keeps_heldout(client):
    api, _home = client
    created = _create(api)
    sft = created['forms']['sft']['values']
    assert sft['dataset'] == ''
    assert sft['source'] == 'public'
    assert sft['max_length'] == 2048
    assert sft['model'] == 'Qwen/Qwen2.5-7B-Instruct'
    assert not created['forms']['rl']['values'].get('dataset')
    assert created['forms']['sft_eval']['values']['eval_dataset'] == 'itops-heldout'
    assert created['forms']['rl_eval']['values']['eval_dataset'] == 'itops-heldout'
    preset = created['forms']['sft']['preset']
    assert preset['dataset'] == ''


def test_multiselect_and_each_stage_round_trip(client):
    api, home = client
    created = _create(api, '多选')
    run_id = created['run']['id']
    base = created['run']['base_model']
    saved = _draft(api, run_id, 'sft', {
        'dataset': DATASETS,
        'max_length': 1024,
        'padding_free': True,
        'split_dataset_ratio': 0.1,
        'source': 'public',
        'model': 'other/model',
        'model_type': 'llama',
        'template': 'llama3',
        'nested': {'skip': True},
    })
    assert saved['params']['dataset'] == ' '.join(DATASETS)
    assert 'model' not in saved['params']
    assert 'nested' not in saved['params']

    _draft(api, run_id, 'sft_eval', {'eval_limit': 7})
    _draft(api, run_id, 'rl', {'num_generations': 3, 'learning_rate': '2e-6'})
    _draft(api, run_id, 'rl_eval', {'eval_limit': 9})
    _draft(api, run_id, 'promote', {'reason': '用 SFT', 'note': '换一批'})
    _draft(api, run_id, 'serve', {
        'temperature': 0.2,
        'model': 'other/model',
        'adapters': '/tmp/should-not-stick',
    })
    _draft(api, run_id, 'export', {
        'max_length': 1536,
        'dataset': 'local/set.jsonl',
        'adapters': '/tmp/lora-draft',
        'model': 'other/model',
    })

    view = api.get(f'/api/runs/{run_id}').json()
    sft = view['forms']['sft']['values']
    assert sft['dataset'].split() == DATASETS
    assert sft['max_length'] == 1024
    assert sft['padding_free'] is True
    assert sft['split_dataset_ratio'] == 0.1
    assert sft['model'] == base
    assert sft['model_type'] == 'qwen2'
    assert sft['template'] == 'qwen2_5'
    assert 'nested' not in sft

    assert view['forms']['sft_eval']['values']['eval_limit'] == 7
    assert view['forms']['sft_eval']['values']['eval_dataset'] == 'itops-heldout'
    rl = view['forms']['rl']['values']
    assert rl['num_generations'] == 3
    assert rl['learning_rate'] == '2e-6'
    assert not rl.get('dataset')
    assert rl['model'] == base
    assert view['forms']['rl_eval']['values']['eval_limit'] == 9
    assert view['forms']['promote']['values']['reason'] == '用 SFT'
    assert view['forms']['promote']['values']['note'] == '换一批'

    serve = view['serve_form']
    assert serve['temperature'] == 0.2
    assert serve['model'] == base
    assert not serve.get('adapters')
    export = view['export_form']
    assert export['max_length'] == 1536
    assert export['dataset'] == 'local/set.jsonl'
    assert export['adapters'] == '/tmp/lora-draft'
    assert export['model'] == base

    other = _create(api, '另一个实验')
    assert other['forms']['sft']['values']['dataset'] == ''
    assert other['forms']['sft']['values']['max_length'] == 2048
    assert other['serve_form']['temperature'] == 0.3
    assert other['forms']['promote']['values'].get('reason') in (None, '')
    assert other['export_form'].get('adapters') != '/tmp/lora-draft'

    empty = _draft(api, run_id, 'sft', {'model': 'hacked'})
    assert empty['params']['dataset'] == ' '.join(DATASETS)
    again = api.get(f'/api/runs/{run_id}').json()
    assert again['forms']['sft']['values']['dataset'].split() == DATASETS

    reset = _draft(api, run_id, 'sft', {
        'dataset': '',
        'source': 'public',
        'max_length': 2048,
        'padding_free': False,
        'split_dataset_ratio': 0,
    })
    assert reset['params']['dataset'] == ''
    restored = api.get(f'/api/runs/{run_id}').json()
    assert restored['forms']['sft']['values']['dataset'] == ''
    assert restored['forms']['sft']['values']['max_length'] == 2048
    assert restored['forms']['sft']['preset']['dataset'] == ''

    missing = api.post(f'/api/runs/{run_id}/stages/sft/run', json={'params': {}})
    assert missing.status_code == 400
    assert missing.json()['detail'] == '请先选择数据集'
    explicit = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
        'params': {'source': 'public', 'dataset': '   '},
    })
    assert explicit.status_code == 400
    assert explicit.json()['detail'] == '请先选择数据集'

    bundled = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
        'params': {'source': 'bundled', 'bundled': 'itops_mix', 'learning_rate': '2e-4'},
    })
    assert bundled.status_code == 200, bundled.text
    trained = api.get(f'/api/runs/{run_id}').json()
    assert trained['forms']['sft']['values']['learning_rate'] == '2e-4'
    assert trained['forms']['sft']['values']['model'] == base

    public = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
        'params': {'source': 'public', 'dataset': DATASETS, 'max_length': 1536},
    })
    assert public.status_code == 200, public.text
    reloaded = api.get(f'/api/runs/{run_id}').json()
    assert reloaded['forms']['sft']['values']['dataset'].split() == DATASETS
    assert reloaded['forms']['sft']['values']['max_length'] == 1536

    unknown = api.post(f'/api/runs/{run_id}/drafts', json={'stage': 'data', 'params': {'dataset': 'x'}})
    assert unknown.status_code == 400
    gone = api.post(f'/api/runs/{run_id}/drafts', json={'stage': 'sft', 'round': 9, 'params': {'dataset': 'x'}})
    assert gone.status_code == 404

    assert api.delete(f'/api/runs/{run_id}').status_code == 200
    db = sqlite3.connect(home / 'rsi.sqlite')
    try:
        count = db.execute('SELECT COUNT(*) FROM drafts WHERE run_id=?', (run_id, )).fetchone()[0]
    finally:
        db.close()
    assert count == 0


def test_draft_script_posts_before_navigation_and_preset():
    script = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert '/drafts' in script
    assert 'keepalive: true' in script
    assert 'scheduleDraftSave' in script
    assert 'localStorage' not in script
    preset = script.split('function applyPreset()', 1)[1].split('\nfunction ', 1)[0]
    assert 'flushDraft()' in preset
    assert "form.status === 'running' ? '运行中'" in script
