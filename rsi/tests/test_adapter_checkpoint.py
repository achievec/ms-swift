# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BASE = 'Qwen/Qwen2.5-7B-Instruct'
ARGS_BASE = '/data/models/Qwen2.5-7B-Instruct'


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    app = create_app(tmp_path / 'rsi-home')
    with TestClient(app) as test_client:
        yield test_client


def _run(api, run_id, stage, params=None):
    return api.post(f'/api/runs/{run_id}/stages/{stage}/run', json={'params': params or {}})


def _preview(api, run_id, stage, params=None):
    response = api.post(f'/api/runs/{run_id}/stages/{stage}/preview', json={'params': params or {}})
    assert response.status_code == 200, response.text
    return response.json()


def _create(api):
    created = api.post('/api/runs', json={
        'name': 'lora-handoff',
        'scenario': 'LoRA 检查点交给下游阶段',
        'base_model': BASE,
        'model_type': 'qwen2',
        'template': 'qwen2_5',
    })
    assert created.status_code == 200, created.text
    return created.json()['run']['id']


def _train_sft(api, run_id):
    assert _run(api, run_id, 'data', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200
    trained = _run(api, run_id, 'sft', {})
    assert trained.status_code == 200, trained.text
    checkpoint = Path(trained.json()['result']['checkpoint'])
    args = json.loads((checkpoint / 'args.json').read_text(encoding='utf-8'))
    args['model'] = ARGS_BASE
    (checkpoint / 'args.json').write_text(json.dumps(args), encoding='utf-8')
    (checkpoint / 'adapter_config.json').write_text('{"peft_type": "LORA"}', encoding='utf-8')
    return checkpoint


def test_lora_checkpoint_is_passed_as_adapters(client):
    api = client
    run_id = _create(api)
    checkpoint = _train_sft(api, run_id)

    evaluated = _preview(api, run_id, 'sft_eval')
    params = evaluated['params']
    command = evaluated['command']
    assert params['model'] == ARGS_BASE
    assert params['adapters'] == str(checkpoint)
    assert params['model_type'] == 'qwen2'
    assert params['template'] == 'qwen2_5'
    assert f'--model {ARGS_BASE}' in command
    assert f'--adapters {checkpoint}' in command
    assert f'--model {checkpoint}' not in command
    assert '--model_type qwen2' in command
    assert '--template qwen2_5' in command
    assert 'swift eval' in command

    rl = _preview(api, run_id, 'rl')
    assert rl['params']['model'] == ARGS_BASE
    assert rl['params']['adapters'] == str(checkpoint)
    assert f'--adapters {checkpoint}' in rl['command']
    assert f'--model {checkpoint}' not in rl['command']
    assert 'swift rlhf' in rl['command']

    stored = json.loads((checkpoint / 'args.json').read_text(encoding='utf-8'))
    stored.pop('model')
    (checkpoint / 'args.json').write_text(json.dumps(stored), encoding='utf-8')
    missing = _preview(api, run_id, 'sft_eval')
    assert missing['params']['model'] == BASE
    assert missing['params']['adapters'] == str(checkpoint)
    assert f'--model {BASE}' in missing['command']
    assert f'--adapters {checkpoint}' in missing['command']

    stored['model'] = ARGS_BASE
    (checkpoint / 'args.json').write_text(json.dumps(stored), encoding='utf-8')
    (checkpoint / 'config.json').write_text('{"model_type": "qwen2"}', encoding='utf-8')
    both = _preview(api, run_id, 'sft_eval')
    assert both['params']['adapters'] == str(checkpoint)
    assert f'--model {checkpoint}' not in both['command']

    assert _run(api, run_id, 'sft_eval', {}).status_code == 200
    trained_rl = _run(api, run_id, 'rl', {})
    assert trained_rl.status_code == 200, trained_rl.text
    assert f'--adapters {checkpoint}' in trained_rl.json()['command']
    rl_checkpoint = Path(trained_rl.json()['result']['checkpoint'])
    assert _run(api, run_id, 'rl_eval', {}).status_code == 200
    rl_eval = _preview(api, run_id, 'rl_eval')
    assert rl_eval['params']['adapters'] == str(rl_checkpoint)
    assert rl_eval['params']['model'] == ARGS_BASE
    assert f'--adapters {rl_checkpoint}' in rl_eval['command']
    assert f'--model {rl_checkpoint}' not in rl_eval['command']
    assert '--model_type qwen2' in rl_eval['command']

    assert _run(api, run_id, 'promote', {'choice': 'sft', 'reason': '用 LoRA'}).status_code == 200
    opened = _run(api, run_id, 'next_round', {'reuse_dataset': True, 'note': '下一轮'})
    assert opened.status_code == 200, opened.text
    # adapter_config.json is still present, so the promoted dir stays an adapter
    # even though the earlier preview also wrote config.json.
    nxt = _preview(api, run_id, 'sft')
    assert nxt['params']['adapters'] == str(checkpoint)
    assert nxt['params']['model'] == ARGS_BASE
    assert f'--adapters {checkpoint}' in nxt['command']
    assert f'--model {checkpoint}' not in nxt['command']
    assert 'swift sft' in nxt['command']
    assert '--model_type qwen2' in nxt['command']
    assert '--template qwen2_5' in nxt['command']


def test_full_checkpoint_stays_on_model(client):
    api = client
    run_id = _create(api)
    checkpoint = _train_sft(api, run_id)
    (checkpoint / 'adapter_config.json').unlink()
    (checkpoint / 'config.json').write_text('{"model_type": "qwen2"}', encoding='utf-8')

    evaluated = _preview(api, run_id, 'sft_eval')
    assert evaluated['params']['model'] == str(checkpoint)
    assert not evaluated['params'].get('adapters')
    assert f'--model {checkpoint}' in evaluated['command']
    assert '--adapters' not in evaluated['command']
    assert '--model_type qwen2' in evaluated['command']
    assert '--template qwen2_5' in evaluated['command']

    rl = _preview(api, run_id, 'rl')
    assert rl['params']['model'] == str(checkpoint)
    assert not rl['params'].get('adapters')
    assert f'--model {checkpoint}' in rl['command']
    assert '--adapters' not in rl['command']
