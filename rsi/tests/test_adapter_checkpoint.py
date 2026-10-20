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


def test_more_params_merge_json_and_flags():
    from rsi.commands import apply_more_params
    merged = apply_more_params({'model': 'base', 'more_params': '{"top_k": 7, "merge_lora": true}'})
    assert merged['model'] == 'base'
    assert merged['top_k'] == 7
    assert merged['merge_lora'] is True
    assert 'more_params' not in merged
    flagged = apply_more_params({'more_params': '--repetition_penalty 1.2 --verbose'})
    assert flagged['repetition_penalty'] == '1.2'
    assert flagged['verbose'] is True


def test_more_params_stay_with_the_experiment(client):
    import socket
    api = client
    run_id = _create(api)
    other_id = _create(api)
    text = '--vllm_gpu_memory_utilization 0.8'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    served = api.post(f'/api/runs/{run_id}/serve', json={'params': {'port': port, 'more_params': text}})
    assert served.status_code == 200, served.text
    body = served.json()
    assert body['view']['serve_form']['more_params'] == text
    assert '--vllm_gpu_memory_utilization 0.8' in body['command']
    assert '--more_params' not in body['command']
    assert 'more_params' not in body['command']

    reloaded = api.get(f'/api/runs/{run_id}').json()
    assert reloaded['serve_form']['more_params'] == text
    assert reloaded['serve_form']['model'] == BASE
    other = api.get(f'/api/runs/{other_id}').json()
    assert not other['serve_form'].get('more_params')

    stopped = api.post(f'/api/runs/{run_id}/serve/stop', json={})
    assert stopped.status_code == 200, stopped.text
    after_stop = api.get(f'/api/runs/{run_id}').json()
    assert after_stop['serve_form']['more_params'] == text
    assert after_stop['service']['status'] != 'running'

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    cleared = api.post(f'/api/runs/{run_id}/serve', json={'params': {'port': port, 'more_params': ''}})
    assert cleared.status_code == 200, cleared.text
    assert not cleared.json()['view']['serve_form'].get('more_params')


def test_running_deploy_is_listed_for_the_runtime_dropdown(client):
    import socket
    api = client
    run_id = _create(api)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    served = api.post(f'/api/runs/{run_id}/serve', json={'params': {'port': port}})
    assert served.status_code == 200, served.text
    view = served.json()['view']
    running = [row for row in view['services'] if row['status'] == 'running']
    assert len(running) == 1
    assert running[0]['port'] == port
    assert 'swift deploy' in (running[0]['command'] or '')
    page = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    tasks = page.index("if (item.id === 'running_tasks')")
    dropdown = page.index("if (item.widget === 'Dropdown')")
    assert 0 <= tasks < dropdown
    assert 'state.view.services' in page
    assert "service.status === 'running'" in page
    assert 'state.inferRuntimeOpen = true' in page


def test_finished_sft_deploys_base_plus_lora(client):
    import socket
    api = client
    run_id = _create(api)
    checkpoint = _train_sft(api, run_id)
    assert _run(api, run_id, 'sft_eval', {}).status_code == 200
    finished = api.post(f'/api/runs/{run_id}/decide', json={'action': 'finish'})
    assert finished.status_code == 200, finished.text
    view = api.get(f'/api/runs/{run_id}').json()
    assert view['run']['status'] == 'completed'
    fill = view['serve_fill']
    assert fill['adapters'] == str(checkpoint)
    assert fill['model'] == ARGS_BASE
    assert fill['checkpoint'] == str(checkpoint)
    assert view['serve_form']['adapters'] == str(checkpoint)
    assert view['serve_form']['model'] == ARGS_BASE
    assert view['serve_form']['top_p'] == 0.7

    preview = _preview(api, run_id, 'serve', {})
    assert preview['params']['adapters'] == str(checkpoint)
    assert preview['params']['model'] == ARGS_BASE
    assert f'--adapters {checkpoint}' in preview['command']
    assert f'--model {ARGS_BASE}' in preview['command']
    assert f'--model {checkpoint}' not in preview['command']
    assert '--merge_lora false' in preview['command']
    assert '--top_k 20' in preview['command']
    assert 'more_params' not in preview['command']

    overridden = _preview(api, run_id, 'serve', {'more_params': '{"max_new_tokens": 32}'})
    assert overridden['params']['max_new_tokens'] == 32
    assert '--max_new_tokens 32' in overridden['command']
    cleared = _preview(api, run_id, 'serve', {'model': BASE, 'adapters': ''})
    assert not cleared['params'].get('adapters')
    assert '--adapters' not in cleared['command']
    assert f'--model {BASE}' in cleared['command']
    bad = api.post(f'/api/runs/{run_id}/stages/serve/preview', json={'params': {'more_params': '['}})
    assert bad.status_code == 400

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    served = api.post(f'/api/runs/{run_id}/serve', json={'params': {'port': port}})
    assert served.status_code == 200, served.text
    body = served.json()
    assert f'--adapters {checkpoint}' in body['command']
    assert body['result']['checkpoint'] == str(checkpoint)

    (checkpoint / 'adapter_config.json').unlink()
    (checkpoint / 'config.json').write_text('{"model_type": "qwen2"}', encoding='utf-8')
    full = api.get(f'/api/runs/{run_id}').json()['serve_fill']
    assert full['model'] == str(checkpoint)
    assert full['adapters'] == ''

    page = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert '去部署推理服务' in page
    assert '填入本轮晋级 checkpoint' in page
    assert 'llm_infer' in page
    from rsi.llm_infer_spec import llm_infer_layout
    assert llm_infer_layout(0)['by_id']['load_checkpoint']['label'] == '部署模型'
