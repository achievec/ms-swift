# Copyright (c) ModelScope Contributors. All rights reserved.
import ast
import json
import sqlite3
import textwrap
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    home = tmp_path / 'rsi-home'
    app = create_app(home)
    with TestClient(app) as test_client:
        yield test_client, home


def _run(client, run_id, stage, params=None):
    response = client.post(f'/api/runs/{run_id}/stages/{stage}/run', json={'params': params or {}})
    return response


def test_happy_path_records_audit_and_serves(client, monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError('mock mode must not spawn a process')

    monkeypatch.setattr('subprocess.run', explode)
    monkeypatch.setattr('subprocess.Popen', explode)
    api, home = client
    health = api.get('/api/health')
    assert health.status_code == 200
    assert health.json()['backend'] == 'mock'
    assert 'A100' in health.json()['hardware']['gpu']

    created = api.post('/api/runs', json={
        'name': 'IT 运维助手',
        'scenario': '企业内部 IT 工单分诊与处理建议',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    assert created.status_code == 200, created.text
    run_id = created.json()['run']['id']

    skipped = _run(api, run_id, 'sft', {})
    assert skipped.status_code == 400
    assert '数据' in skipped.json()['detail']

    data = _run(api, run_id, 'data', {'source': 'bundled', 'bundled': 'itops_mix'})
    assert data.status_code == 200, data.text
    validation = data.json()['result']['validation']
    assert validation['ok'] is True
    assert validation['num_rows'] == 6

    sft = _run(api, run_id, 'sft', {'extra_yaml': 'learning_rate: 2e-5\n'})
    assert sft.status_code == 200, sft.text
    sft_body = sft.json()
    assert 'swift sft' in sft_body['command']
    assert '2e-5' in sft_body['command']
    assert 'itops_mix.jsonl' in sft_body['command']
    assert sft_body['result']['mock'] is True
    sft_ckpt = sft_body['result']['checkpoint']
    assert sft_ckpt.endswith('checkpoint-final')
    assert json_file(Path(sft_ckpt) / 'args.json')['rsi_mock'] is True

    sft_eval = _run(api, run_id, 'sft_eval', {})
    assert sft_eval.status_code == 200, sft_eval.text
    assert 'swift eval' in sft_eval.json()['command']
    sft_acc = sft_eval.json()['result']['metrics']['accuracy']
    assert sft_eval.json()['result']['metrics']['eval_rows'] == 4

    rl = _run(api, run_id, 'rl', {})
    assert rl.status_code == 200, rl.text
    rl_command = rl.json()['command']
    assert 'swift rlhf' in rl_command
    assert 'grpo' in rl_command
    assert 'colocate' in rl_command
    assert 'itops_checklist' in rl_command
    assert 'itops_reward.py' in rl_command
    assert sft_ckpt in rl_command

    rl_eval = _run(api, run_id, 'rl_eval', {})
    assert rl_eval.status_code == 200, rl_eval.text
    rl_acc = rl_eval.json()['result']['metrics']['accuracy']
    assert rl_acc > sft_acc

    rejected = _run(api, run_id, 'promote', {'choice': 'nope'})
    assert rejected.status_code == 400

    view = api.get(f'/api/runs/{run_id}').json()
    assert view['recommendation'] == 'rl'
    promoted = next(item['path'] for item in view['candidates'] if item['kind'] == 'rl')

    promote = _run(api, run_id, 'promote', {'choice': 'rl', 'reason': 'RL accuracy 更高'})
    assert promote.status_code == 200, promote.text
    assert promote.json()['result']['promoted_checkpoint'] == promoted

    nxt = _run(api, run_id, 'next_round', {'reuse_dataset': True, 'note': '用 RL 检查点开下一轮'})
    assert nxt.status_code == 200, nxt.text
    assert nxt.json()['view']['run']['round_index'] == 2
    assert nxt.json()['view']['run']['current_stage'] == 'data'
    assert nxt.json()['view']['round']['base_model'] == promoted

    again = _run(api, run_id, 'data', {})
    assert again.status_code == 200, again.text
    round2 = api.get(f'/api/runs/{run_id}').json()
    assert round2['forms']['sft']['values']['model'] == promoted
    assert round2['run']['current_stage'] == 'sft'

    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]

    served = api.post(f'/api/runs/{run_id}/serve', json={'params': {'port': port, 'api_key': 'rsi-test-key'}})
    assert served.status_code == 200, served.text
    body = served.json()
    assert 'swift deploy' in body['command']
    assert body['result']['checkpoint'] == promoted
    base = body['result']['base_url']

    missing = httpx.get(f'{base}/models', timeout=5)
    assert missing.status_code == 401
    headers = {'Authorization': 'Bearer rsi-test-key'}
    models = httpx.get(f'{base}/models', headers=headers, timeout=5)
    assert models.status_code == 200
    assert models.json()['data'][0]['id'] == body['result']['served_model_name']

    completion = httpx.post(
        f'{base}/chat/completions',
        headers=headers,
        json={'model': body['result']['served_model_name'], 'messages': [{'role': 'user', 'content': 'VPN 连不上'}]},
        timeout=5)
    assert completion.status_code == 200
    text = completion.json()['choices'][0]['message']['content']
    assert 'checkpoint-final' in text
    assert 'VPN 连不上' in text

    proxied = api.post(f'/api/runs/{run_id}/chat', json={'messages': [{'role': 'user', 'content': '打印机脱机'}]})
    assert proxied.status_code == 200
    assert '打印机脱机' in proxied.json()['choices'][0]['message']['content']

    stopped = api.post(f'/api/runs/{run_id}/serve/stop')
    assert stopped.status_code == 200
    assert stopped.json()['view']['service']['status'] == 'stopped'

    conn = sqlite3.connect(home / 'rsi.sqlite')
    columns = [row[1] for row in conn.execute('PRAGMA table_info(steps)')]
    for name in ('started_at', 'ended_at', 'params_json', 'result_json', 'logs', 'status', 'command'):
        assert name in columns
    rows = conn.execute(
        'SELECT stage, status, started_at, ended_at, params_json, result_json, logs FROM steps ORDER BY id').fetchall()
    stages = [row[0] for row in rows]
    for stage in ('data', 'sft', 'sft_eval', 'rl', 'rl_eval', 'promote', 'next_round', 'serve'):
        assert stage in stages
    for stage, status, started, ended, params, result, logs in rows:
        assert status in {'succeeded', 'failed', 'superseded', 'running'}
        assert started and ended and params and result is not None and logs
        assert ended >= started
    assert conn.execute('SELECT COUNT(*) FROM rounds').fetchone()[0] == 2
    assert conn.execute("SELECT promoted_kind FROM rounds WHERE round_index=1").fetchone()[0] == 'rl'


def test_rerun_supersedes_later_stages(client):
    api, _home = client
    run_id = api.post('/api/runs', json={
        'name': '回退',
        'scenario': '验证作废后续阶段',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    }).json()['run']['id']
    assert _run(api, run_id, 'data', {'source': 'bundled'}).status_code == 200
    assert _run(api, run_id, 'sft', {}).status_code == 200
    assert _run(api, run_id, 'data', {'source': 'bundled'}).status_code == 200
    view = api.get(f'/api/runs/{run_id}').json()
    sft_steps = [step for step in view['steps'] if step['stage'] == 'sft']
    assert sft_steps and all(step['status'] == 'superseded' for step in sft_steps)
    assert view['forms']['sft']['status'] == 'ready'
    blocked = _run(api, run_id, 'sft_eval', {})
    assert blocked.status_code == 400


def test_upload_and_public_dataset(client, tmp_path):
    api, home = client
    good = tmp_path / 'alpaca.jsonl'
    good.write_text('{"instruction": "hi", "output": "there"}\n', encoding='utf-8')
    public = api.post('/api/datasets/validate', json={'params': {'source': 'public', 'dataset': 'modelscope/gsm8k'}})
    assert public.status_code == 200
    assert public.json()['ok'] is True
    local = api.post('/api/datasets/validate', json={'params': {'source': 'path', 'dataset': str(good)}})
    assert local.json()['ok'] is True
    assert local.json()['formats']['alpaca'] == 1

    upload = api.post('/api/datasets/upload', files={'file': ('bad.jsonl', b'{"nope": 1}\n', 'application/jsonl')})
    assert upload.status_code == 200
    assert upload.json()['validation']['ok'] is False
    uploaded = Path(upload.json()['path'])
    assert uploaded.is_file()
    assert home in uploaded.parents or uploaded.parent == home / 'uploads'


def test_swift_backend_subprocess(tmp_path, monkeypatch):
    script = tmp_path / 'fake-swift.py'
    script.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        print('SWIFT_OK', args[0] if args else '')
        if '--output_dir' in args:
            out = Path(args[args.index('--output_dir') + 1])
            ckpt = out / 'checkpoint-7'
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / 'args.json').write_text('{"model": "fake"}')
        if args and args[0] == 'eval':
            print('{"accuracy": 0.42}')
        raise SystemExit(0)
    """), encoding='utf-8')
    script.chmod(0o755)
    monkeypatch.setenv('RSI_BACKEND', 'swift')
    monkeypatch.setenv('RSI_SWIFT_BIN', str(script))
    from rsi.app import create_app
    home = tmp_path / 'home'
    app = create_app(home)
    with TestClient(app) as api:
        assert api.get('/api/health').json()['backend'] == 'swift'
        run_id = api.post('/api/runs', json={
            'name': 'swift-stub',
            'scenario': '确认真实后端会拉起 swift 进程',
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
        }).json()['run']['id']
        assert _run(api, run_id, 'data', {'source': 'bundled'}).status_code == 200
        started = _run(api, run_id, 'sft', {})
        assert started.status_code == 200, started.text
        view = started.json()['view']
        status = view['forms']['sft']['status']
        logs = view['forms']['sft']['logs'] or ''
        deadline = time.time() + 8
        while time.time() < deadline and status == 'running':
            time.sleep(0.1)
            view = api.get(f'/api/runs/{run_id}').json()
            status = view['forms']['sft']['status']
            logs = view['forms']['sft']['logs'] or ''
        assert status == 'succeeded', logs
        assert 'SWIFT_OK' in logs
        assert logs
        checkpoint = view['forms']['sft']['result']['checkpoint']
        assert checkpoint.endswith('checkpoint-7')
        assert view['forms']['sft']['result']['mock'] is False


def test_default_backend_calls_real_swift(tmp_path, monkeypatch):
    monkeypatch.delenv('RSI_BACKEND', raising=False)
    monkeypatch.delenv('RSI_SERVE_BACKEND', raising=False)
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    from rsi.app import create_app
    from rsi.backends import swift_launch
    app = create_app(tmp_path / 'home')
    with TestClient(app) as api:
        health = api.get('/api/health').json()
        assert health['backend'] == 'swift'
        assert health['serve_backend'] == 'swift'
        assert 'gpu' in health
        run_id = api.post('/api/runs', json={
            'name': 'real-swift',
            'scenario': '确认默认后端会生成真实 swift eval 参数',
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
        }).json()['run']['id']
        preview = api.post(f'/api/runs/{run_id}/stages/sft_eval/preview', json={'params': {}})
        assert preview.status_code == 200, preview.text
        command = preview.json()['command']
        assert 'swift eval' in command
        assert 'general_qa' in command
        assert 'eval_dataset_args' in command
        assert 'itops-heldout' not in command
        params = preview.json()['params']
        qa_dir = Path(params['eval_dataset_args']['general_qa']['local_path'])
        assert (qa_dir / 'itops.jsonl').is_file()
        first = json.loads((qa_dir / 'itops.jsonl').read_text(encoding='utf-8').splitlines()[0])
        assert first['query'] and first['response']
        kept = api.post(f'/api/runs/{run_id}/stages/sft_eval/preview', json={
            'params': {'eval_backend': 'OpenCompass', 'eval_dataset': 'gsm8k ceval'},
        })
        assert kept.status_code == 200, kept.text
        kept_params = kept.json()['params']
        assert kept_params['eval_dataset'] == 'gsm8k ceval'
        assert 'eval_dataset_args' not in kept_params
        assert '--eval_dataset gsm8k ceval' in kept.json()['command'] or (
            'gsm8k' in kept.json()['command'] and 'ceval' in kept.json()['command'])
    argv, env = swift_launch(['swift', 'sft', '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    assert argv[1:4] == ['-m', 'swift.cli.main', 'sft']
    assert env['CUDA_VISIBLE_DEVICES'] == '0'
    assert str(Path(__file__).resolve().parents[2]) in env['PYTHONPATH'].split(':')


def test_swift_ui_fallback_on_same_port(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    from rsi.app import create_app
    from rsi.__main__ import mount_swift_ui
    app = create_app(tmp_path / 'home')
    mounted = mount_swift_ui(app)
    with TestClient(app) as api:
        page = api.get('/swift-ui/')
        assert page.status_code == 200
        assert 'SWIFT Web-UI' in page.text
        assert '返回大模型训练平台' in page.text
        if not mounted:
            assert app.state.swift_ui is False
            assert 'swift web-ui' not in page.text
            assert 'torch' not in page.text


def test_reward_plugin_parses():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'itops_reward.py'
    ast.parse(path.read_text(encoding='utf-8'))


def test_delete_run_removes_audit_and_only_its_directory(client):
    api, home = client
    created = api.post('/api/runs', json={
        'name': '要删除的实验',
        'scenario': '删除后审计和实验目录都应消失',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    assert created.status_code == 200, created.text
    run_id = created.json()['run']['id']
    other = api.post('/api/runs', json={
        'name': '留下的实验',
        'scenario': '删除邻居时这个实验还在',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    other_id = other.json()['run']['id']
    data = _run(api, run_id, 'data', {'source': 'bundled', 'bundled': 'itops_mix'})
    assert data.status_code == 200, data.text

    workspace = home / 'runs' / run_id / 'round_1' / 'sft'
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / 'marker.txt').write_text('artifact', encoding='utf-8')
    uploads = home / 'uploads'
    uploads.mkdir(exist_ok=True)
    kept = uploads / 'user-dataset.jsonl'
    kept.write_text('{"messages":[]}\n', encoding='utf-8')
    neighbor = home / 'runs' / 'not-this-run'
    neighbor.mkdir()
    (neighbor / 'stay.txt').write_text('stay', encoding='utf-8')

    missing = api.delete('/api/runs/does-not-exist')
    assert missing.status_code == 404

    removed = api.delete(f'/api/runs/{run_id}')
    assert removed.status_code == 200, removed.text
    body = removed.json()
    assert body['ok'] is True
    assert body['id'] == run_id
    assert body['workspace_removed'] is True
    assert api.get(f'/api/runs/{run_id}').status_code == 404
    listed = {item['id'] for item in api.get('/api/runs').json()['runs']}
    assert run_id not in listed
    assert other_id in listed
    assert not (home / 'runs' / run_id).exists()
    assert kept.is_file()
    assert (neighbor / 'stay.txt').read_text(encoding='utf-8') == 'stay'
    pipeline = api.app.state.pipeline
    assert pipeline._remove_run_workspace('../uploads') is False
    assert pipeline._remove_run_workspace('..') is False
    assert kept.is_file()

    db = sqlite3.connect(home / 'rsi.sqlite')
    try:
        assert db.execute('SELECT COUNT(*) FROM runs WHERE id=?', (run_id, )).fetchone()[0] == 0
        for table in ('rounds', 'steps', 'checkpoints', 'services'):
            count = db.execute(f'SELECT COUNT(*) FROM {table} WHERE run_id=?', (run_id, )).fetchone()[0]
            assert count == 0, table
        assert db.execute('SELECT COUNT(*) FROM runs WHERE id=?', (other_id, )).fetchone()[0] == 1
    finally:
        db.close()

    alias = api.delete(f'/api/experiments/{other_id}')
    assert alias.status_code == 200, alias.text
    assert api.get('/api/runs').json()['runs'] == []
    assert api.delete(f'/api/experiments/{other_id}').status_code == 404


def test_empty_post_body_is_not_internal_server_error(client):
    api, _home = client
    created = api.post('/api/runs', json={
        'name': '空请求',
        'scenario': '空 body 不能变成 500',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    assert created.status_code == 200, created.text
    run_id = created.json()['run']['id']
    headers = {'Content-Type': 'application/json'}
    started = api.post(f'/api/runs/{run_id}/stages/sft/run', content=b'', headers=headers)
    assert started.status_code != 500, started.text
    assert started.headers['content-type'].startswith('application/json')
    assert started.json()['detail']
    assert 'Internal Server Error' not in started.text
    blank = api.post(f'/api/runs/{run_id}/stages/sft/run', content=b'   \n', headers=headers)
    assert blank.status_code != 500, blank.text
    assert blank.headers['content-type'].startswith('application/json')
    bad = api.post(f'/api/runs/{run_id}/stages/sft/run', content=b'not-json', headers=headers)
    assert bad.status_code == 400
    assert bad.json()['detail'] == '请求体不是合法的 JSON'
    stopped = api.post(f'/api/runs/{run_id}/serve/stop', content=b'', headers=headers)
    assert stopped.status_code == 400, stopped.text
    assert stopped.headers['content-type'].startswith('application/json')
    assert stopped.json()['detail']
    assert 'Internal Server Error' not in stopped.text
    script = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert "body: '{}'" in script
    assert '模型同步中' in script


def test_unhandled_error_returns_json_detail(client, monkeypatch):
    api, _home = client
    created = api.post('/api/runs', json={
        'name': '内部错误',
        'scenario': '异常也要返回 JSON',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    run_id = created.json()['run']['id']

    def boom(*_args, **_kwargs):
        raise RuntimeError('训练进程异常')

    monkeypatch.setattr(api.app.state.pipeline, 'run_stage', boom)
    quiet = TestClient(api.app, raise_server_exceptions=False)
    response = quiet.post(f'/api/runs/{run_id}/stages/sft/run', json={'params': {}})
    assert response.status_code == 500, response.text
    assert response.headers['content-type'].startswith('application/json')
    assert response.json()['detail'] == '训练进程异常'
    assert 'Internal Server Error' not in response.text


def test_preset_yaml_matches_schema():
    from rsi.schemas import presets
    data = presets()
    assert data['sft']['tuner_type'] == 'lora'
    assert data['sft']['per_device_train_batch_size'] == 1
    assert data['rl']['rlhf_type'] == 'grpo'
    assert data['rl']['vllm_mode'] == 'colocate'
    assert data['rl']['num_generations'] == 4
    assert data['hardware']['gpu'].startswith('1x NVIDIA A100')


def json_file(path: Path):
    import json
    return json.loads(path.read_text(encoding='utf-8'))
