# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import logging
import re
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import rsi.model_fetch as model_fetch
from rsi.backends import swift_launch
from rsi.model_fetch import (
    LOGIN_MESSAGE,
    ModelFetcher,
    apply_runtime_env,
    clone_argv,
    dataset_block_message,
    describe_startup,
    model_source,
    parse_hope_list,
    public_clone_command,
)


TOKEN = 'super-secret-token'
USER = 'rsi-hope-user'
REPO = 'Qwen/Qwen2.5-7B-Instruct'


class Completed:

    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _reset_fetch_globals():
    model_fetch._INFLIGHT.clear()
    model_fetch._HOPE_WARNED = False
    yield
    model_fetch._INFLIGHT.clear()


@pytest.fixture
def hope_env(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    monkeypatch.setenv('RSI_MODEL_ROOT', str(tmp_path / 'models'))
    monkeypatch.setenv('RSI_DATA_ROOT', str(tmp_path / 'datasets'))
    monkeypatch.setenv('RSI_FETCH_INLINE', '1')
    monkeypatch.setenv('RSI_HOPE_USERNAME', USER)
    monkeypatch.setenv('RSI_HOPE_TOKEN', TOKEN)
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')


def _fetcher(tmp_path, runner):
    return ModelFetcher(tmp_path / 'home', runner=runner)


def test_reuse_when_config_json_exists(tmp_path, hope_env):
    calls = []
    dest = tmp_path / 'models' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    dest.mkdir(parents=True)
    (dest / 'config.json').write_text('{}', encoding='utf-8')
    status = _fetcher(tmp_path, lambda *args, **kwargs: calls.append(args)).ensure('model', REPO)
    assert status.state == '就绪'
    assert status.message == '已在本地'
    assert status.local_path == str(dest)
    assert calls == []
    assert TOKEN not in json.dumps(status.as_row(), ensure_ascii=False)


def test_syncing_submits_clone_without_leaking_credentials(tmp_path, hope_env):
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), kwargs.get('cwd')))
        return Completed(stdout=f'任务已存在，请等待同步 token={TOKEN}')

    status = _fetcher(tmp_path, runner).ensure('model', REPO, background=False)
    assert status.state == '同步中'
    assert status.message == '同步中'
    argv, cwd = calls[0]
    assert argv[:3] == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    assert f'--username={USER}' in argv
    assert f'--token={TOKEN}' in argv
    assert cwd == str(tmp_path / 'models' / 'Qwen')
    real, public = clone_argv('https://huggingface.co/Qwen/Qwen2.5-7B-Instruct')
    assert public == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    assert TOKEN not in public
    assert USER not in public
    assert f'--token={TOKEN}' in real
    shown = public_clone_command('https://huggingface.co/Qwen/Qwen2.5-7B-Instruct')
    assert shown == 'hope clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct'
    assert TOKEN not in shown
    blob = json.dumps(status.as_row(), ensure_ascii=False)
    assert TOKEN not in blob
    assert USER not in blob


def test_parse_hope_list_copies_status_and_does_not_invent_percent():
    listed = 'Qwen2.5-7B-Instruct  同步中  1.2GB/15.2GB  8%\n'
    parsed = parse_hope_list(listed, REPO)
    assert parsed == 'Qwen2.5-7B-Instruct 同步中 1.2GB/15.2GB 8%'
    waiting = parse_hope_list('已存在，等待同步\n', REPO)
    assert waiting == '已存在，等待同步'
    assert '%' not in waiting
    assert parse_hope_list('submitted, pending\n', REPO) == 'submitted, pending'
    size_only = parse_hope_list('1.2GB/15.2GB\n', REPO)
    assert size_only == '1.2GB/15.2GB'
    assert '%' not in size_only
    assert parse_hope_list('', REPO) == ''
    assert parse_hope_list('   \n', REPO) == ''


def test_hope_list_progress_is_shown_until_local_files_exist(tmp_path, hope_env):
    listed = f'Qwen2.5-7B-Instruct  同步中  1.2GB/15.2GB  8% token={TOKEN}\n'

    def runner(argv, **kwargs):
        if argv[1] == 'list':
            return Completed(stdout=listed)
        return Completed(stdout='submitted, pending')

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '看进度',
            'scenario': '同步时显示 hope list 原文',
            'base_model': REPO,
        }).json()['run']['id']
        refreshed = api.post(f'/api/runs/{run_id}/fetch', json={'model': REPO})
        assert refreshed.status_code == 200, refreshed.text
        row = next(item for item in refreshed.json()['fetches'] if item['kind'] == 'model')
        assert row['state'] == '同步中'
        assert row['repo_id'] == REPO
        assert '8%' in row['message']
        assert '1.2GB/15.2GB' in row['message']
        assert TOKEN not in row['message']
        assert row['state'] != '就绪'
        dest = tmp_path / 'models' / 'Qwen' / 'Qwen2.5-7B-Instruct'
        dest.mkdir(parents=True)
        (dest / 'config.json').write_text('{}', encoding='utf-8')
        again = api.get(f'/api/runs/{run_id}')
        assert again.status_code == 200, again.text
        ready = next(item for item in again.json()['fetches'] if item['kind'] == 'model')
        assert ready['state'] == '就绪'
        assert ready['message'] == '已在本地'
    page = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert 'data-fetch-note' in page
    assert '重新拉取' in page


def test_refresh_lists_then_clone_completes(tmp_path, hope_env):
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == 'clone' and len(calls) > 1:
            dest = Path(kwargs['cwd']) / 'Qwen2.5-7B-Instruct'
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'config.json').write_text('{"model_type":"qwen2"}', encoding='utf-8')
        return Completed(returncode=1, stdout='已存在，等待同步')

    fetcher = _fetcher(tmp_path, runner)
    waiting = fetcher.ensure('model', REPO, background=False)
    assert waiting.state == '同步中'
    assert calls == [['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct', f'--username={USER}',
                      f'--token={TOKEN}']]
    done = fetcher.refresh('model', REPO, background=False)
    assert done.state == '就绪'
    assert done.message == '已在本地'
    assert calls[1] == ['hope', 'list', '--creator=all', '--name_like=Qwen2.5-7B-Instruct']
    assert calls[2][0:3] == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    calls.clear()
    again = fetcher.ensure('model', REPO)
    assert again.state == '就绪'
    assert calls == []
    assert TOKEN not in json.dumps(done.as_row(), ensure_ascii=False)


def test_missing_hope_binary_is_a_failure(tmp_path, hope_env, capsys):

    def runner(*_args, **_kwargs):
        raise FileNotFoundError('hope')

    status = _fetcher(tmp_path, runner).ensure('model', REPO, background=False)
    assert status.state == '失败'
    assert status.message == '下载工具不可用'
    assert 'pip3 install hope==3.6.6' in capsys.readouterr().err
    assert TOKEN not in status.message


def test_modelscope_ids_are_blocked_and_config_file_is_overridden_by_env(tmp_path, monkeypatch):
    monkeypatch.delenv('RSI_MODEL_SOURCE', raising=False)
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text('model_source: hope\nmodel_root: /data/models\ndata_root: /data/datasets\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    assert model_source() == 'hope'
    fetcher = ModelFetcher(tmp_path / 'home', runner=lambda *args, **kwargs: None)
    assert fetcher.model_root() == Path('/data/models')
    assert fetcher.data_root() == Path('/data/datasets')
    assert '不能在当前网络下载，请改用本地文件' in dataset_block_message('modelscope/gsm8k')
    assert '不能在当前网络下载，请改用本地文件' in dataset_block_message('AI-ModelScope/alpaca-gpt4-data-zh')
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'local')
    assert model_source() == 'local'
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'nope')
    assert model_source() == 'modelscope'
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'modelscope')
    assert dataset_block_message('modelscope/gsm8k') == ''
    monkeypatch.delenv('RSI_MODEL_SOURCE', raising=False)
    monkeypatch.delenv('RSI_FETCH_CONFIG', raising=False)
    assert model_source() == 'hope'
    assert '不能在当前网络下载，请改用本地文件' in dataset_block_message('modelscope/gsm8k')


def test_bundled_fetch_yaml_defaults_to_hope(monkeypatch):
    monkeypatch.delenv('RSI_MODEL_SOURCE', raising=False)
    monkeypatch.delenv('RSI_FETCH_CONFIG', raising=False)
    bundled = Path(__file__).resolve().parents[1] / 'configs' / 'fetch.yaml'
    text = bundled.read_text(encoding='utf-8')
    assert 'model_source: hope' in text
    assert model_source() == 'hope'


def test_default_source_does_not_call_hope_or_force_offline(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'modelscope')
    monkeypatch.delenv('RSI_FETCH_CONFIG', raising=False)
    calls = []
    status = _fetcher(tmp_path, lambda *args, **kwargs: calls.append(args)).ensure('model', REPO)
    assert status.state == '保持'
    assert status.local_path == REPO
    assert calls == []
    env = apply_runtime_env({})
    assert 'HF_HUB_OFFLINE' not in env
    assert 'USE_HF' not in env


def test_runtime_env_offline_and_strips_credentials(monkeypatch):
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    monkeypatch.setenv('RSI_HOPE_TOKEN', TOKEN)
    monkeypatch.setenv('RSI_HOPE_USERNAME', USER)
    argv, env = swift_launch(['swift', 'sft', '--model', '/data/models/Qwen/Qwen2.5-7B-Instruct'])
    assert argv[1:4] == ['-m', 'swift.cli.main', 'sft']
    assert env['HF_HUB_OFFLINE'] == '1'
    assert env['TRANSFORMERS_OFFLINE'] == '1'
    assert env['HF_DATASETS_OFFLINE'] == '1'
    assert TOKEN not in env.values()
    assert USER not in env.values()
    assert TOKEN not in ' '.join(argv)
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hf')
    _argv, hf_env = swift_launch(['swift', 'sft', '--model', REPO])
    assert hf_env['USE_HF'] == '1'
    assert 'HF_HUB_OFFLINE' not in hf_env
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'local')
    local_env = apply_runtime_env({})
    assert local_env['HF_DATASETS_OFFLINE'] == '1'


def _client(tmp_path, runner):
    from rsi.app import create_app
    app = create_app(tmp_path / 'home')
    app.state.pipeline.fetcher.runner = runner
    return TestClient(app), app


def test_pipeline_uses_local_model_and_keeps_type(tmp_path, hope_env):
    calls = []
    dest = tmp_path / 'models' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    dest.mkdir(parents=True)
    (dest / 'config.json').write_text('{}', encoding='utf-8')
    client, _app = _client(tmp_path, lambda *args, **kwargs: calls.append(args))
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '内网模型',
            'scenario': '把 hub id 换成本地目录',
            'base_model': REPO,
        }).json()['run']['id']
        api.post(f'/api/runs/{run_id}/stages/data/run', json={'params': {'source': 'bundled', 'bundled': 'itops_mix'}})
        preview = api.post(f'/api/runs/{run_id}/stages/sft/preview', json={'params': {}})
        assert preview.status_code == 200, preview.text
        body = preview.json()
        assert body['params']['model'] == str(dest)
        assert body['params']['model_type'] == 'qwen2'
        assert body['params']['template'] == 'qwen2_5'
        command = body['command']
        assert f'--model {dest}' in command
        assert '--model_type qwen2\n' in command or '--model_type qwen2 \\' in command
        assert '--template qwen2_5' in command
        assert 'huggingface.co' not in command
        assert TOKEN not in command
        assert USER not in command
        assert calls == []
        schema = api.get('/api/schema').json()
        assert schema['model_source'] == 'hope'
        assert 'openai/gsm8k' in schema['catalog']['hf_datasets']
        assert all(not item.startswith('modelscope/') and not item.startswith('AI-ModelScope/')
                   for item in schema['catalog']['hf_datasets'])


def test_pipeline_syncing_audit_and_modelscope_error(tmp_path, hope_env):
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), kwargs.get('cwd')))
        return Completed(stdout='submitted, pending')

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '同步中',
            'scenario': '外网同步还没完成',
            'base_model': REPO,
        }).json()['run']['id']
        blocked = api.post('/api/runs/' + run_id + '/stages/data/run', json={
            'params': {'source': 'public', 'dataset': 'modelscope/gsm8k'},
        })
        assert blocked.status_code == 400
        assert blocked.json()['detail'] == 'modelscope/gsm8k 不能在当前网络下载，请改用本地文件'
        assert calls == []
        syncing = api.post(f'/api/runs/{run_id}/stages/data/run', json={
            'params': {'source': 'public', 'dataset': 'openai/gsm8k'},
        })
        assert syncing.status_code == 200, syncing.text
        body = syncing.json()
        assert body['status'] == 'syncing'
        assert body['command'] == ''
        blob = json.dumps(body, ensure_ascii=False)
        assert '同步中' in blob
        assert TOKEN not in blob
        assert USER not in blob
        assert 'huggingface.co' not in blob
        assert calls[0][0][:3] == ['hope', 'clone', 'https://huggingface.co/datasets/openai/gsm8k']
        assert calls[0][1] == str(tmp_path / 'datasets' / 'openai')
        view = body['view']
        assert view['forms']['data']['status'] == 'syncing'
        assert view['fetches'][0]['state'] == '同步中'
        assert view['fetches'][0]['repo_id'] == 'openai/gsm8k'


def test_eval_prefers_local_benchmark_and_keeps_general_qa(tmp_path, hope_env):
    calls = []
    model_dir = tmp_path / 'models' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    model_dir.mkdir(parents=True)
    (model_dir / 'config.json').write_text('{}', encoding='utf-8')
    bench = tmp_path / 'datasets' / 'benchmarks' / 'gsm8k'
    bench.mkdir(parents=True)
    (bench / 'test.jsonl').write_text('{}\n', encoding='utf-8')
    client, _app = _client(tmp_path, lambda *args, **kwargs: calls.append(args))
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '评测本地',
            'scenario': '评测集放在本机',
            'base_model': REPO,
        }).json()['run']['id']
        assert api.post(f'/api/runs/{run_id}/stages/data/run', json={
            'params': {'source': 'bundled', 'bundled': 'itops_mix'},
        }).status_code == 200
        assert api.post(f'/api/runs/{run_id}/stages/sft/run', json={'params': {}}).status_code == 200
        pointed = tmp_path / 'my-gsm8k'
        pointed.mkdir()
        (pointed / 'gsm8k.jsonl').write_text('{}\n', encoding='utf-8')
        preview = api.post(f'/api/runs/{run_id}/stages/sft_eval/preview', json={'params': {
            'eval_dataset': 'gsm8k',
            'eval_dataset_args': {'gsm8k': {'local_path': str(pointed)}},
        }})
        assert preview.status_code == 200, preview.text
        assert preview.json()['params']['eval_dataset_args']['gsm8k']['local_path'] == str(pointed)
        assert str(pointed) in preview.json()['command']
        fallback = api.post(f'/api/runs/{run_id}/stages/sft_eval/preview', json={'params': {'eval_dataset': 'gsm8k'}})
        assert fallback.json()['params']['eval_dataset_args']['gsm8k']['local_path'] == str(bench)
        missing = api.post(f'/api/runs/{run_id}/stages/sft_eval/run', json={'params': {'eval_dataset': 'ceval'}})
        assert missing.status_code == 400
        assert missing.json()['detail'] == '评测集 ceval 需要放在本地后再运行'
        assert 'evalscope' not in missing.json()['detail']
        assert calls == []


def test_hope_with_swift_backend_keeps_general_qa(tmp_path, hope_env, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'swift')
    model_dir = tmp_path / 'models' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    model_dir.mkdir(parents=True)
    (model_dir / 'config.json').write_text('{}', encoding='utf-8')
    def reject_spawn(*_args, **_kwargs):
        raise AssertionError('preview must not spawn')

    client, _app = _client(tmp_path, reject_spawn)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '真实评测参数',
            'scenario': '内网仍用本地 general_qa',
            'base_model': REPO,
        }).json()['run']['id']
        preview = api.post(f'/api/runs/{run_id}/stages/sft_eval/preview', json={'params': {'model': REPO}})
        assert preview.status_code == 200, preview.text
        params = preview.json()['params']
        command = preview.json()['command']
        assert params['eval_dataset'] == 'general_qa'
        assert params['model'] == str(model_dir)
        assert params['model_type'] == 'qwen2'
        assert params['template'] == 'qwen2_5'
        qa = Path(params['eval_dataset_args']['general_qa']['local_path'])
        assert (qa / 'itops.jsonl').is_file()
        assert 'general_qa' in command
        assert 'itops-heldout' not in command
        assert str(qa) in command
        assert TOKEN not in command


def test_sft_does_not_launch_while_model_is_syncing(tmp_path, hope_env, monkeypatch):
    import subprocess
    import time
    spawned = []

    def boom(*args, **kwargs):
        spawned.append(args)
        raise AssertionError('swift must not start while the model is still syncing')

    monkeypatch.setenv('RSI_BACKEND', 'swift')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    monkeypatch.setattr(subprocess, 'Popen', boom)
    monkeypatch.setattr('rsi.backends.subprocess.Popen', boom)
    monkeypatch.setattr('rsi.backends.subprocess.run', boom)

    def runner(argv, **kwargs):
        return Completed(stdout='submitted, pending')

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '先同步',
            'scenario': '模型还在同步时不能启动训练',
            'base_model': REPO,
        }).json()['run']['id']
        assert api.post(f'/api/runs/{run_id}/stages/data/run', json={
            'params': {'source': 'bundled', 'bundled': 'itops_mix'},
        }).status_code == 200
        blocked = api.post('/api/runs', json={
            'name': '魔搭 id',
            'scenario': '不能在当前网络下载',
            'base_model': 'modelscope/Qwen2.5-7B-Instruct',
        }).json()['run']['id']
        assert api.post(f'/api/runs/{blocked}/stages/data/run', json={
            'params': {'source': 'bundled', 'bundled': 'itops_mix'},
        }).status_code == 200
        rejected = api.post(f'/api/runs/{blocked}/stages/sft/run', json={'params': {}})
        assert rejected.status_code == 400
        assert rejected.json()['detail'] == 'modelscope/Qwen2.5-7B-Instruct 不能在当前网络下载，请改用本地文件'
        started = api.post(f'/api/runs/{run_id}/stages/sft/run', json={'params': {}})
        assert started.status_code == 200, started.text
        body = started.json()
        assert body['status'] == 'syncing'
        assert body['command'] == ''
        step = [item for item in body['view']['steps'] if item['stage'] == 'sft'][-1]
        assert step['status'] == 'syncing'
        model = step['params']['model']
        assert model != REPO
        assert Path(model).is_absolute()
        assert str(tmp_path / 'models') in model
        preview = api.post(f'/api/runs/{run_id}/stages/sft/preview', json={'params': {}})
        assert preview.status_code == 200, preview.text
        command = preview.json()['command']
        assert f'--model {REPO}' not in command
        assert f'--model {preview.json()["params"]["model"]}' in command
        assert Path(preview.json()['params']['model']).is_absolute()
    time.sleep(0.2)
    assert spawned == []


def test_refresh_endpoint_and_local_source(tmp_path, hope_env):
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == 'clone':
            dest = Path(kwargs['cwd']) / 'Qwen2.5-7B-Instruct'
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'config.json').write_text('{}', encoding='utf-8')
        return Completed()

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '重新拉取',
            'scenario': '检查同步是否完成',
            'base_model': REPO,
        }).json()['run']['id']
        refreshed = api.post(f'/api/runs/{run_id}/fetch', json={'model': REPO, 'dataset': 'modelscope/gsm8k'})
        assert refreshed.status_code == 400
        assert calls == []
        refreshed = api.post(f'/api/runs/{run_id}/fetch', json={'model': REPO, 'dataset': 'openai/gsm8k'})
        assert refreshed.status_code == 200, refreshed.text
        rows = {row['kind']: row for row in refreshed.json()['fetches']}
        assert rows['model']['state'] == '就绪'
        assert rows['dataset']['state'] == '同步中'
        blob = refreshed.text
        assert TOKEN not in blob
        assert 'https://huggingface.co' not in blob
        assert any(item[:2] == ['hope', 'list'] for item in calls)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'local')
    try:
        quiet = []
        status = _fetcher(tmp_path, lambda *args, **kwargs: quiet.append(args)).ensure('model', 'org/missing')
        assert status.state == '失败'
        assert status.message == '本地没有该文件'
        assert quiet == []
    finally:
        monkeypatch.undo()


def test_upsert_fetch_persists_status_without_on_conflict(tmp_path, hope_env):
    from rsi.store import Store
    store = Store(tmp_path / 'direct.sqlite')
    first = {
        'kind': 'model',
        'repo_id': REPO,
        'local_path': str(tmp_path / 'models' / 'Qwen' / 'Qwen2.5-7B-Instruct'),
        'state': '同步中',
        'message': '同步中',
    }
    store.upsert_fetch(first)
    store.upsert_fetch({**first, 'state': '就绪', 'message': '已在本地'})
    rows = store.list_fetches()
    assert len(rows) == 1
    assert rows[0]['kind'] == 'model'
    assert rows[0]['repo_id'] == REPO
    assert rows[0]['state'] == '就绪'
    assert rows[0]['message'] == '已在本地'
    assert rows[0]['local_path'] == first['local_path']
    source = Path(__file__).resolve().parents[1].joinpath('store.py').read_text(encoding='utf-8')
    code = '\n'.join(line for line in source.splitlines() if not line.strip().startswith('#'))
    assert 'ON CONFLICT' not in code

    def runner(argv, **kwargs):
        return Completed(stdout='submitted, pending')

    client, app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '记下同步',
            'scenario': '不下载也能记下 fetch 状态',
            'base_model': REPO,
        }).json()['run']['id']
        assert api.post(f'/api/runs/{run_id}/stages/data/run', json={
            'params': {'source': 'bundled', 'bundled': 'itops_mix'},
        }).status_code == 200
        started = api.post(f'/api/runs/{run_id}/stages/sft/run', json={'params': {}})
        assert started.status_code == 200, started.text
        assert started.json()['status'] == 'syncing'
        saved = app.state.pipeline.store.list_fetches()
    assert len(saved) == 1
    assert saved[0]['kind'] == 'model'
    assert saved[0]['repo_id'] == REPO
    assert saved[0]['state'] == '同步中'


def test_run_sh_exits_when_hope_cli_is_missing():
    import os
    import shutil
    import subprocess
    env = os.environ.copy()
    env.pop('RSI_MODEL_SOURCE', None)
    env.pop('RSI_FETCH_CONFIG', None)
    env['PATH'] = '/usr/bin:/bin'
    if shutil.which('hope', path=env['PATH']):
        return
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ['bash', 'run.sh', 'start'], cwd=root, env=env, capture_output=True, text=True, timeout=20, check=False)
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert '未找到 hope' in output
    assert 'pip3 install hope==3.6.6' in output


def test_run_sh_does_not_install_hope():
    text = Path(__file__).resolve().parents[2].joinpath('run.sh').read_text(encoding='utf-8')
    assert 'command -v hope' in text
    assert 'fetch.yaml' in text
    assert 'pip3 install hope==3.6.6' in text
    assert 'exit 1' in text
    executable = '\n'.join(line for line in text.splitlines() if not line.strip().startswith('echo '))
    assert 'pip3 install hope' not in executable
    readme = Path(__file__).resolve().parents[1].joinpath('README.md').read_text(encoding='utf-8')
    assert '内网部署' in readme
    assert 'RSI_MODEL_SOURCE' in readme
    assert 'RSI_MODEL_SOURCE=modelscope' in readme
    assert 'RSI_HOPE_TOKEN' in readme
    page = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert '模型来源：' in page


PASSWORD_PROMPT = 'Warning: Password input may be echoed. mis密码(非jumper密码):'


def _hope_handler():
    found = [handler for handler in model_fetch.logger.handlers if getattr(handler, '_rsi_hope', False)]
    assert found, 'hope log handler missing'
    return found[0]


def test_hope_clone_and_list_log_redacted_commands(tmp_path, hope_env, caplog, monkeypatch):
    monkeypatch.setenv('RSI_HOPE_PARAMS', 'secret-params')
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    seen = []

    def runner(argv, **kwargs):
        seen.append((list(argv), dict(kwargs)))
        if argv[1] == 'list':
            stdout = f'mis无关 token={TOKEN} user={USER} params=secret-params\n' + ('已存在，等待同步 ' * 400)
            return Completed(stdout=stdout)
        return Completed(stdout=f'任务已存在，请等待同步 token={TOKEN} user={USER}')

    fetcher = _fetcher(tmp_path, runner)
    waiting = fetcher.ensure('model', REPO, background=False)
    assert waiting.state == '同步中'
    listed = fetcher.refresh('model', REPO, background=False)
    assert listed.state == '同步中'
    text = caplog.text
    assert 'hope 开始 hope clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct ' in text
    assert 'hope 开始 hope list --creator=all --name_like=Qwen2.5-7B-Instruct ' in text
    assert 'exit=0' in text
    assert f'cwd={tmp_path / "models" / "Qwen"}' in text
    assert f'target={tmp_path / "models" / "Qwen" / "Qwen2.5-7B-Instruct"}' in text
    assert TOKEN not in text
    assert USER not in text
    assert 'secret-params' not in text
    assert '--token=' not in text
    assert '--username=' not in text
    assert '--params=' not in text
    assert '***' in text
    assert '…截断' in text
    clone_kwargs = seen[0][1]
    list_kwargs = seen[1][1]
    assert clone_kwargs['stdin'] is subprocess.DEVNULL
    assert 'timeout' not in clone_kwargs
    assert list_kwargs['stdin'] is subprocess.DEVNULL
    assert list_kwargs['timeout'] == model_fetch.LIST_TIMEOUT
    assert seen[0][0][:3] == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    assert f'--token={TOKEN}' in seen[0][0]
    formatted = _hope_handler().format(caplog.records[0])
    assert re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', formatted)
    assert '%(asctime)s' in _hope_handler().formatter._fmt


def test_password_prompt_fails_fetch_and_does_not_inherit_stdin(tmp_path, hope_env, caplog):
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    seen = []

    def runner(argv, **kwargs):
        seen.append(dict(kwargs))
        dest = Path(kwargs['cwd']) / 'Qwen2.5-7B-Instruct'
        dest.mkdir(parents=True, exist_ok=True)
        return Completed(returncode=1, stderr=PASSWORD_PROMPT)

    status = _fetcher(tmp_path, runner).ensure('model', REPO, background=False)
    assert status.state == '失败'
    assert status.message == LOGIN_MESSAGE
    assert 'RSI_HOPE_USERNAME' in status.message
    assert 'RSI_HOPE_TOKEN' in status.message
    assert TOKEN not in status.message
    assert USER not in status.message
    assert seen[0]['stdin'] is subprocess.DEVNULL
    assert 'timeout' not in seen[0]
    assert 'hope 需要登录' in caplog.text
    assert TOKEN not in caplog.text


def test_list_password_prompt_stops_before_clone(tmp_path, hope_env):
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return Completed(returncode=1, stdout='请输入密码')

    status = _fetcher(tmp_path, runner).refresh('model', REPO, background=False)
    assert status.state == '失败'
    assert status.message == LOGIN_MESSAGE
    assert len(calls) == 1
    assert calls[0][0][1] == 'list'
    assert calls[0][1]['stdin'] is subprocess.DEVNULL
    assert calls[0][1]['timeout'] == model_fetch.LIST_TIMEOUT
    assert TOKEN not in status.message


def test_ready_and_inflight_skips_are_logged(tmp_path, hope_env, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    dest = tmp_path / 'models' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    dest.mkdir(parents=True)
    (dest / 'config.json').write_text('{}', encoding='utf-8')

    def refuse(*_args, **_kwargs):
        raise AssertionError('ready model should not call hope')

    ready = _fetcher(tmp_path, refuse).ensure('model', REPO)
    assert ready.state == '就绪'
    assert 'hope 跳过 model Qwen/Qwen2.5-7B-Instruct 已就绪' in caplog.text

    monkeypatch.delenv('RSI_FETCH_INLINE', raising=False)
    caplog.clear()
    started = threading.Event()
    release = threading.Event()

    def runner(argv, **kwargs):
        started.set()
        assert release.wait(3)
        return Completed(stdout='submitted, pending')

    (dest / 'config.json').unlink()
    fetcher = _fetcher(tmp_path, runner)
    try:
        first = fetcher.ensure('model', REPO, background=True)
        assert first.state == '同步中'
        assert started.wait(3)
        second = fetcher.ensure('model', REPO, background=True)
        assert second.state == '同步中'
        assert '已在进行' in caplog.text
    finally:
        release.set()


def test_clone_ready_flip_is_logged(tmp_path, hope_env, caplog):
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')

    def runner(argv, **kwargs):
        dest = Path(kwargs['cwd']) / 'Qwen2.5-7B-Instruct'
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'config.json').write_text('{}', encoding='utf-8')
        return Completed(stdout='done')

    status = _fetcher(tmp_path, runner).ensure('model', REPO, background=False)
    assert status.state == '就绪'
    assert status.message == '已在本地'
    assert 'hope 就绪 model Qwen/Qwen2.5-7B-Instruct' in caplog.text
    assert 'exit=0' in caplog.text


def test_describe_startup_reports_presence_only(monkeypatch, capsys, caplog):
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    monkeypatch.setenv('RSI_HOPE_TOKEN', 'super-secret')
    monkeypatch.delenv('RSI_HOPE_USERNAME', raising=False)
    monkeypatch.delenv('RSI_HOPE_PARAMS', raising=False)
    text = describe_startup()
    assert text == 'model_source=hope hope_username=no hope_token=yes'
    assert 'super-secret' not in text
    monkeypatch.setenv('RSI_HOPE_USERNAME', 'alice')
    monkeypatch.delenv('RSI_HOPE_TOKEN', raising=False)
    present = describe_startup()
    assert present == 'model_source=hope hope_username=yes hope_token=no'
    assert 'alice' not in present

    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    import rsi.__main__ as entry
    app = SimpleNamespace(state=SimpleNamespace(pipeline=SimpleNamespace(backend=SimpleNamespace(name='mock'))))
    monkeypatch.setattr(entry, 'create_app', lambda: app)
    monkeypatch.setattr(entry, 'mount_swift_ui', lambda _app: True)
    monkeypatch.setattr(entry.uvicorn, 'run', lambda *args, **kwargs: None)
    monkeypatch.setenv('RSI_HOPE_TOKEN', 'super-secret')
    monkeypatch.delenv('RSI_HOPE_USERNAME', raising=False)
    entry.main()
    captured = capsys.readouterr()
    combined = captured.out + captured.err + caplog.text
    assert 'RSI 启动 model_source=hope hope_username=no hope_token=yes' in combined
    assert 'super-secret' not in combined
    assert 'alice' not in combined
