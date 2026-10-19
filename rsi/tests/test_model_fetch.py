# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rsi.model_fetch as model_fetch
from rsi.backends import swift_launch
from rsi.model_fetch import (
    ModelFetcher,
    apply_runtime_env,
    clone_argv,
    dataset_block_message,
    model_source,
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
    monkeypatch.delenv('RSI_MODEL_SOURCE', raising=False)
    monkeypatch.delenv('RSI_FETCH_CONFIG', raising=False)
    assert dataset_block_message('modelscope/gsm8k') == ''


def test_default_source_does_not_call_hope_or_force_offline(tmp_path, monkeypatch):
    monkeypatch.delenv('RSI_MODEL_SOURCE', raising=False)
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


def test_run_sh_does_not_install_hope():
    text = Path(__file__).resolve().parents[2].joinpath('run.sh').read_text(encoding='utf-8')
    assert 'command -v hope' in text
    assert 'pip3 install hope==3.6.6' in text
    executable = '\n'.join(line for line in text.splitlines() if not line.strip().startswith('echo '))
    assert 'pip3 install hope' not in executable
    readme = Path(__file__).resolve().parents[1].joinpath('README.md').read_text(encoding='utf-8')
    assert '内网部署' in readme
    assert 'RSI_MODEL_SOURCE' in readme
    assert 'RSI_HOPE_TOKEN' in readme
