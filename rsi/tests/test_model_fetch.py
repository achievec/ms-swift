# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import rsi.model_fetch as model_fetch
from rsi.backends import swift_launch
from rsi.model_fetch import (
    LOGIN_WAITING,
    ModelFetcher,
    apply_runtime_env,
    clone_argv,
    dataset_block_message,
    dataset_url,
    describe_startup,
    hope_clone_dir,
    hope_login_succeeded,
    hope_needs_login,
    kick_hope_login,
    login_unconfirmed_message,
    model_source,
    public_clone_command,
)


TOKEN = 'super-secret-token'
USER = 'rsi-hope-user'
REPO = 'Qwen/Qwen2.5-7B-Instruct'
MS_DATASET = 'AI-ModelScope/alpaca-gpt4-data-zh'
MS_DATASET_URL = 'https://www.modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh.git'


def _ready_ms_dataset(root: Path, repo=MS_DATASET) -> Path:
    org, name = repo.split('/', 1)
    dest = Path(root) / 'www.modelscope.cn' / 'datasets' / org / name
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / 'train.jsonl'
    if not marker.is_file():
        marker.write_text('{"instruction": "q", "output": "a"}\n', encoding='utf-8')
    return dest


class Completed:

    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _reset_fetch_globals():
    model_fetch._INFLIGHT.clear()
    model_fetch._HOPE_WARNED = False
    model_fetch.reset_hope_login_state()
    yield
    model_fetch._INFLIGHT.clear()
    model_fetch.reset_hope_login_state()


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


def test_huggingface_co_layout_is_ready_and_used_as_model(tmp_path, hope_env):
    model_dir = tmp_path / 'models' / 'huggingface.co' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    model_dir.mkdir(parents=True)
    (model_dir / 'config.json').write_text('{}', encoding='utf-8')
    data_dir = tmp_path / 'datasets' / 'huggingface.co' / 'datasets' / 'openai' / 'gsm8k'
    data_dir.mkdir(parents=True)
    (data_dir / 'train.jsonl').write_text('{"instruction": "q", "output": "a"}\n', encoding='utf-8')
    _ready_ms_dataset(tmp_path / 'datasets')
    legacy = tmp_path / 'models' / 'Qwen' / 'Qwen2.5-7B-Instruct'

    def refuse(*_args, **_kwargs):
        raise AssertionError('a repo that is already on disk must not call hope')

    fetcher = _fetcher(tmp_path, refuse)
    status = fetcher.ensure('model', REPO)
    assert status.state == '就绪'
    assert status.message == '已在本地'
    assert status.local_path == str(model_dir)
    assert not legacy.exists()
    dataset = fetcher.ensure('dataset', 'openai/gsm8k')
    assert dataset.state == '就绪'
    assert dataset.local_path == str(data_dir)

    legacy.mkdir(parents=True)
    (legacy / 'config.json').write_text('{}', encoding='utf-8')
    both = fetcher.status_of('model', REPO)
    assert both.state == '就绪'
    assert both.local_path == str(model_dir)

    client, _app = _client(tmp_path, refuse)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': 'hope 目录',
            'scenario': 'clone 落在 huggingface.co 下',
            'base_model': REPO,
        }).json()['run']['id']
        preview = api.post(f'/api/runs/{run_id}/stages/sft/preview', json={'params': {}})
        assert preview.status_code == 200, preview.text
        assert preview.json()['params']['model'] == str(model_dir)
        assert f'--model {model_dir}' in preview.json()['command']
        loaded = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'public', 'dataset': 'openai/gsm8k'},
        })
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()['result']['dataset'] == str(data_dir)


def test_ready_huggingface_co_repo_skips_clone_until_refetch(tmp_path, hope_env):
    model_dir = tmp_path / 'models' / 'huggingface.co' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    model_dir.mkdir(parents=True)
    (model_dir / 'config.json').write_text('{}', encoding='utf-8')
    data_dir = tmp_path / 'datasets' / 'huggingface.co' / 'datasets' / 'openai' / 'gsm8k'
    data_dir.mkdir(parents=True)
    (data_dir / 'train.jsonl').write_text('{}\n', encoding='utf-8')
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == 'login':
            return Completed(stdout='登录成功')
        return Completed(stdout='submitted, pending')

    fetcher = _fetcher(tmp_path, runner)
    status = fetcher.ensure('model', REPO, background=False)
    assert status.state == '就绪'
    assert status.message == '已在本地'
    assert status.local_path == str(model_dir)
    looked = fetcher.status_of('model', REPO)
    assert looked.state == '就绪'
    assert looked.local_path == str(model_dir)
    dataset = fetcher.ensure('dataset', 'openai/gsm8k', background=False)
    assert dataset.state == '就绪'
    assert dataset.local_path == str(data_dir)
    assert calls == []

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '已在本地',
            'scenario': '有文件就不要再 clone',
            'base_model': REPO,
        }).json()['run']['id']
        viewed = api.get(f'/api/runs/{run_id}')
        assert viewed.status_code == 200
        assert calls == []
        forced = api.post(f'/api/runs/{run_id}/fetch', json={'model': REPO, 'dataset': 'openai/gsm8k'})
        assert forced.status_code == 200, forced.text
        kinds = [item[1] for item in calls]
        assert kinds.count('clone') == 2
        assert 'list' not in kinds
        rows = {row['kind']: row for row in forced.json()['fetches']}
        assert rows['model']['state'] == '就绪'
        assert rows['model']['local_path'] == str(model_dir)
        assert rows['dataset']['state'] == '就绪'
        assert rows['dataset']['local_path'] == str(data_dir)
    calls.clear()
    skipped = fetcher.ensure('model', REPO, background=False)
    assert skipped.state == '就绪'
    assert skipped.local_path == str(model_dir)
    assert calls == []


def test_clone_into_huggingface_co_flips_the_fetch_row(tmp_path, hope_env):
    def runner(argv, **kwargs):
        if argv[1] == 'login':
            return Completed(stdout='登录成功')
        cwd = Path(kwargs['cwd'])
        if '/datasets/' in argv[2]:
            dest = cwd / 'huggingface.co' / 'datasets' / 'openai' / 'gsm8k'
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'train.jsonl').write_text('{}\n', encoding='utf-8')
            return Completed(stdout=f'模型下载成功\n数据存储在: {dest}')
        dest = cwd / 'huggingface.co' / 'Qwen' / 'Qwen2.5-7B-Instruct'
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'config.json').write_text('{}', encoding='utf-8')
        return Completed(stdout=f'模型下载成功\n数据存储在: {dest}')

    model_dir = tmp_path / 'models' / 'huggingface.co' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    data_dir = tmp_path / 'datasets' / 'huggingface.co' / 'datasets' / 'openai' / 'gsm8k'
    fetcher = _fetcher(tmp_path, runner)
    done = fetcher.ensure('model', REPO, background=False)
    assert done.state == '就绪'
    assert done.local_path == str(model_dir)
    assert (model_dir / 'config.json').is_file()
    dataset = fetcher.ensure('dataset', 'openai/gsm8k', background=False)
    assert dataset.state == '就绪'
    assert dataset.local_path == str(data_dir)

    calls = []

    def pending(argv, **kwargs):
        calls.append(list(argv))
        return Completed(stdout='submitted, pending')

    client, _app = _client(tmp_path, pending)
    legacy = tmp_path / 'models' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    if model_dir.exists():
        for child in model_dir.iterdir():
            child.unlink()
        model_dir.rmdir()
        model_dir.parent.rmdir()
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '等 config',
            'scenario': '文件出现在 huggingface.co 后才就绪',
            'base_model': REPO,
        }).json()['run']['id']
        waiting = api.post(f'/api/runs/{run_id}/fetch', json={'model': REPO})
        assert waiting.status_code == 200, waiting.text
        row = next(item for item in waiting.json()['fetches'] if item['kind'] == 'model')
        assert row['state'] == '同步中'
        assert not legacy.exists()
        model_dir.mkdir(parents=True)
        (model_dir / 'config.json').write_text('{}', encoding='utf-8')
        again = api.get(f'/api/runs/{run_id}')
        ready = next(item for item in again.json()['fetches'] if item['kind'] == 'model')
        assert ready['state'] == '就绪'
        assert ready['local_path'] == str(model_dir)
        assert ready['message'] == '已在本地'


def test_syncing_submits_clone_without_leaking_credentials(tmp_path, hope_env):
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), kwargs.get('cwd')))
        return Completed(stdout=f'任务已存在，请等待同步 token={TOKEN}')

    status = _fetcher(tmp_path, runner).ensure('model', REPO, background=False)
    assert status.state == '同步中'
    assert status.message == '同步中'
    assert calls[0][0] == ['hope', 'login', USER]
    argv, cwd = calls[1]
    assert argv[:3] == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    assert f'--username={USER}' in argv
    assert f'--token={TOKEN}' in argv
    assert cwd == str(tmp_path / 'models')
    assert (tmp_path / 'models').is_dir()
    assert not (tmp_path / 'models' / 'Qwen').exists()
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


def test_tilde_model_root_is_the_clone_cwd(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    house = tmp_path / 'house'
    house.mkdir()
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text('model_source: hope\nmodel_root: ~/models\ndata_root: $HOME/datasets\n', encoding='utf-8')
    monkeypatch.setenv('HOME', str(house))
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    monkeypatch.setenv('RSI_FETCH_INLINE', '1')
    monkeypatch.setenv('RSI_HOPE_USERNAME', USER)
    monkeypatch.delenv('RSI_MODEL_ROOT', raising=False)
    monkeypatch.delenv('RSI_DATA_ROOT', raising=False)
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), kwargs.get('cwd')))
        if argv[1] == 'login':
            return Completed(stdout='登录成功')
        return Completed(stdout='submitted, pending')

    fetcher = ModelFetcher(tmp_path / 'rsi-home', runner=runner)
    assert fetcher.model_root() == house / 'models'
    assert fetcher.data_root() == house / 'datasets'
    assert fetcher.model_root().is_absolute()
    status = fetcher.ensure('model', REPO, background=False)
    root = house / 'models'
    assert root.is_dir()
    assert not (root / 'Qwen').exists()
    clone = next(item for item in calls if item[0][1] == 'clone')
    assert clone[0][:3] == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    assert clone[1] == str(root)
    assert status.local_path == str(root / 'Qwen' / 'Qwen2.5-7B-Instruct')
    assert status.state == '同步中'
    text = caplog.text
    assert f'cwd={root} target={root / "Qwen" / "Qwen2.5-7B-Instruct"}' in text
    assert 'hope 开始 hope clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct ' in text
    assert f'cwd={root / "Qwen"}' not in text
    monkeypatch.setenv('RSI_MODEL_ROOT', '~/other-models')
    assert fetcher.model_root() == house / 'other-models'


def test_sync_status_comes_from_local_files_not_hope_list(tmp_path, hope_env):
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        assert argv[1] != 'list'
        return Completed(stdout='submitted, pending')

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '看进度',
            'scenario': '同步状态不看 hope list',
            'base_model': REPO,
        }).json()['run']['id']
        refreshed = api.post(f'/api/runs/{run_id}/fetch', json={'model': REPO})
        assert refreshed.status_code == 200, refreshed.text
        row = next(item for item in refreshed.json()['fetches'] if item['kind'] == 'model')
        assert row['state'] == '同步中'
        assert row['repo_id'] == REPO
        assert row['message'] == '同步中'
        assert '8%' not in row['message']
        assert TOKEN not in row['message']
        assert all(item[1] != 'list' for item in calls)
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
    assert '进度见 rsi/.rsi.log' in page
    assert 'hope list' not in page


def test_refresh_lists_then_clone_completes(tmp_path, hope_env):
    calls = []

    clones = {'n': 0}

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == 'login':
            return Completed(stdout='登录成功')
        if argv[1] == 'clone':
            clones['n'] += 1
            if clones['n'] > 1:
                dest = Path(kwargs['cwd']) / 'Qwen' / 'Qwen2.5-7B-Instruct'
                dest.mkdir(parents=True, exist_ok=True)
                (dest / 'config.json').write_text('{"model_type":"qwen2"}', encoding='utf-8')
        return Completed(returncode=1, stdout='已存在，等待同步')

    fetcher = _fetcher(tmp_path, runner)
    waiting = fetcher.ensure('model', REPO, background=False)
    assert waiting.state == '同步中'
    assert calls[0] == ['hope', 'login', USER]
    assert calls[1] == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct', f'--username={USER}',
                        f'--token={TOKEN}']
    done = fetcher.refresh('model', REPO, background=False)
    assert done.state == '就绪'
    assert done.message == '已在本地'
    assert calls[2][0:3] == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    assert all(item[1] != 'list' for item in calls)
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
    assert dataset_block_message('modelscope/gsm8k', kind='dataset') == ''
    assert dataset_block_message(MS_DATASET, kind='dataset') == ''
    assert dataset_block_message(MS_DATASET + '#500', kind='dataset') == ''
    assert '不能在当前网络下载，请改用本地文件' in dataset_block_message('modelscope/Qwen2.5-7B-Instruct')
    assert '不能在当前网络下载，请改用本地文件' in dataset_block_message(MS_DATASET)
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'local')
    assert model_source() == 'local'
    assert '不能在当前网络下载，请改用本地文件' in dataset_block_message(MS_DATASET, kind='dataset')
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'nope')
    assert model_source() == 'modelscope'
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'modelscope')
    assert dataset_block_message('modelscope/gsm8k', kind='dataset') == ''
    monkeypatch.delenv('RSI_MODEL_SOURCE', raising=False)
    monkeypatch.delenv('RSI_FETCH_CONFIG', raising=False)
    assert model_source() == 'hope'
    assert dataset_block_message(MS_DATASET, kind='dataset') == ''
    assert '不能在当前网络下载，请改用本地文件' in dataset_block_message('modelscope/Qwen2.5-7B-Instruct')


def test_modelscope_dataset_url_uses_data_root_layout():
    assert dataset_url(MS_DATASET) == MS_DATASET_URL
    assert dataset_url('iic/ms_bench') == 'https://www.modelscope.cn/datasets/iic/ms_bench.git'
    assert dataset_url('openai/gsm8k') == 'https://huggingface.co/datasets/openai/gsm8k'
    root = Path('/data/datasets')
    assert hope_clone_dir(root, 'dataset', MS_DATASET) == (
        root / 'www.modelscope.cn' / 'datasets' / 'AI-ModelScope' / 'alpaca-gpt4-data-zh')
    assert hope_clone_dir(root, 'dataset', 'openai/gsm8k') == (
        root / 'huggingface.co' / 'datasets' / 'openai' / 'gsm8k')
    assert hope_clone_dir(Path('/data/models'), 'model', REPO) == (
        Path('/data/models') / 'huggingface.co' / 'Qwen' / 'Qwen2.5-7B-Instruct')


def test_modelscope_dataset_logs_in_then_clones_and_skips_when_present(tmp_path, hope_env):
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), kwargs.get('cwd')))
        if argv[1] == 'login':
            return Completed(stdout='登录成功')
        if argv[1] == 'clone' and argv[2] == MS_DATASET_URL:
            dest = Path(kwargs['cwd']) / 'www.modelscope.cn' / 'datasets' / 'AI-ModelScope' / 'alpaca-gpt4-data-zh'
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'train.jsonl').write_text('{}\n', encoding='utf-8')
            return Completed(stdout=f'数据存储在: {dest}')
        raise AssertionError(argv)

    fetcher = _fetcher(tmp_path, runner)
    done = fetcher.ensure('dataset', MS_DATASET, background=False)
    assert done.state == '就绪'
    assert done.local_path == str(
        tmp_path / 'datasets' / 'www.modelscope.cn' / 'datasets' / 'AI-ModelScope' / 'alpaca-gpt4-data-zh')
    assert calls[0][0] == ['hope', 'login', USER]
    assert calls[0][1] == str(tmp_path / 'datasets')
    assert calls[1][0][:3] == ['hope', 'clone', MS_DATASET_URL]
    assert calls[1][1] == str(tmp_path / 'datasets')
    assert any(flag.startswith('--token=') for flag in calls[1][0])
    calls.clear()
    again = fetcher.ensure('dataset', MS_DATASET, background=False)
    assert again.state == '就绪'
    assert again.local_path == done.local_path
    assert calls == []

    git_name = tmp_path / 'datasets' / 'www.modelscope.cn' / 'datasets' / 'modelscope' / 'gsm8k.git'
    git_name.mkdir(parents=True)
    (git_name / 'dataset_info.json').write_text('{}', encoding='utf-8')

    def refuse(*_args, **_kwargs):
        raise AssertionError('a dataset that is already on disk must not log in or clone')

    skipped = _fetcher(tmp_path, refuse).ensure('dataset', 'modelscope/gsm8k', background=False)
    assert skipped.state == '就绪'
    assert skipped.local_path == str(git_name)


def test_dataset_clone_does_not_start_when_login_fails(tmp_path, hope_env):
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == 'login':
            return Completed(returncode=1, stdout='登录失败')
        raise AssertionError('clone must wait for the same hope login as models')

    status = _fetcher(tmp_path, runner).ensure('dataset', MS_DATASET, background=False)
    assert calls == [['hope', 'login', USER]]
    assert status.state == '失败'
    assert status.message == f'hope 登录未确认：请点重新拉取再发一次，或在终端执行 hope login {USER}'


def test_sft_uses_each_modelscope_dataset_and_skips_one_already_present(tmp_path, hope_env):
    model_dir = tmp_path / 'models' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    model_dir.mkdir(parents=True)
    (model_dir / 'config.json').write_text('{}', encoding='utf-8')
    alpaca = _ready_ms_dataset(tmp_path / 'datasets')
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), kwargs.get('cwd')))
        if argv[1] == 'login':
            return Completed(stdout='登录成功')
        return Completed(stdout='submitted, pending')

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '多数据集',
            'scenario': '每个公开 id 各自拉取',
            'base_model': REPO,
        }).json()['run']['id']
        syncing = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'public', 'dataset': f'modelscope/gsm8k {MS_DATASET}'},
        })
        assert syncing.status_code == 200, syncing.text
        assert '不能在当前网络下载' not in syncing.text
        assert syncing.json()['status'] == 'syncing'
        assert calls[0][0] == ['hope', 'login', USER]
        clones = [item for item in calls if item[0][1] == 'clone']
        assert [item[0][2] for item in clones] == ['https://www.modelscope.cn/datasets/modelscope/gsm8k.git']
        assert clones[0][1] == str(tmp_path / 'datasets')
        assert MS_DATASET_URL not in json.dumps(calls)
        calls.clear()
        loaded = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'public', 'dataset': MS_DATASET},
        })
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()['result']['dataset'] == str(alpaca)
        assert f'--dataset {alpaca}' in loaded.json()['command']
        assert calls == []


def test_bundled_fetch_yaml_defaults_to_hope(monkeypatch):
    monkeypatch.delenv('RSI_MODEL_SOURCE', raising=False)
    monkeypatch.delenv('RSI_FETCH_CONFIG', raising=False)
    monkeypatch.delenv('RSI_HOPE_USERNAME', raising=False)
    monkeypatch.delenv('RSI_HOPE_TOKEN', raising=False)
    monkeypatch.delenv('RSI_HOPE_PARAMS', raising=False)
    bundled = Path(__file__).resolve().parents[1] / 'configs' / 'fetch.yaml'
    text = bundled.read_text(encoding='utf-8')
    assert 'model_source: hope' in text
    assert 'hope_username: liyuhao03' in text
    assert 'model_root: /opt/meituan/dolphinfs_liubo108/ms-swift/models' in text
    assert 'data_root: /opt/meituan/dolphinfs_liubo108/ms-swift/datasets' in text
    assert 'rsi_home: /opt/meituan/dolphinfs_liubo108/ms-swift/workspace' in text
    assert 'hope_token' not in text
    assert model_source() == 'hope'
    startup = describe_startup()
    assert startup == 'model_source=hope hope_username=yes hope_token=no'
    assert 'liyuhao03' not in startup
    real, public = clone_argv('https://huggingface.co/Qwen/Qwen2.5-7B-Instruct')
    assert '--username=liyuhao03' in real
    assert public == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    monkeypatch.setenv('RSI_HOPE_USERNAME', 'other-user')
    overridden, _public = clone_argv('https://huggingface.co/Qwen/Qwen2.5-7B-Instruct')
    assert '--username=other-user' in overridden
    assert '--username=liyuhao03' not in overridden


def test_unwritable_model_root_fails_in_chinese(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    monkeypatch.setenv('RSI_MODEL_ROOT', '/dev/null/models')
    calls = []
    status = _fetcher(tmp_path, lambda *args, **kwargs: calls.append(args)).ensure('model', REPO)
    assert status.state == '失败'
    assert status.message.startswith('无法创建目录')
    assert calls == []


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
    _ready_ms_dataset(tmp_path / 'datasets')
    client, _app = _client(tmp_path, lambda *args, **kwargs: calls.append(args))
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '内网模型',
            'scenario': '把 hub id 换成本地目录',
            'base_model': REPO,
        }).json()['run']['id']
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
        blocked = api.post('/api/runs/' + run_id + '/stages/sft/run', json={
            'params': {'source': 'public', 'dataset': MS_DATASET},
        })
        assert blocked.status_code == 200, blocked.text
        assert '不能在当前网络下载' not in blocked.text
        assert blocked.json()['status'] == 'syncing'
        assert calls[0][0] == ['hope', 'login', USER]
        ms_clone = next(item for item in calls if item[0][1] == 'clone' and 'modelscope.cn/datasets/' in item[0][2])
        assert ms_clone[0][:3] == ['hope', 'clone', MS_DATASET_URL]
        assert ms_clone[1] == str(tmp_path / 'datasets')
        assert calls.index(calls[0]) < calls.index(ms_clone)
        before = len(calls)
        syncing = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
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
        clone = next(item for item in calls[before:] if item[0][1] == 'clone' and item[0][2].endswith('openai/gsm8k'))
        assert clone[0][:3] == ['hope', 'clone', 'https://huggingface.co/datasets/openai/gsm8k']
        assert clone[1] == str(tmp_path / 'datasets')
        assert not (tmp_path / 'datasets' / 'openai').exists()
        view = body['view']
        assert view['forms']['sft']['status'] == 'syncing'
        gsm = next(row for row in view['fetches'] if row['repo_id'] == 'openai/gsm8k')
        assert gsm['state'] == '同步中'


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
        assert api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'bundled', 'bundled': 'itops_mix'},
        }).status_code == 200
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
        blocked = api.post('/api/runs', json={
            'name': '魔搭 id',
            'scenario': '不能在当前网络下载',
            'base_model': 'modelscope/Qwen2.5-7B-Instruct',
        }).json()['run']['id']
        rejected = api.post(f'/api/runs/{blocked}/stages/sft/run', json={
            'params': {'source': 'bundled', 'bundled': 'itops_mix'},
        })
        assert rejected.status_code == 400
        assert rejected.json()['detail'] == 'modelscope/Qwen2.5-7B-Instruct 不能在当前网络下载，请改用本地文件'
        started = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'bundled', 'bundled': 'itops_mix'},
        })
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
        if argv[1] == 'clone' and argv[2].rstrip('/').endswith('Qwen/Qwen2.5-7B-Instruct'):
            dest = Path(kwargs['cwd']) / 'Qwen' / 'Qwen2.5-7B-Instruct'
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
        refreshed = api.post(f'/api/runs/{run_id}/fetch', json={
            'model': 'modelscope/Qwen2.5-7B-Instruct',
            'dataset': 'openai/gsm8k',
        })
        assert refreshed.status_code == 400
        assert calls == []
        refreshed = api.post(f'/api/runs/{run_id}/fetch', json={'model': REPO, 'dataset': MS_DATASET})
        assert refreshed.status_code == 200, refreshed.text
        rows = {row['kind']: row for row in refreshed.json()['fetches']}
        assert rows['model']['state'] == '就绪'
        assert rows['dataset']['state'] == '同步中'
        assert rows['dataset']['repo_id'] == MS_DATASET
        blob = refreshed.text
        assert TOKEN not in blob
        assert 'https://huggingface.co' not in blob
        assert '不能在当前网络下载' not in blob
        assert calls[0] == ['hope', 'login', USER]
        assert [MS_DATASET_URL] == [item[2] for item in calls if item[1] == 'clone' and 'datasets' in item[2]]
        assert any(item[1] == 'clone' and item[2] == 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct' for item in calls)
        assert all(item[1] != 'list' for item in calls)

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
        started = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'bundled', 'bundled': 'itops_mix'},
        })
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
    html = Path(__file__).resolve().parents[1].joinpath('static', 'index.html').read_text(encoding='utf-8')
    assert '模型来源：' not in page
    assert '模型来源' not in html
    assert '重新拉取' in page
    assert '同步中，进度见 rsi/.rsi.log' in page


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
    assert 'hope list' not in text
    assert 'exit=0' in text
    assert f'cwd={tmp_path / "models"} target={tmp_path / "models" / "Qwen" / "Qwen2.5-7B-Instruct"}' in text
    assert f'cwd={tmp_path / "models" / "Qwen"}' not in text
    assert TOKEN not in text
    assert USER not in text
    assert 'secret-params' not in text
    assert '--token=' not in text
    assert '--username=' not in text
    assert '--params=' not in text
    assert '***' in text
    logins = [item for item in seen if item[0][1] == 'login']
    clones = [item for item in seen if item[0][1] == 'clone']
    assert logins[0][0] == ['hope', 'login', USER]
    assert logins[0][1]['stdin'] is subprocess.DEVNULL
    assert logins[0][1]['timeout'] == model_fetch.LOGIN_TIMEOUT
    assert '-p' not in logins[0][0]
    clone_kwargs = clones[0][1]
    assert clone_kwargs['stdin'] is subprocess.DEVNULL
    assert 'timeout' not in clone_kwargs
    assert clones[0][0][:3] == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    assert f'--token={TOKEN}' in clones[0][0]
    assert all(item[0][1] != 'list' for item in seen)
    assert 'hope 开始 hope login ***' in text
    formatted = _hope_handler().format(caplog.records[0])
    assert re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', formatted)
    assert '%(asctime)s' in _hope_handler().formatter._fmt


def test_password_prompt_fails_fetch_and_does_not_inherit_stdin(tmp_path, hope_env, caplog):
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    seen = []

    def runner(argv, **kwargs):
        seen.append((list(argv), dict(kwargs)))
        return Completed(returncode=1, stderr=PASSWORD_PROMPT)

    status = _fetcher(tmp_path, runner).ensure('model', REPO, background=False)
    assert status.state == '失败'
    assert status.message == login_unconfirmed_message()
    assert status.message == f'hope 登录未确认：请点重新拉取再发一次，或在终端执行 hope login {USER}'
    assert 'RSI_HOPE_TOKEN' not in status.message
    assert TOKEN not in status.message
    assert seen[0][0] == ['hope', 'login', USER]
    assert seen[0][1]['stdin'] is subprocess.DEVNULL
    assert seen[0][1]['timeout'] == model_fetch.LOGIN_TIMEOUT
    assert all(item[0][1] != 'clone' for item in seen)
    assert status.message in caplog.text
    assert TOKEN not in caplog.text


def test_list_password_prompt_stops_before_clone(tmp_path, hope_env):
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return Completed(returncode=1, stdout='请输入密码')

    status = _fetcher(tmp_path, runner).refresh('model', REPO, background=False)
    assert status.state == '失败'
    assert status.message == login_unconfirmed_message()
    assert f'hope login {USER}' in status.message
    assert [item[0][1] for item in calls] == ['login']
    assert calls[0][1]['stdin'] is subprocess.DEVNULL
    assert calls[0][1]['timeout'] == model_fetch.LOGIN_TIMEOUT
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
        if argv[1] == 'clone':
            dest = Path(kwargs['cwd']) / 'Qwen' / 'Qwen2.5-7B-Instruct'
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'config.json').write_text('{}', encoding='utf-8')
        return Completed(stdout='done')

    status = _fetcher(tmp_path, runner).ensure('model', REPO, background=False)
    assert status.state == '就绪'
    assert status.message == '已在本地'
    assert 'hope 就绪 model Qwen/Qwen2.5-7B-Instruct' in caplog.text
    assert 'exit=0' in caplog.text


def test_describe_startup_reports_presence_only(tmp_path, monkeypatch, capsys, caplog):
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text('model_source: hope\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    monkeypatch.setenv('RSI_HOPE_TOKEN', 'super-secret')
    monkeypatch.delenv('RSI_HOPE_USERNAME', raising=False)
    monkeypatch.delenv('RSI_HOPE_PARAMS', raising=False)
    text = describe_startup()
    assert text == 'model_source=hope hope_username=no hope_token=yes'
    assert 'super-secret' not in text
    assert login_unconfirmed_message() == 'hope 未登录：请设置 hope_username'
    assert 'super-secret' not in login_unconfirmed_message()
    monkeypatch.setenv('RSI_HOPE_USERNAME', 'alice')
    assert 'hope login alice' in login_unconfirmed_message()
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
    logins = []
    monkeypatch.setattr(
        ModelFetcher, '_ensure_login',
        lambda self, **kwargs: logins.append(kwargs) or 'ok')
    monkeypatch.setenv('RSI_HOPE_TOKEN', 'super-secret')
    monkeypatch.setenv('RSI_HOPE_USERNAME', 'alice')
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    entry.main()
    time.sleep(0.3)
    captured = capsys.readouterr()
    combined = captured.out + captured.err + caplog.text
    assert 'RSI 启动 model_source=hope hope_username=yes hope_token=yes' in combined
    assert 'super-secret' not in combined
    assert 'alice' not in combined
    assert logins == []
    assert 'hope 开始 hope login' not in combined
    assert '大象' not in combined


LOGIN_BANNER = (
    'HOPE登录认证(帮助文档:https://km.sankuai.com/page/69700154)\n'
    '用户名(misid): {user} HOPE_LOGIN_MODE: dx_confirm\n'
    '认证消息已发送至大象，请关注万象机器学习公众号消息并确认登录\n'
)


def test_list_sends_username_and_login_banner_fails(tmp_path, hope_env, caplog):
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    banner = LOGIN_BANNER.format(user=USER)
    assert 'mis密码' not in banner
    assert hope_needs_login(banner)
    assert not hope_needs_login('已存在，等待同步')
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        return Completed(returncode=1, stdout=banner)

    status = _fetcher(tmp_path, runner).refresh('model', REPO, background=False)
    assert calls[0] == ['hope', 'login', USER]
    assert '-p' not in calls[0]
    assert status.state == '失败'
    assert status.message == login_unconfirmed_message()
    assert f'hope login {USER}' in status.message
    assert '重新拉取' in status.message
    assert f'hope login {USER}' in caplog.text
    assert TOKEN not in caplog.text
    assert '--username=' not in caplog.text
    assert '--token=' not in caplog.text
    assert all(item[1] != 'list' for item in calls)


def test_list_timeout_on_login_banner_is_a_failure(tmp_path, hope_env, caplog):
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    banner = LOGIN_BANNER.format(user=USER)
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == 'login':
            raise subprocess.TimeoutExpired(argv, kwargs.get('timeout', 90), output='快速登录超时')
        raise AssertionError('list should wait until login succeeds')

    status = _fetcher(tmp_path, runner).refresh('model', REPO, background=False)
    assert calls == [['hope', 'login', USER]]
    assert status.state == '失败'
    assert status.message == login_unconfirmed_message()
    assert f'hope login {USER}' in status.message
    assert 'exit=超时' in caplog.text
    assert f'hope login {USER}' in caplog.text
    assert TOKEN not in status.message
    assert hope_needs_login('快速登录超时')
    assert hope_needs_login(banner)


def test_yaml_username_is_used_until_env_is_set(tmp_path, monkeypatch):
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text('model_source: hope\nhope_username: from-yaml\nhope_token: nope\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    monkeypatch.delenv('RSI_HOPE_USERNAME', raising=False)
    monkeypatch.delenv('RSI_HOPE_TOKEN', raising=False)
    monkeypatch.delenv('RSI_HOPE_PARAMS', raising=False)
    real, public = clone_argv('https://huggingface.co/Qwen/Qwen2.5-7B-Instruct')
    assert '--username=from-yaml' in real
    assert not any(item.startswith('--token=') for item in real)
    assert public == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    assert describe_startup() == 'model_source=hope hope_username=yes hope_token=no'
    assert login_unconfirmed_message() == (
        'hope 登录未确认：请点重新拉取再发一次，或在终端执行 hope login from-yaml')
    monkeypatch.setenv('RSI_HOPE_USERNAME', 'from-env')
    real, public = clone_argv('https://huggingface.co/Qwen/Qwen2.5-7B-Instruct')
    assert '--username=from-env' in real
    assert '--username=from-yaml' not in real
    assert public == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']


def test_committed_username_is_not_in_code_or_token_file():
    root = Path(__file__).resolve().parents[1]
    fetch = (root / 'configs' / 'fetch.yaml').read_text(encoding='utf-8')
    assert 'hope_username: liyuhao03' in fetch
    assert 'hope_token' not in fetch
    source = (root / 'model_fetch.py').read_text(encoding='utf-8')
    assert 'liyuhao03' not in source
    assert "['hope', 'list'" not in source
    assert 'hope list' not in (root / 'static' / 'app.js').read_text(encoding='utf-8')
    readme = (root / 'README.md').read_text(encoding='utf-8')
    assert 'hope_username: liyuhao03' in readme
    assert 'hope login liyuhao03' in readme
    assert '已向大象发送登录确认，请在大象里确认' in readme
    assert 'RSI_HOPE_TOKEN' in readme


REAL_LOGIN = (
    '2026-10-19 19:56:36 HOPE登录认证(帮助文档:https://km.sankuai.com/page/69700154)\n'
    '2026-10-19 19:56:36 HOPE_LOGIN_MODE: dx_confirm\n'
    '2026-10-19 19:56:36 认证消息已发送至大象，请关注万象机器学习公众号消息并确认登录\n'
    '2026-10-19 19:56:45 登录成功\n'
)


def test_confirmed_login_is_success_and_is_not_repeated(tmp_path, hope_env, caplog):
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    assert hope_needs_login(REAL_LOGIN)
    assert hope_login_succeeded(Completed(returncode=0, stdout=REAL_LOGIN))
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == 'login':
            assert kwargs['stdin'] is subprocess.DEVNULL
            assert kwargs['timeout'] == model_fetch.LOGIN_TIMEOUT
            assert '-p' not in argv
            return Completed(returncode=0, stdout=REAL_LOGIN)
        return Completed(stdout='submitted, pending')

    fetcher = _fetcher(tmp_path, runner)
    status = fetcher.ensure('model', REPO, background=False)
    assert status.state == '同步中'
    assert calls[0] == ['hope', 'login', USER]
    assert calls[1][1] == 'clone'
    other = fetcher.ensure('dataset', 'openai/gsm8k', background=False)
    assert other.state == '同步中'
    assert [item[1] for item in calls] == ['login', 'clone', 'clone']
    assert 'hope 开始 hope login ***' in caplog.text
    assert 'exit=0' in caplog.text
    assert '登录成功' in caplog.text
    assert USER not in caplog.text
    assert TOKEN not in caplog.text


def test_login_wait_is_shared_and_shown_on_the_fetch_bar(tmp_path, hope_env, monkeypatch):
    monkeypatch.delenv('RSI_FETCH_INLINE', raising=False)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), kwargs.get('timeout')))
        if argv[1] == 'login':
            started.set()
            assert release.wait(3)
            return Completed(stdout='登录成功')
        return Completed(stdout='submitted, pending')

    fetcher = _fetcher(tmp_path, runner)
    fetcher.ensure('model', REPO, background=True)
    assert started.wait(3)
    assert fetcher.status_of('model', REPO).state == '同步中'
    assert fetcher.status_of('model', REPO).message == LOGIN_WAITING
    fetcher.ensure('dataset', 'openai/gsm8k', background=True)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if fetcher.status_of('dataset', 'openai/gsm8k').message == LOGIN_WAITING:
            break
        time.sleep(0.05)
    assert fetcher.status_of('dataset', 'openai/gsm8k').message == LOGIN_WAITING
    assert [item[0][1] for item in calls] == ['login']
    assert calls[0][1] == model_fetch.LOGIN_TIMEOUT
    release.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        kinds = [item[0][1] for item in calls]
        model_done = fetcher.status_of('model', REPO).message != LOGIN_WAITING
        data_done = fetcher.status_of('dataset', 'openai/gsm8k').state == '同步中'
        if model_done and data_done and kinds.count('clone') >= 2 and not model_fetch._login_inflight:
            break
        time.sleep(0.05)
    assert [item[0][1] for item in calls].count('login') == 1
    assert [item[0][1] for item in calls].count('clone') == 2
    assert fetcher.status_of('model', REPO).message != LOGIN_WAITING


def test_login_timeout_does_not_spam_until_refetch(tmp_path, hope_env, caplog):
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs['stdin'] is subprocess.DEVNULL
        if argv[1] == 'login':
            raise subprocess.TimeoutExpired(argv, kwargs.get('timeout', 90), output='快速登录超时')
        raise AssertionError(argv)

    fetcher = _fetcher(tmp_path, runner)
    first = fetcher.ensure('model', REPO, background=False)
    assert first.state == '失败'
    assert first.message == f'hope 登录未确认：请点重新拉取再发一次，或在终端执行 hope login {USER}'
    assert TOKEN not in first.message
    second = fetcher.ensure('model', REPO, background=False)
    assert second.state == '失败'
    assert second.message == first.message
    assert calls == [['hope', 'login', USER]]
    again = fetcher.refresh('model', REPO, background=False)
    assert again.state == '失败'
    assert calls == [['hope', 'login', USER], ['hope', 'login', USER]]
    assert caplog.text.count('exit=超时') == 2


def test_list_banner_retries_login_once_then_cools_down(tmp_path, hope_env):
    calls = []
    banner = LOGIN_BANNER.format(user=USER)

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == 'login':
            return Completed(returncode=0, stdout='登录成功')
        return Completed(returncode=1, stdout=banner)

    fetcher = _fetcher(tmp_path, runner)
    status = fetcher.refresh('model', REPO, background=False)
    assert [item[1] for item in calls] == ['login', 'clone', 'login', 'clone']
    assert calls[1][:3] == ['hope', 'clone', 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']
    assert any(item.startswith('--username=') for item in calls[1])
    assert not any(item[1] == 'list' for item in calls)
    assert status.state == '失败'
    assert status.message == login_unconfirmed_message()
    calls.clear()
    held = fetcher.ensure('model', REPO, background=False)
    assert held.state == '失败'
    assert calls == []
    fetcher.refresh('model', REPO, background=False)
    assert calls[0] == ['hope', 'login', USER]


def test_missing_username_asks_to_set_hope_username(tmp_path, monkeypatch):
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text('model_source: hope\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    monkeypatch.setenv('RSI_FETCH_INLINE', '1')
    monkeypatch.delenv('RSI_HOPE_USERNAME', raising=False)
    monkeypatch.delenv('RSI_HOPE_TOKEN', raising=False)
    calls = []
    status = _fetcher(tmp_path, lambda *args, **kwargs: calls.append(args)).ensure(
        'model', REPO, background=False)
    assert calls == []
    assert status.state == '失败'
    assert status.message == 'hope 未登录：请设置 hope_username'


def test_startup_does_not_hope_login_download_still_does(tmp_path, hope_env, monkeypatch, capsys, caplog):
    """Process start never logs in. A missing-model fetch still runs hope login first."""
    caplog.set_level(logging.INFO, logger='rsi.model_fetch')
    import rsi.__main__ as entry
    app = SimpleNamespace(state=SimpleNamespace(pipeline=SimpleNamespace(backend=SimpleNamespace(name='mock'))))
    monkeypatch.setattr(entry, 'create_app', lambda: app)
    monkeypatch.setattr(entry, 'mount_swift_ui', lambda _app: True)
    monkeypatch.setattr(entry.uvicorn, 'run', lambda *args, **kwargs: None)
    started = []
    real_ensure = ModelFetcher._ensure_login

    def spy_ensure(self, **kwargs):
        started.append(kwargs)
        return 'ok'

    monkeypatch.setattr(ModelFetcher, '_ensure_login', spy_ensure)
    entry.main()
    time.sleep(0.3)
    captured = capsys.readouterr()
    combined = captured.out + captured.err + caplog.text
    assert started == []
    assert 'hope 开始 hope login' not in combined
    assert '大象' not in combined

    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == 'login':
            return Completed(stdout='登录成功')
        return Completed(stdout='')

    monkeypatch.setattr(ModelFetcher, '_ensure_login', real_ensure)
    _fetcher(tmp_path, runner).ensure('model', REPO, background=False)
    assert calls[0] == ['hope', 'login', USER]
    assert calls[1][1] == 'clone'


def test_kick_hope_login_is_nonblocking(tmp_path, hope_env, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs['stdin'] is subprocess.DEVNULL
        assert kwargs['timeout'] == model_fetch.LOGIN_TIMEOUT
        started.set()
        assert release.wait(3)
        return Completed(stdout='登录成功')

    kick_hope_login(runner=runner)
    assert started.wait(3)
    assert calls == [['hope', 'login', USER]]
    release.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and model_fetch._login_inflight:
        time.sleep(0.05)
    assert model_fetch._login_ok
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'modelscope')
    kick_hope_login(runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('logged in again')))
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text('model_source: hope\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    monkeypatch.delenv('RSI_HOPE_USERNAME', raising=False)
    kick_hope_login(runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('missing misid')))


def test_clone_chunks_are_in_the_process_log_before_hope_returns(tmp_path, hope_env, monkeypatch):
    log = tmp_path / '.rsi.log'
    monkeypatch.setattr(model_fetch, 'process_log_path', lambda: log)
    seen = {}

    def runner(argv, **kwargs):
        if argv[1] == 'login':
            return Completed(stdout='登录成功')
        kwargs['on_chunk'](f'下载 1% token={TOKEN}\n'.encode())
        seen['mid'] = log.read_text(encoding='utf-8')
        kwargs['on_chunk']('下载 8%\n'.encode())
        return Completed(stdout='should-not-be-appended-again')

    status = _fetcher(tmp_path, runner).ensure('model', REPO, background=False)
    assert status.state == '同步中'
    assert '下载 1%' in seen['mid']
    assert '下载 8%' not in seen['mid']
    body = log.read_text(encoding='utf-8')
    assert '下载 1%' in body
    assert '下载 8%' in body
    assert TOKEN not in body
    assert 'should-not-be-appended-again' not in body


def test_stream_subprocess_writes_before_the_child_exits(tmp_path, monkeypatch):
    log = tmp_path / '.rsi.log'
    flag = tmp_path / 'go'
    monkeypatch.setattr(model_fetch, 'process_log_path', lambda: log)
    script = (
        'import sys, time\n'
        'from pathlib import Path\n'
        'flag = Path(sys.argv[1])\n'
        'sys.stdout.buffer.write(b"tick-1\\n")\n'
        'sys.stdout.buffer.flush()\n'
        'for _ in range(40):\n'
        '    if flag.exists():\n'
        '        break\n'
        '    time.sleep(0.05)\n'
        'else:\n'
        '    raise SystemExit(2)\n'
        'sys.stdout.buffer.write(b"tick-2\\n")\n'
        'sys.stdout.buffer.flush()\n'
    )

    def on_chunk(data):
        model_fetch.append_process_log(data.decode())
        if b'tick-1' in data and not flag.exists():
            text = log.read_text(encoding='utf-8')
            assert 'tick-1' in text
            assert 'tick-2' not in text
            flag.write_text('go', encoding='utf-8')

    code = model_fetch.stream_subprocess(
        [sys.executable, '-c', script, str(flag)], tmp_path, on_chunk)
    assert code == 0
    assert flag.is_file()
    final = log.read_text(encoding='utf-8')
    assert 'tick-1' in final
    assert 'tick-2' in final


def _seed_model(root: Path) -> Path:
    dest = root / 'Qwen' / 'Qwen2.5-7B-Instruct'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'config.json').write_text('{}', encoding='utf-8')
    return dest


def _sft_log(home: Path, run_id: str) -> Path:
    return home / 'runs' / run_id / 'round_1' / 'sft' / 'swift.log'


def test_clone_chunks_reach_the_stage_log_before_hope_returns(tmp_path, hope_env, monkeypatch):
    process = tmp_path / 'process.log'
    stage = tmp_path / 'swift.log'
    monkeypatch.setattr(model_fetch, 'process_log_path', lambda: process)
    seen = {}

    def runner(argv, **kwargs):
        if argv[1] == 'login':
            assert kwargs['timeout'] == model_fetch.LOGIN_TIMEOUT
            assert kwargs['stdin'] is subprocess.DEVNULL
            return Completed(stdout=f'登录成功 token={TOKEN} user={USER}')
        kwargs['on_chunk'](f'下载 1% token={TOKEN}\n'.encode())
        seen['stage'] = stage.read_text(encoding='utf-8')
        seen['process'] = process.read_text(encoding='utf-8')
        kwargs['on_chunk']('下载 8%\n'.encode())
        return Completed(stdout='should-not-be-appended-again')

    with model_fetch.use_stage_log(stage):
        status = _fetcher(tmp_path, runner).ensure('model', REPO, background=False)
    assert status.state == '同步中'
    assert '[模型] ' + REPO in seen['stage']
    assert '登录成功' in seen['stage']
    assert '下载 1%' in seen['stage']
    assert '下载 8%' not in seen['stage']
    assert '下载 1%' in seen['process']
    assert '[模型]' not in seen['process']
    body = stage.read_text(encoding='utf-8')
    assert body.index('登录成功') < body.index('下载 1%') < body.index('下载 8%')
    assert TOKEN not in body
    assert USER not in body
    assert 'should-not-be-appended-again' not in body
    copied = process.read_text(encoding='utf-8')
    assert '下载 8%' in copied
    assert TOKEN not in copied
    assert USER not in copied


def test_ready_repo_does_not_write_a_stage_section(tmp_path, hope_env):
    _seed_model(tmp_path / 'models')
    stage = tmp_path / 'swift.log'

    def refuse(*_args, **_kwargs):
        raise AssertionError('a repo that is already on disk must not call hope')

    with model_fetch.use_stage_log(stage):
        status = _fetcher(tmp_path, refuse).ensure('model', REPO, background=False)
    assert status.state == '就绪'
    assert not stage.exists()


def test_live_stage_log_streams_during_a_mocked_fetch(tmp_path, hope_env, monkeypatch):
    monkeypatch.delenv('RSI_FETCH_INLINE', raising=False)
    process = tmp_path / 'process.log'
    monkeypatch.setattr(model_fetch, 'process_log_path', lambda: process)
    started = threading.Event()
    release = threading.Event()
    _seed_model(tmp_path / 'models')

    def runner(argv, **kwargs):
        if argv[1] == 'login':
            return Completed(stdout=f'登录成功 token={TOKEN} user={USER}\n')
        kwargs['on_chunk'](f'下载 1% token={TOKEN}\n'.encode())
        started.set()
        assert release.wait(5)
        kwargs['on_chunk']('下载 8%\n'.encode())
        return Completed(stdout='should-not-be-appended-again')

    script = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert "form.status === 'running' || form.status === 'syncing'" in script
    assert "step.status === 'running' || step.status === 'syncing'" in script
    assert "form.status === 'running' ? '运行中'" in script

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '实时日志',
            'scenario': '同步时阶段日志跟着刷新',
            'base_model': REPO,
        }).json()['run']['id']
        try:
            started_run = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
                'params': {'source': 'public', 'dataset': MS_DATASET},
            })
            assert started_run.status_code == 200, started_run.text
            assert started_run.json()['status'] == 'syncing'
            assert started.wait(5)
            mid = api.get(f'/api/runs/{run_id}/stages/sft/log')
            assert mid.status_code == 200, mid.text
            payload = mid.json()
            assert payload['done'] is False
            assert payload['status'] == 'syncing'
            text = payload['text']
            assert f'[数据集] {MS_DATASET}' in text
            assert '下载 1%' in text
            assert '下载 8%' not in text
            assert TOKEN not in text
            assert USER not in text
            assert 'should-not-be-appended-again' not in text
            process_mid = process.read_text(encoding='utf-8')
            assert '下载 1%' in process_mid
            assert TOKEN not in process_mid
        finally:
            release.set()
        deadline = time.monotonic() + 3
        later = ''
        while time.monotonic() < deadline:
            later = api.get(f'/api/runs/{run_id}/stages/sft/log').json()['text']
            if '下载 8%' in later:
                break
            time.sleep(0.05)
        assert '下载 1%' in later
        assert '下载 8%' in later
        assert later.index('下载 1%') < later.index('下载 8%')
        assert TOKEN not in later
        assert USER not in later


def test_multiselect_labels_each_repo_in_the_stage_log(tmp_path, hope_env, monkeypatch):
    process = tmp_path / 'process.log'
    monkeypatch.setattr(model_fetch, 'process_log_path', lambda: process)
    _seed_model(tmp_path / 'models')
    other = 'openai/gsm8k'

    def runner(argv, **kwargs):
        if argv[1] == 'login':
            return Completed(stdout=f'token={TOKEN} user={USER}\n')
        kwargs['on_chunk'](f'{argv[2]} 拉取中\n'.encode())
        return Completed(stdout='should-not-be-appended-again')

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '多选',
            'scenario': '每个仓库单独成段',
            'base_model': REPO,
        }).json()['run']['id']
        started = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'public', 'dataset': f'{MS_DATASET} {other}'},
        })
        assert started.status_code == 200, started.text
        assert started.json()['status'] == 'syncing'
        text = _sft_log(tmp_path / 'home', run_id).read_text(encoding='utf-8')
    assert text.index(f'[数据集] {MS_DATASET}') < text.index(f'[数据集] {other}')
    assert MS_DATASET_URL + ' 拉取中' in text
    assert 'https://huggingface.co/datasets/openai/gsm8k 拉取中' in text
    assert TOKEN not in text
    assert USER not in text
    copied = process.read_text(encoding='utf-8')
    assert '拉取中' in copied
    assert '[数据集]' not in copied
    assert TOKEN not in copied


def test_fetch_lines_stay_above_the_stage_command(tmp_path, hope_env, monkeypatch):
    monkeypatch.setattr(model_fetch, 'process_log_path', lambda: tmp_path / 'process.log')
    _seed_model(tmp_path / 'models')

    def runner(argv, **kwargs):
        if argv[1] == 'login':
            return Completed(stdout=f'登录成功 token={TOKEN}\n')
        kwargs['on_chunk'](b'clone-progress\n')
        return Completed()

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '进度留在上面',
            'scenario': '训练输出接在拉取后面',
            'base_model': REPO,
        }).json()['run']['id']
        waiting = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'public', 'dataset': MS_DATASET},
        })
        assert waiting.status_code == 200, waiting.text
        assert waiting.json()['status'] == 'syncing'
        log_path = _sft_log(tmp_path / 'home', run_id)
        assert 'clone-progress' in log_path.read_text(encoding='utf-8')
        _ready_ms_dataset(tmp_path / 'datasets')
        trained = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'public', 'dataset': MS_DATASET},
        })
        assert trained.status_code == 200, trained.text
        assert trained.json()['status'] == 'succeeded'
        body = log_path.read_text(encoding='utf-8')
        assert f'[数据集] {MS_DATASET}' in body
        assert body.index('clone-progress') < body.index('swift sft')
        assert TOKEN not in body
        shown = trained.json()['view']['forms']['sft']['logs']
        assert 'clone-progress' in shown
        assert 'swift sft' in shown
        again = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'public', 'dataset': MS_DATASET},
        })
        assert again.status_code == 200, again.text
        assert again.json()['status'] == 'succeeded'
        fresh = log_path.read_text(encoding='utf-8')
        assert 'clone-progress' not in fresh
        assert 'swift sft' in fresh


def test_model_fetch_lines_stay_above_the_swift_command(tmp_path, hope_env, monkeypatch):
    monkeypatch.setattr(model_fetch, 'process_log_path', lambda: tmp_path / 'process.log')

    def runner(argv, **kwargs):
        if argv[1] == 'login':
            return Completed(stdout=f'登录成功 token={TOKEN}\n')
        kwargs['on_chunk']('模型进度\n'.encode())
        if argv[1] == 'clone':
            dest = Path(kwargs['cwd']) / 'huggingface.co' / 'Qwen' / 'Qwen2.5-7B-Instruct'
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'config.json').write_text('{}', encoding='utf-8')
        return Completed()

    client, _app = _client(tmp_path, runner)
    with client as api:
        run_id = api.post('/api/runs', json={
            'name': '模型进度',
            'scenario': '模型拉取写进阶段日志',
            'base_model': REPO,
        }).json()['run']['id']
        trained = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
            'params': {'source': 'bundled', 'bundled': 'itops_mix'},
        })
        assert trained.status_code == 200, trained.text
        assert trained.json()['status'] == 'succeeded', trained.text
        body = _sft_log(tmp_path / 'home', run_id).read_text(encoding='utf-8')
    assert f'[模型] {REPO}' in body
    assert body.index('模型进度') < body.index('swift sft')
    assert TOKEN not in body
    assert '登录成功' in body
