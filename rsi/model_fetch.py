# Copyright (c) ModelScope Contributors. All rights reserved.
"""Bring hub ids onto local disk before swift runs.

The default source leaves ids untouched, which is what swift already does
(ModelScope, or Hugging Face when ``USE_HF=1``). ``RSI_MODEL_SOURCE=hope``
copies a Hugging Face repo with the hope client and rewrites the swift
command to that directory. Credentials stay in the process environment.
"""
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

SOURCES = ('hope', 'hf', 'modelscope', 'local')
WAITING_MARKERS = ('等待', '已存在', '同步', 'submitted', 'exist', 'pending')
DATASET_READY_NAMES = {'.gitattributes', 'README.md', 'dataset_info.json', 'dataset_infos.json'}
NOT_HF_ORGS = {'modelscope', 'AI-ModelScope', 'AI-MO', 'swift'}
HF_CLONEABLE_DATASETS = (
    'openai/gsm8k',
    'cais/mmlu',
    'Rowan/hellaswag',
    'tatsu-lab/alpaca',
)
REPO_ID = re.compile(r'^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$')
_LOCK = threading.Lock()
_INFLIGHT = set()
_HOPE_WARNED = False


@dataclass
class FetchStatus:
    kind: str
    repo_id: str
    local_path: str
    state: str
    message: str

    def as_row(self) -> dict:
        return {
            'kind': self.kind,
            'repo_id': self.repo_id,
            'local_path': self.local_path,
            'state': self.state,
            'message': self.message,
        }


def _config_file() -> Path | None:
    raw = os.environ.get('RSI_FETCH_CONFIG', '').strip()
    if raw:
        path = Path(raw)
        return path if path.is_file() else None
    bundled = Path(__file__).resolve().parent / 'configs' / 'fetch.yaml'
    return bundled if bundled.is_file() else None


def _file_settings() -> dict:
    path = _config_file()
    if path is None:
        return {}
    with path.open(encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def model_source() -> str:
    file_value = str(_file_settings().get('model_source') or '').strip().lower()
    raw = os.environ.get('RSI_MODEL_SOURCE', '').strip().lower() or file_value or 'modelscope'
    if raw not in SOURCES:
        return 'modelscope'
    return raw


def _root(env_name: str, file_key: str, home: Path, folder: str) -> Path:
    env = os.environ.get(env_name, '').strip()
    if env:
        return Path(env).expanduser()
    file_value = str(_file_settings().get(file_key) or '').strip()
    if file_value:
        return Path(file_value).expanduser()
    return Path(home) / folder


def redact(text: str) -> str:
    if not text:
        return ''
    cleaned = text
    for key in ('RSI_HOPE_TOKEN', 'RSI_HOPE_USERNAME', 'RSI_HOPE_PARAMS'):
        secret = os.environ.get(key, '').strip()
        if secret:
            cleaned = cleaned.replace(secret, '***')
    cleaned = re.sub(r'--token=\S+', '--token=***', cleaned)
    cleaned = re.sub(r'--username=\S+', '--username=***', cleaned)
    cleaned = re.sub(r'--params=\S+', '--params=***', cleaned)
    return cleaned


def repo_id_of(value: str) -> str:
    text = (value or '').strip()
    if '#' in text:
        text = text.split('#', 1)[0]
    return text


def is_repo_id(value: str) -> bool:
    return bool(REPO_ID.match(repo_id_of(value)))


def is_modelscope_id(value: str) -> bool:
    repo = repo_id_of(value)
    if not is_repo_id(repo):
        return False
    return repo.split('/', 1)[0] in NOT_HF_ORGS


def hf_cloneable(value: str) -> bool:
    repo = repo_id_of(value)
    if '#' in (value or ''):
        return False
    return is_repo_id(repo) and not is_modelscope_id(repo)


def hf_dataset_choices(extra_ids=()) -> list:
    found = list(HF_CLONEABLE_DATASETS)
    seen = set(found)
    for item in extra_ids or []:
        if hf_cloneable(item) and item not in seen:
            found.append(item)
            seen.add(item)
    return found


def dataset_block_message(value: str) -> str:
    raw = (value or '').strip()
    if not raw or model_source() not in {'hope', 'local'}:
        return ''
    if is_modelscope_id(raw) or '#' in raw:
        return f'{raw} 不能在当前网络下载，请改用本地文件'
    return ''


def _split_repo(repo_id: str) -> tuple:
    org, name = repo_id_of(repo_id).split('/', 1)
    return org, name


def destination(root: Path, repo_id: str) -> Path:
    org, name = _split_repo(repo_id)
    return Path(root) / org / name


def model_url(repo_id: str) -> str:
    return f'https://huggingface.co/{repo_id_of(repo_id)}'


def dataset_url(repo_id: str) -> str:
    return f'https://huggingface.co/datasets/{repo_id_of(repo_id)}'


def public_clone_command(url: str) -> str:
    return f'hope clone {url}'


def clone_argv(url: str) -> tuple:
    """Return the real argv and the argv safe to show or store."""
    public = ['hope', 'clone', url]
    argv = list(public)
    username = os.environ.get('RSI_HOPE_USERNAME', '').strip()
    token = os.environ.get('RSI_HOPE_TOKEN', '').strip()
    params = os.environ.get('RSI_HOPE_PARAMS', '').strip()
    if username:
        argv.append(f'--username={username}')
    if token:
        argv.append(f'--token={token}')
    if params:
        argv.append(f'--params={params}')
    return argv, public


def list_argv(repo_id: str) -> list:
    _org, name = _split_repo(repo_id)
    return ['hope', 'list', '--creator=all', f'--name_like={name}']


def apply_runtime_env(env: dict) -> dict:
    for key in ('RSI_HOPE_TOKEN', 'RSI_HOPE_USERNAME', 'RSI_HOPE_PARAMS'):
        env.pop(key, None)
    source = model_source()
    if source == 'hf':
        env['USE_HF'] = '1'
    elif source in {'hope', 'local'}:
        env['HF_HUB_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'
        env['HF_DATASETS_OFFLINE'] = '1'
    return env


def _warn_missing_hope() -> None:
    global _HOPE_WARNED
    if _HOPE_WARNED:
        return
    _HOPE_WARNED = True
    print(
        '未找到 hope。安装：pip3 install hope==3.6.6 --no-cache-dir -i https://pypi.sankuai.com/simple',
        file=sys.stderr,
    )


def _model_ready(path: Path) -> bool:
    return path.is_dir() and (path / 'config.json').is_file()


def _dataset_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.name in DATASET_READY_NAMES or child.suffix in {'.jsonl', '.parquet', '.arrow', '.csv', '.md'}:
            return True
        if child.is_dir() and child.name not in {'.git'}:
            return True
    return False


def _ready(kind: str, path: Path) -> bool:
    if kind == 'model':
        return _model_ready(path)
    return _dataset_ready(path)


class ModelFetcher:
    def __init__(self, home: Path, runner=None):
        self.home = Path(home)
        self.runner = runner or subprocess.run
        self._notes = {}

    def model_root(self) -> Path:
        return _root('RSI_MODEL_ROOT', 'model_root', self.home, 'models')

    def data_root(self) -> Path:
        return _root('RSI_DATA_ROOT', 'data_root', self.home, 'datasets')

    def _path_for(self, kind: str, repo_id: str) -> Path:
        root = self.model_root() if kind == 'model' else self.data_root()
        return destination(root, repo_id)

    def _url_for(self, kind: str, repo_id: str) -> str:
        if kind == 'dataset':
            return dataset_url(repo_id)
        return model_url(repo_id)

    def status_of(self, kind: str, repo_id: str) -> FetchStatus:
        path = self._path_for(kind, repo_id)
        if _ready(kind, path):
            return FetchStatus(kind, repo_id_of(repo_id), str(path), '就绪', '已在本地')
        key = f'{kind}:{repo_id_of(repo_id)}'
        with _LOCK:
            inflight = key in _INFLIGHT
            note = self._notes.get(key, '')
        if note == '下载工具不可用' and not inflight:
            return FetchStatus(kind, repo_id_of(repo_id), str(path), '失败', '下载工具不可用')
        if inflight or note == '同步中' or path.exists():
            return FetchStatus(kind, repo_id_of(repo_id), str(path), '同步中', '同步中')
        return FetchStatus(kind, repo_id_of(repo_id), str(path), '缺少', '本地还没有这份文件')

    def ensure(self, kind: str, repo_id: str, background: bool = True) -> FetchStatus:
        repo = repo_id_of(repo_id)
        current = self.status_of(kind, repo)
        if current.state == '就绪':
            return current
        if model_source() == 'local':
            return FetchStatus(kind, repo, current.local_path, '失败', '本地没有该文件')
        if model_source() != 'hope':
            return FetchStatus(kind, repo, repo, '保持', '')
        inline = os.environ.get('RSI_FETCH_INLINE', '').strip() == '1'
        self._start_clone(kind, repo, background=background and not inline)
        latest = self.status_of(kind, repo)
        if latest.state in {'就绪', '失败'}:
            return latest
        return FetchStatus(kind, repo, latest.local_path, '同步中', '同步中')

    def refresh(self, kind: str, repo_id: str, background: bool = True) -> FetchStatus:
        repo = repo_id_of(repo_id)
        if model_source() != 'hope' or not is_repo_id(repo):
            return self.status_of(kind, repo) if is_repo_id(repo) else FetchStatus(kind, repo, '', '保持', '')

        def job():
            self._run(list_argv(repo), cwd=self.home)
            self._start_clone(kind, repo, background=False)

        inline = os.environ.get('RSI_FETCH_INLINE', '').strip() == '1'
        if background and not inline:
            threading.Thread(target=job, daemon=True).start()
            current = self.status_of(kind, repo)
            if current.state == '就绪':
                return current
            return FetchStatus(kind, repo, current.local_path, '同步中', '同步中')
        job()
        latest = self.status_of(kind, repo)
        if latest.state in {'就绪', '失败'}:
            return latest
        return FetchStatus(kind, repo, latest.local_path, '同步中', '同步中')

    def _start_clone(self, kind: str, repo_id: str, background: bool) -> None:
        key = f'{kind}:{repo_id}'
        path = self._path_for(kind, repo_id)
        if _ready(kind, path):
            return
        with _LOCK:
            if key in _INFLIGHT:
                return
            _INFLIGHT.add(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        def job():
            try:
                argv, _public = clone_argv(self._url_for(kind, repo_id))
                result = self._run(argv, cwd=path.parent)
                with _LOCK:
                    if result == 'missing-tool':
                        self._notes[key] = '下载工具不可用'
                    else:
                        self._notes.pop(key, None)
            finally:
                with _LOCK:
                    _INFLIGHT.discard(key)
                    if _ready(kind, path):
                        self._notes.pop(key, None)
                    elif self._notes.get(key) != '下载工具不可用':
                        self._notes[key] = '同步中'

        if background:
            threading.Thread(target=job, daemon=True).start()
            return
        job()

    def _run(self, argv: list, cwd: Path):
        try:
            completed = self.runner(argv, cwd=str(cwd), capture_output=True, text=True, check=False)
        except FileNotFoundError:
            _warn_missing_hope()
            return 'missing-tool'
        stdout = redact(getattr(completed, 'stdout', '') or '')
        stderr = redact(getattr(completed, 'stderr', '') or '')
        text = f'{stdout}\n{stderr}'
        code = getattr(completed, 'returncode', 1)
        if code not in (0, None) and not any(marker in text.lower() for marker in (item.lower() for item in WAITING_MARKERS)):
            return completed
        return completed


def benchmark_dir(home: Path, name: str) -> Path:
    return _root('RSI_DATA_ROOT', 'data_root', home, 'datasets') / 'benchmarks' / name
