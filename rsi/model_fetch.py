# Copyright (c) ModelScope Contributors. All rights reserved.
"""Bring hub ids onto local disk before swift runs.

``rsi/configs/fetch.yaml`` sets the default source to hope. ``RSI_MODEL_SOURCE``
still wins: ``modelscope`` leaves ids untouched, ``hf`` sets ``USE_HF=1``,
and ``hope`` copies a Hugging Face repo locally before swift runs.
Credentials stay in the process environment.
"""
import logging
import os
import re
import subprocess
import sys
import threading
import time
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
LOGIN_MESSAGE = 'hope 需要登录：请在启动前设置 RSI_HOPE_USERNAME 和 RSI_HOPE_TOKEN'
_FAILURE_NOTES = {'下载工具不可用', LOGIN_MESSAGE}
LIST_TIMEOUT = 60
_LOG_LIMIT = 2000
logger = logging.getLogger('rsi.model_fetch')


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


def public_hope_command(argv: list) -> str:
    """Command safe to write in the launch log. Drops credential flags."""
    kept = []
    for token in argv or []:
        if token.startswith(('--username=', '--token=', '--params=')):
            continue
        kept.append(token)
    return redact(' '.join(kept))


def hope_needs_login(text: str) -> bool:
    raw = text or ''
    lowered = raw.lower()
    if 'mis密码' in raw or '请输入密码' in raw:
        return True
    if 'password input may be echoed' in lowered:
        return True
    if 'authentication failed' in lowered or '认证失败' in raw or '登录失败' in raw:
        return True
    return False


def describe_startup() -> str:
    """One line for process start. Presence only, never the secret values."""
    username = 'yes' if os.environ.get('RSI_HOPE_USERNAME', '').strip() else 'no'
    token = 'yes' if os.environ.get('RSI_HOPE_TOKEN', '').strip() else 'no'
    return f'model_source={model_source()} hope_username={username} hope_token={token}'


def _trim_log(text: str, limit: int = _LOG_LIMIT) -> str:
    cleaned = redact(text or '').strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + f'\n…截断，共 {len(cleaned)} 字符'


class _StderrHandler(logging.StreamHandler):
    """Write to the process stderr at emit time so run.sh's tee still sees it."""

    def __init__(self):
        super().__init__(sys.stderr)
        self._rsi_hope = True

    def emit(self, record):
        self.stream = sys.stderr
        super().emit(record)


def _install_hope_logger() -> None:
    if any(getattr(handler, '_rsi_hope', False) for handler in logger.handlers):
        return
    handler = _StderrHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [hope] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


_install_hope_logger()


@dataclass
class _HopeResult:
    returncode: int
    stdout: str
    stderr: str


def _as_text(value) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return str(value)


def _result_text(result) -> str:
    if result in (None, 'missing-tool'):
        return ''
    stdout = _as_text(getattr(result, 'stdout', '') or '')
    stderr = _as_text(getattr(result, 'stderr', '') or '')
    return f'{stdout}\n{stderr}'


_PERCENT = re.compile(r'\d+(?:\.\d+)?\s*%')
_SIZE = re.compile(r'\d+(?:\.\d+)?\s*(?:B|KB|MB|GB|TB|KiB|MiB|GiB|字节)', re.IGNORECASE)
_STATUS_WORDS = ('下载', '同步', '等待', '完成', '失败', '排队', '已存在', 'running', 'pending', 'waiting', 'done', 'failed', 'submitted')


def parse_hope_list(text: str, repo_id: str) -> str:
    """Pick the most informative line hope already printed.

    A percent or size is copied only when it appears in the output. Nothing is
    estimated when hope does not print one.
    """
    cleaned = redact(text or '').replace('\r\n', '\n').replace('\r', '\n')
    cleaned = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', cleaned)
    lines = [' '.join(line.split()) for line in cleaned.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ''
    name = ''
    repo = repo_id_of(repo_id)
    if '/' in repo:
        name = repo.split('/', 1)[1]
    named = [line for line in lines if name and name in line]
    pool = named or lines

    def score(line: str) -> int:
        value = 0
        if _PERCENT.search(line):
            value += 4
        if _SIZE.search(line):
            value += 3
        lowered = line.lower()
        for word in _STATUS_WORDS:
            if word.lower() in lowered:
                value += 1
        return value

    best = max(pool, key=score)
    if len(best) > 160:
        best = best[:160] + '…'
    return best


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
        self._progress = {}
        self._probed_at = {}
        self._probe_inflight = set()

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
            progress = self._progress.get(key, '')
        if note in _FAILURE_NOTES and not inflight:
            return FetchStatus(kind, repo_id_of(repo_id), str(path), '失败', note)
        if inflight or note == '同步中' or path.exists() or progress:
            return FetchStatus(kind, repo_id_of(repo_id), str(path), '同步中', progress or '同步中')
        return FetchStatus(kind, repo_id_of(repo_id), str(path), '缺少', '本地还没有这份文件')

    def ensure(self, kind: str, repo_id: str, background: bool = True) -> FetchStatus:
        repo = repo_id_of(repo_id)
        current = self.status_of(kind, repo)
        if current.state == '就绪':
            logger.info('hope 跳过 %s %s 已就绪 %s', kind, repo, current.local_path)
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
        message = latest.message if latest.state == '同步中' and latest.message else '同步中'
        return FetchStatus(kind, repo, latest.local_path, '同步中', message)

    def refresh(self, kind: str, repo_id: str, background: bool = True) -> FetchStatus:
        repo = repo_id_of(repo_id)
        if model_source() != 'hope' or not is_repo_id(repo):
            return self.status_of(kind, repo) if is_repo_id(repo) else FetchStatus(kind, repo, '', '保持', '')

        key = f'{kind}:{repo}'
        with _LOCK:
            self._probed_at[key] = time.monotonic()

        def job():
            listed = self._run(list_argv(repo), cwd=self.home, target=self._path_for(kind, repo))
            if self._remember_progress(kind, repo, listed):
                return
            self._start_clone(kind, repo, background=False)

        inline = os.environ.get('RSI_FETCH_INLINE', '').strip() == '1'
        if background and not inline:
            threading.Thread(target=job, daemon=True).start()
            current = self.status_of(kind, repo)
            if current.state == '就绪':
                return current
            message = current.message if current.state == '同步中' and current.message else '同步中'
            return FetchStatus(kind, repo, current.local_path, '同步中', message)
        job()
        latest = self.status_of(kind, repo)
        if latest.state in {'就绪', '失败'}:
            return latest
        return FetchStatus(kind, repo, latest.local_path, '同步中', latest.message or '同步中')

    def probe(self, kind: str, repo_id: str, background: bool = True) -> FetchStatus:
        """Ask hope list for the latest status text. Does not start another clone."""
        repo = repo_id_of(repo_id)
        current = self.status_of(kind, repo)
        if current.state != '同步中' or model_source() != 'hope' or not is_repo_id(repo):
            return current
        key = f'{kind}:{repo}'
        now = time.monotonic()
        inline = os.environ.get('RSI_FETCH_INLINE', '').strip() == '1'
        with _LOCK:
            if key in self._probe_inflight:
                return current
            if now - self._probed_at.get(key, 0) < 5:
                return current
            self._probed_at[key] = now
            self._probe_inflight.add(key)

        def job():
            try:
                listed = self._run(list_argv(repo), cwd=self.home, target=self._path_for(kind, repo))
                self._remember_progress(kind, repo, listed)
            finally:
                with _LOCK:
                    self._probe_inflight.discard(key)

        if background and not inline:
            threading.Thread(target=job, daemon=True).start()
            return current
        job()
        return self.status_of(kind, repo)

    def _remember_progress(self, kind: str, repo_id: str, result) -> bool:
        """Store list text. Return True when hope asked for a login and clone must stop."""
        if result in (None, 'missing-tool'):
            return False
        text = _result_text(result)
        key = f'{kind}:{repo_id_of(repo_id)}'
        if hope_needs_login(text):
            with _LOCK:
                self._notes[key] = LOGIN_MESSAGE
                self._progress.pop(key, None)
            logger.info('hope 需要登录 %s %s', kind, repo_id_of(repo_id))
            return True
        message = parse_hope_list(text, repo_id)
        if not message:
            return False
        with _LOCK:
            self._progress[key] = message
        return False

    def _start_clone(self, kind: str, repo_id: str, background: bool) -> None:
        key = f'{kind}:{repo_id}'
        path = self._path_for(kind, repo_id)
        if _ready(kind, path):
            logger.info('hope 跳过 %s %s 已就绪 %s', kind, repo_id, path)
            return
        with _LOCK:
            if key in _INFLIGHT:
                logger.info('hope 跳过 %s %s 已在进行', kind, repo_id)
                return
            _INFLIGHT.add(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        def job():
            ready_now = False
            try:
                argv, _public = clone_argv(self._url_for(kind, repo_id))
                result = self._run(argv, cwd=path.parent, target=path)
                with _LOCK:
                    if result == 'missing-tool':
                        self._notes[key] = '下载工具不可用'
                    elif hope_needs_login(_result_text(result)):
                        self._notes[key] = LOGIN_MESSAGE
                        self._progress.pop(key, None)
                        logger.info('hope 需要登录 %s %s', kind, repo_id)
                    else:
                        self._notes.pop(key, None)
            finally:
                with _LOCK:
                    _INFLIGHT.discard(key)
                    if _ready(kind, path):
                        self._notes.pop(key, None)
                        self._progress.pop(key, None)
                        ready_now = True
                    elif self._notes.get(key) not in _FAILURE_NOTES:
                        self._notes[key] = '同步中'
            if ready_now:
                logger.info('hope 就绪 %s %s %s', kind, repo_id, path)

        if background:
            threading.Thread(target=job, daemon=True).start()
            return
        job()

    def _run(self, argv: list, cwd: Path, target: Path | None = None):
        command = public_hope_command(argv)
        kwargs = {
            'cwd': str(cwd),
            'capture_output': True,
            'text': True,
            'check': False,
            'stdin': subprocess.DEVNULL,
        }
        if len(argv) >= 2 and argv[1] == 'list':
            kwargs['timeout'] = LIST_TIMEOUT
        where = f'cwd={cwd}'
        if target is not None:
            where += f' target={target}'
        logger.info('hope 开始 %s %s', command, where)
        try:
            completed = self.runner(argv, **kwargs)
        except FileNotFoundError:
            _warn_missing_hope()
            logger.info('hope 结束 %s %s exit=缺失', command, where)
            return 'missing-tool'
        except subprocess.TimeoutExpired as exc:
            stdout = _as_text(getattr(exc, 'stdout', '') or '')
            stderr = _as_text(getattr(exc, 'stderr', '') or '')
            logger.info('hope 结束 %s %s exit=超时\n%s', command, where, _trim_log(f'{stdout}\n{stderr}'))
            return _HopeResult(124, stdout, stderr)
        stdout = _as_text(getattr(completed, 'stdout', '') or '')
        stderr = _as_text(getattr(completed, 'stderr', '') or '')
        code = getattr(completed, 'returncode', 1)
        logger.info('hope 结束 %s %s exit=%s\n%s', command, where, code, _trim_log(f'{stdout}\n{stderr}'))
        return completed


def benchmark_dir(home: Path, name: str) -> Path:
    return _root('RSI_DATA_ROOT', 'data_root', home, 'datasets') / 'benchmarks' / name
