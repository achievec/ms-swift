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
_TOOL_FAILURE = '下载工具不可用'
LOGIN_WAITING = '已向大象发送登录确认，请在大象里确认'
LOGIN_TIMEOUT = 90
LOGIN_COOLDOWN = 180
_LOGIN_FAIL_PREFIX = 'hope 登录未确认：'
_LOGIN_MISSING_USER = 'hope 未登录：请设置 hope_username'
_login_mu = threading.Lock()
_login_cv = threading.Condition(_login_mu)
_login_inflight = False
_login_ok = False
_login_failed_at = 0.0
_login_generation = 0
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


def _expand_path(value: str) -> Path:
    """Turn ``~`` and ``$HOME`` into an absolute directory hope can use as cwd."""
    text = os.path.expandvars(os.path.expanduser((value or '').strip()))
    path = Path(text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _root(env_name: str, file_key: str, home: Path, folder: str) -> Path:
    env = os.environ.get(env_name, '').strip()
    if env:
        return _expand_path(env)
    file_value = str(_file_settings().get(file_key) or '').strip()
    if file_value:
        return _expand_path(file_value)
    return Path(home) / folder


def hope_username() -> str:
    """misid for hope. The environment wins over ``hope_username`` in fetch.yaml."""
    env = os.environ.get('RSI_HOPE_USERNAME', '').strip()
    if env:
        return env
    return str(_file_settings().get('hope_username') or '').strip()


def redact(text: str) -> str:
    if not text:
        return ''
    cleaned = text
    for key in ('RSI_HOPE_TOKEN', 'RSI_HOPE_USERNAME', 'RSI_HOPE_PARAMS'):
        secret = os.environ.get(key, '').strip()
        if secret:
            cleaned = cleaned.replace(secret, '***')
    username = hope_username()
    if username:
        cleaned = cleaned.replace(username, '***')
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


def hope_credential_flags() -> list:
    """Git credentials for ``hope clone`` only."""
    flags = []
    username = hope_username()
    token = os.environ.get('RSI_HOPE_TOKEN', '').strip()
    params = os.environ.get('RSI_HOPE_PARAMS', '').strip()
    if username:
        flags.append(f'--username={username}')
    if token:
        flags.append(f'--token={token}')
    if params:
        flags.append(f'--params={params}')
    return flags


def clone_argv(url: str) -> tuple:
    """Return the real argv and the argv safe to show or store."""
    public = ['hope', 'clone', url]
    return public + hope_credential_flags(), public


def login_argv() -> list:
    """Platform login. Clone's ``--username`` is a git credential, not this."""
    return ['hope', 'login', hope_username()]


def public_hope_command(argv: list) -> str:
    """Command safe to write in the launch log. Drops credential flags."""
    kept = []
    for token in argv or []:
        if token.startswith(('--username=', '--token=', '--params=')):
            continue
        kept.append(token)
    return redact(' '.join(kept))


def login_unconfirmed_message() -> str:
    """Shown when ``hope login`` timed out, failed, or no misid is configured."""
    misid = hope_username()
    if not misid:
        return _LOGIN_MISSING_USER
    return f'{_LOGIN_FAIL_PREFIX}请点重新拉取再发一次，或在终端执行 hope login {misid}'


def _is_failure_note(note: str) -> bool:
    text = note or ''
    return text == _TOOL_FAILURE or text.startswith('hope 未登录：') or text.startswith(_LOGIN_FAIL_PREFIX)


def hope_login_succeeded(result) -> bool:
    """Success is exit 0, or ``登录成功`` anywhere in the output.

    A confirmed login still prints the dx_confirm banner before ``登录成功``,
    so the banner alone is not a failure.
    """
    if result in (None, 'missing-tool'):
        return False
    text = _result_text(result)
    if '登录成功' in text:
        return True
    code = getattr(result, 'returncode', 1)
    return code == 0 and not hope_needs_login(text)


def hope_needs_login(text: str) -> bool:
    raw = text or ''
    lowered = raw.lower()
    if 'password input may be echoed' in lowered or 'authentication failed' in lowered:
        return True
    if 'dx_confirm' in lowered or 'hope_login_mode' in lowered:
        return True
    for marker in (
            'mis密码', '请输入密码', '用户名(misid)', '用户名（misid）', 'HOPE登录认证',
            '认证消息已发送至大象', '确认登录', '快速登录超时', '认证失败', '登录失败'):
        if marker in raw:
            return True
    return False


def reset_hope_login_state() -> None:
    """Drop cached login success and wake anyone waiting on an old attempt."""
    global _login_inflight, _login_ok, _login_failed_at, _login_generation
    with _login_mu:
        _login_inflight = False
        _login_ok = False
        _login_failed_at = 0.0
        _login_generation += 1
        _login_cv.notify_all()


def _login_cooled_down(force: bool) -> bool:
    if force or not _login_failed_at:
        return False
    return time.monotonic() - _login_failed_at < LOGIN_COOLDOWN


def _claim_login() -> int:
    global _login_inflight, _login_ok
    _login_inflight = True
    _login_ok = False
    return _login_generation


def _invalidate_login() -> None:
    global _login_ok
    with _login_mu:
        _login_ok = False


def _cooldown_login() -> None:
    global _login_ok, _login_failed_at
    with _login_mu:
        _login_ok = False
        _login_failed_at = time.monotonic()


def describe_startup() -> str:
    """One line for process start. Presence only, never the secret values."""
    username = 'yes' if hope_username() else 'no'
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
        if progress == LOGIN_WAITING:
            return FetchStatus(kind, repo_id_of(repo_id), str(path), '同步中', LOGIN_WAITING)
        if _is_failure_note(note) and not inflight:
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
        """重新拉取: log in again if needed, then clone. Does not call hope list."""
        repo = repo_id_of(repo_id)
        if model_source() != 'hope' or not is_repo_id(repo):
            return self.status_of(kind, repo) if is_repo_id(repo) else FetchStatus(kind, repo, '', '保持', '')
        current = self.status_of(kind, repo)
        if current.state == '就绪':
            return current
        inline = os.environ.get('RSI_FETCH_INLINE', '').strip() == '1'
        self._start_clone(kind, repo, background=background and not inline, force_login=True)
        latest = self.status_of(kind, repo)
        if latest.state in {'就绪', '失败'}:
            return latest
        message = latest.message if latest.state == '同步中' and latest.message else '同步中'
        return FetchStatus(kind, repo, latest.local_path, '同步中', message)

    def _start_clone(self, kind: str, repo_id: str, background: bool, force_login: bool = False) -> None:
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
        root = self.model_root() if kind == 'model' else self.data_root()
        root.mkdir(parents=True, exist_ok=True)

        def job():
            ready_now = False
            try:
                argv, _public = clone_argv(self._url_for(kind, repo_id))
                # hope clone writes <org>/<name> under its cwd. Do not cd into org/ first.
                result = self._run(argv, cwd=root, target=path, repo_key=key, force_login=force_login)
                if result == 'missing-tool':
                    with _LOCK:
                        self._notes[key] = _TOOL_FAILURE
                elif result in ('login-failed', 'no-user'):
                    pass
                elif hope_needs_login(_result_text(result)):
                    self._note_login_failure(key)
                else:
                    with _LOCK:
                        self._notes.pop(key, None)
            finally:
                with _LOCK:
                    _INFLIGHT.discard(key)
                    if _ready(kind, path):
                        self._notes.pop(key, None)
                        self._progress.pop(key, None)
                        ready_now = True
                    elif not _is_failure_note(self._notes.get(key, '')):
                        self._notes[key] = '同步中'
            if ready_now:
                logger.info('hope 就绪 %s %s %s', kind, repo_id, path)

        if background:
            threading.Thread(target=job, daemon=True).start()
            return
        job()

    def _mark_waiting(self, repo_key: str | None, *, announce: bool) -> None:
        if repo_key:
            with _LOCK:
                self._progress[repo_key] = LOGIN_WAITING
        if announce:
            logger.info('%s', LOGIN_WAITING)

    def _clear_login_failure(self, repo_key: str | None) -> None:
        if not repo_key:
            return
        with _LOCK:
            if _is_failure_note(self._notes.get(repo_key, '')):
                self._notes.pop(repo_key, None)
            if self._progress.get(repo_key) == LOGIN_WAITING:
                self._progress.pop(repo_key, None)

    def _note_login_failure(self, repo_key: str | None) -> None:
        message = login_unconfirmed_message()
        if not repo_key:
            logger.info('%s', message)
            return
        with _LOCK:
            changed = self._notes.get(repo_key) != message
            self._notes[repo_key] = message
            self._progress.pop(repo_key, None)
        if changed:
            logger.info('%s', message)

    def _ensure_login(self, *, force: bool, repo_key: str | None, cwd: Path) -> str:
        """Run ``hope login`` once. Other fetches wait on that attempt.

        A cached success is reused until a later command prints the login banner.
        A failure or timeout is not repeated until ``LOGIN_COOLDOWN`` passes.
        ``force`` is the manual 重新拉取 path and skips that cooldown. stdin stays
        closed and the process is killed at ``LOGIN_TIMEOUT`` so a missed
        大象 confirm cannot fall through to getpass.
        """
        if not hope_username():
            self._note_login_failure(repo_key)
            return 'no-user'
        while True:
            with _login_mu:
                if _login_inflight:
                    role = 'wait'
                elif _login_ok:
                    role = 'ok'
                elif _login_cooled_down(force):
                    role = 'failed'
                else:
                    role = 'run'
                    generation = _claim_login()
            if role == 'wait':
                self._mark_waiting(repo_key, announce=False)
                with _login_mu:
                    if _login_inflight:
                        _login_cv.wait()
                force = False
                continue
            if role == 'ok':
                self._clear_login_failure(repo_key)
                return 'ok'
            if role == 'failed':
                self._note_login_failure(repo_key)
                return 'login-failed'
            self._mark_waiting(repo_key, announce=True)
            return self._finish_login_attempt(cwd, repo_key, generation)

    def _finish_login_attempt(self, cwd: Path, repo_key: str | None, generation: int) -> str:
        global _login_inflight, _login_ok, _login_failed_at
        missing = False
        ok = False
        error = None
        try:
            result = self._execute(login_argv(), cwd=cwd, timeout=LOGIN_TIMEOUT)
            missing = result == 'missing-tool'
            ok = (not missing) and hope_login_succeeded(result)
        except Exception as exc:
            error = exc
        finally:
            with _login_mu:
                if generation == _login_generation:
                    _login_inflight = False
                    if ok:
                        _login_ok = True
                        _login_failed_at = 0.0
                    else:
                        _login_ok = False
                        if not missing:
                            _login_failed_at = time.monotonic()
                _login_cv.notify_all()
        if error is not None:
            self._clear_login_failure(repo_key)
            raise error
        if missing:
            self._clear_login_failure(repo_key)
            return 'missing-tool'
        if ok:
            self._clear_login_failure(repo_key)
            return 'ok'
        self._note_login_failure(repo_key)
        return 'login-failed'

    def _run(self, argv: list, cwd: Path, target: Path | None = None, *,
             repo_key: str | None = None, force_login: bool = False, allow_login_retry: bool = True):
        kind = argv[1] if len(argv) >= 2 else ''
        if kind == 'clone':
            outcome = self._ensure_login(force=force_login, repo_key=repo_key, cwd=cwd)
            if outcome != 'ok':
                return outcome
            result = self._execute_clone(argv, cwd=cwd, target=target)
        else:
            result = self._execute(argv, cwd=cwd, target=target)
        if kind != 'clone' or not hope_needs_login(_result_text(result)):
            return result
        if not allow_login_retry:
            self._note_login_failure(repo_key)
            _cooldown_login()
            return 'login-failed'
        _invalidate_login()
        outcome = self._ensure_login(force=False, repo_key=repo_key, cwd=cwd)
        if outcome != 'ok':
            return outcome
        return self._run(
            argv, cwd, target, repo_key=repo_key, force_login=False, allow_login_retry=False)

    def _execute_clone(self, argv: list, cwd: Path, target: Path | None = None):
        """Stream hope clone into the process log as bytes arrive."""
        command = public_hope_command(argv)
        where = f'cwd={cwd}'
        if target is not None:
            where += f' target={target}'
        logger.info('hope 开始 %s %s', command, where)
        emitted = False
        pieces = []

        def on_chunk(data):
            nonlocal emitted
            emitted = True
            text = redact(_as_text(data))
            pieces.append(text)
            append_process_log(text)

        try:
            if self.runner is subprocess.run:
                code = stream_subprocess(argv, cwd, on_chunk)
                result = _HopeResult(code, ''.join(pieces), '')
            else:
                completed = self.runner(
                    argv,
                    cwd=str(cwd),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    check=False,
                    on_chunk=on_chunk,
                )
                if completed == 'missing-tool':
                    logger.info('hope 结束 %s %s exit=缺失', command, where)
                    return completed
                if not emitted:
                    text = redact(_result_text(completed))
                    if text:
                        append_process_log(text)
                        pieces.append(text)
                code = getattr(completed, 'returncode', 1)
                stderr = '' if emitted else _as_text(getattr(completed, 'stderr', '') or '')
                stdout = ''.join(pieces) if emitted else _as_text(getattr(completed, 'stdout', '') or '')
                result = _HopeResult(code, stdout, stderr)
        except FileNotFoundError:
            _warn_missing_hope()
            logger.info('hope 结束 %s %s exit=缺失', command, where)
            return 'missing-tool'
        logger.info('hope 结束 %s %s exit=%s', command, where, getattr(result, 'returncode', 1))
        return result

    def _execute(self, argv: list, cwd: Path, target: Path | None = None, timeout: int | None = None):
        command = public_hope_command(argv)
        kwargs = {
            'cwd': str(cwd),
            'capture_output': True,
            'text': True,
            'check': False,
            'stdin': subprocess.DEVNULL,
        }
        if timeout is not None:
            kwargs['timeout'] = timeout
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


_LOG_APPEND = threading.Lock()


def process_log_path() -> Path:
    """Same file ``/server-log`` tails. ``run.sh`` also redirects this process there."""
    from rsi.app import server_log_path
    return server_log_path()


def append_process_log(text: str) -> None:
    """Append clone output immediately so the process log shows it before hope exits."""
    if not text:
        return
    data = text.encode('utf-8') if isinstance(text, str) else bytes(text)
    path = process_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_APPEND:
        with path.open('ab', buffering=0) as handle:
            handle.write(data)
            handle.flush()


def stream_subprocess(argv, cwd: Path, on_chunk) -> int:
    """Read a process in 4KB chunks and hand each chunk off before the next read."""
    proc = subprocess.Popen(
        list(argv),
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    try:
        stream = proc.stdout
        while stream is not None:
            block = stream.read(4096)
            if not block:
                break
            on_chunk(block)
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def kick_hope_login(runner=None) -> None:
    """Start one non-blocking ``hope login`` when this process serves hope."""
    if model_source() != 'hope' or not hope_username():
        return
    fetcher = ModelFetcher(Path.cwd(), runner=runner)

    def job():
        try:
            fetcher._ensure_login(force=False, repo_key=None, cwd=Path.cwd())
        except Exception:
            logger.exception('hope 登录')

    threading.Thread(target=job, daemon=True, name='hope-login').start()


def benchmark_dir(home: Path, name: str) -> Path:
    return _root('RSI_DATA_ROOT', 'data_root', home, 'datasets') / 'benchmarks' / name
