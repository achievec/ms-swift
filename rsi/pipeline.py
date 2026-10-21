# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stage machine for one RSI run.

The trunk is SFT → SFT eval. Dataset settings live on SFT. After that eval
the user chooses RL, another round, or an end. RL eval offers the same kind
of choice. A failed eval cannot start RL.
"""
import json
import logging
import re
import shutil
import threading
import uuid
from pathlib import Path

import yaml

from rsi.backends import BackendError, swift_launch
from rsi.eval_report import inside_directory, merge_eval_report
from rsi.commands import (
    STAGE_COMMAND,
    apply_more_params,
    apply_process_env,
    build_argv,
    env_prefix,
    format_shell,
    stage_shell,
)
from rsi.llm_infer_spec import chat_request_body, cuda_visible_devices, template_default_system
from rsi.dataset_validate import MAX_BYTES, validate_file, validate_public_id
from rsi.flow import ACTION_LABELS, assemble_flow, blocked_message, decision_actions, stage_unlocked
from rsi.logtail import announce_log, read_log_window, slice_bytes
from rsi.model_fetch import (
    ModelFetcher,
    activate_stage_log,
    benchmark_dir,
    dataset_block_message,
    deactivate_stage_log,
    hf_cloneable,
    hope_dataset_cloneable,
    is_repo_id,
    model_source,
    reset_stage_log,
    split_dataset_sample,
    use_stage_log,
)
from rsi.schemas import STAGE_META, STAGE_ORDER, presets
from rsi.webui_catalog import match_model
from rsi.serving import find_free_port, loopback, port_is_free
from rsi.store import Store

LABELS = {item['id']: item['label'] for item in STAGE_META}
logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent / 'examples'
SWIFT_STAGES = {'sft', 'sft_eval', 'rl', 'rl_eval'}
CHECKPOINT_RESET = {
    'sft': ['sft', 'rl'],
    'sft_eval': ['rl'],
    'rl': ['rl'],
    'rl_eval': [],
    'promote': [],
    'next_round': [],
}


class _ExportProcs:
    """In-flight ``swift export`` processes, one per experiment."""

    def __init__(self):
        self._lock = threading.Lock()
        self._procs = {}

    def track(self, run_id: str, proc) -> None:
        with self._lock:
            previous = self._procs.get(run_id)
            self._procs[run_id] = proc
        if previous is not None and previous.poll() is None and previous is not proc:
            previous.terminate()

    def running(self, run_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(run_id)
        return proc is not None and proc.poll() is None

    def stop(self, run_id: str) -> bool:
        with self._lock:
            proc = self._procs.pop(run_id, None)
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
        return True


class PipelineError(Exception):

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _loads(text, fallback=None):
    if not text:
        return {} if fallback is None else fallback
    return json.loads(text)


def _merge(defaults: dict, user: dict) -> dict:
    incoming = dict(user or {})
    extra_yaml = incoming.pop('extra_yaml', None)
    merged = dict(defaults)
    for key, value in incoming.items():
        if value is None or value == '':
            continue
        merged[key] = value
    if extra_yaml and str(extra_yaml).strip():
        try:
            loaded = yaml.safe_load(extra_yaml)
        except yaml.YAMLError as exc:
            raise PipelineError(f'额外 YAML 无法解析：{exc}') from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise PipelineError('额外 YAML 必须是键值映射')
        for key, value in loaded.items():
            if value is not None:
                merged[key] = value
        merged['extra_yaml'] = extra_yaml
    return merged


def _clean_draft(params: dict) -> dict:
    """One stage snapshot. Locked identity stays on the experiment, not in the draft."""
    cleaned = {}
    for key, value in (params or {}).items():
        if not isinstance(key, str) or not key or key.startswith('_'):
            continue
        if key in {'model', 'model_type', 'template', 'trained_model'}:
            continue
        if isinstance(value, dict):
            continue
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, (dict, list)):
                    continue
                text = str(item).strip()
                if text:
                    parts.append(text)
            cleaned[key] = ' '.join(parts)
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            cleaned[key] = value
            continue
        if isinstance(value, (int, float)):
            cleaned[key] = value
            continue
        cleaned[key] = str(value)
    return cleaned


def _metric(step, name):
    if not step or step.get('status') != 'succeeded' or not step.get('result_json'):
        return None
    metrics = _loads(step['result_json']).get('metrics') or {}
    value = metrics.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _format_errors(report: dict) -> str:
    parts = []
    for item in report.get('errors') or []:
        if item.get('row'):
            parts.append(f"第 {item['row']} 行：{item['error']}")
        else:
            parts.append(item['error'])
    return '数据格式不符合要求：' + ('；'.join(parts) if parts else '未知错误')


def _is_adapter_checkpoint(path: Path) -> bool:
    """LoRA (and mock) checkpoints have no full ``config.json``."""
    if not path.is_dir():
        return False
    if (path / 'adapter_config.json').is_file():
        return True
    return not (path / 'config.json').is_file()


def _model_from_args(path: Path) -> str:
    args_path = path / 'args.json'
    if not args_path.is_file():
        return ''
    try:
        data = json.loads(args_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return ''
    if not isinstance(data, dict):
        return ''
    for key in ('model', 'model_id_or_path'):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


def _path_key(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except OSError:
        return str(path)


def _adapter_base_model(checkpoint: Path, fallback: str) -> str:
    """Base ``--model`` from the checkpoint args, else the RSI run base."""
    seen = set()
    current = checkpoint
    while True:
        key = _path_key(current)
        if key in seen:
            break
        seen.add(key)
        recorded = _model_from_args(current)
        if not recorded or _path_key(Path(recorded)) in seen:
            break
        candidate = Path(recorded).expanduser()
        if candidate.is_dir() and _is_adapter_checkpoint(candidate):
            current = candidate
            continue
        return recorded
    text = (fallback or '').strip()
    if not text:
        return ''
    fallback_path = Path(text).expanduser()
    try:
        if fallback_path.is_dir() and _is_adapter_checkpoint(fallback_path) and _path_key(fallback_path) == _path_key(
                checkpoint):
            return ''
    except OSError:
        return text
    return text


def _slug(name: str, round_index: int, kind: str) -> str:
    ascii_name = re.sub(r'[^A-Za-z0-9]+', '-', name or '').strip('-').lower()
    if not ascii_name:
        ascii_name = 'rsi-model'
    return f'{ascii_name}-r{round_index}-{kind or "ckpt"}'[:64]


def materialize_general_qa(source: Path, dest_dir: Path):
    """Turn a messages/query jsonl into the folder layout `swift eval --eval_dataset general_qa` expects."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open(encoding='utf-8') as handle, (dest_dir / 'itops.jsonl').open('w', encoding='utf-8') as target:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get('query') and item.get('response'):
                query, response = item['query'], item['response']
            else:
                query = ''
                response = ''
                for message in item.get('messages') or []:
                    if not isinstance(message, dict):
                        continue
                    role = message.get('role')
                    content = message.get('content') or ''
                    if role == 'user' and not query:
                        query = content
                    elif role == 'assistant':
                        response = content
            if not query or not response:
                continue
            target.write(json.dumps({'query': query, 'response': response}, ensure_ascii=False) + '\n')
            written += 1
    if written == 0:
        raise PipelineError(f'评测文件里没有问答：{source}')
    return dest_dir, written


def _resolve_existing(value):
    if not isinstance(value, str) or not value.strip():
        return value
    path = Path(value)
    if path.is_file():
        return str(path.resolve())
    candidate = REPO_ROOT / value
    if candidate.is_file():
        return str(candidate.resolve())
    return value


class Pipeline:

    def __init__(self, home: Path, backend):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.store = Store(self.home / 'rsi.sqlite')
        self.store.recover_interrupted()
        self.examples = EXAMPLES
        self._resume_mu = threading.Lock()
        self._resume_busy = False
        self._resume_again = False
        self.fetcher = ModelFetcher(self.home, on_settled=self._continue_after_fetch)
        from rsi.serving import ServeManager
        self.serve = ServeManager()
        self._exports = _ExportProcs()

    def stage_dir(self, run_id: str, round_index: int, stage: str) -> Path:
        path = self.home / 'runs' / run_id / f'round_{round_index}' / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_log_dir(self, run_id: str, round_index: int, stage: str) -> Path:
        """Where this stage's swift.log lives. Eval uses ``eval_output`` under the stage."""
        path = self.stage_dir(run_id, round_index, stage)
        if stage in {'sft_eval', 'rl_eval'}:
            path = path / 'eval_output'
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _concrete_output(self, run_id: str, round_index: int, stage: str, raw, eval_output: bool) -> str:
        """Keep ``.`` and relative output dirs inside the experiment. Absolute paths stay."""
        fallback = self.stage_log_dir(run_id, round_index, stage) if eval_output else self.stage_dir(
            run_id, round_index, stage)
        text = str(raw or '').strip()
        path = Path(text).expanduser() if text else None
        if path is None or not path.is_absolute():
            return str(fallback)
        return str(path)

    def create_run(self, name: str, scenario: str, base_model: str, model_type: str = '', template: str = '') -> dict:
        name = (name or '').strip()
        scenario = (scenario or '').strip()
        base_model = (base_model or '').strip()
        matched = match_model(base_model) or {}
        model_type = (model_type or matched.get('model_type') or '').strip()
        template = (template or matched.get('template') or '').strip()
        if not name or len(name) > 80:
            raise PipelineError('实验名称需要 1 到 80 个字符')
        if not scenario or len(scenario) > 500:
            raise PipelineError('场景说明需要 1 到 500 个字符')
        if not base_model or len(base_model) > 300:
            raise PipelineError('基座模型需要 1 到 300 个字符')
        run_id = uuid.uuid4().hex[:8]
        self.store.create_run(run_id, name, scenario, base_model, model_type, template)
        return self.view(run_id)

    def _apply_model_meta(self, preset: dict, run: dict):
        if run.get('model_type') and not preset.get('model_type'):
            preset['model_type'] = run['model_type']
        if run.get('template') and not preset.get('template'):
            preset['template'] = run['template']

    def list_runs(self) -> list:
        return self.store.list_runs()

    def delete_run(self, run_id: str) -> dict:
        if not self.store.get_run(run_id):
            raise PipelineError('实验不存在', 404)
        self.serve.stop(run_id)
        deleted = self.store.delete_run(run_id)
        if not deleted:
            raise PipelineError('实验不存在', 404)
        return {'ok': True, 'id': run_id, 'workspace_removed': self._remove_run_workspace(run_id)}

    def _remove_run_workspace(self, run_id: str) -> bool:
        """Remove only this experiment's directory under RSI home/runs."""
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id or ''):
            return False
        root = (self.home / 'runs').resolve()
        target = (root / run_id).resolve()
        if target.parent != root or not target.is_dir():
            return False
        shutil.rmtree(target)
        return True

    def _require_run(self, run_id: str) -> dict:
        run = self.store.get_run(run_id)
        if not run:
            raise PipelineError('实验不存在', 404)
        return run

    def save_upload(self, filename: str, data: bytes) -> dict:
        if len(data) > MAX_BYTES:
            raise PipelineError('文件超过 20MB')
        suffix = Path(filename or 'data.jsonl').suffix.lower()
        if suffix not in {'.jsonl', '.json', '.txt'}:
            suffix = '.jsonl'
        dest_dir = self.home / 'uploads'
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f'{uuid.uuid4().hex}{suffix}'
        dest.write_bytes(data)
        report = validate_file(dest)
        return {'path': str(dest), 'filename': filename, 'validation': report}

    def _upload_path(self, raw: str) -> Path:
        uploads = (self.home / 'uploads').resolve()
        path = Path(raw).resolve()
        if path != uploads and uploads not in path.parents:
            raise PipelineError('上传文件必须位于上传目录')
        if not path.is_file():
            raise PipelineError('上传文件不存在')
        return path

    def resolve_dataset(self, params: dict, *, fetch: bool = True):
        source = params.get('source') or 'bundled'
        if source == 'bundled':
            name = params.get('bundled') or 'itops_mix'
            path = self.examples / f'{name}.jsonl'
            if not path.is_file():
                raise PipelineError(f'未知内置数据集：{name}')
            return str(path.resolve()), validate_file(path)
        if source == 'public':
            return self._resolve_public(params.get('dataset') or '', fetch=fetch)
        if source == 'upload':
            path = self._upload_path(params.get('dataset') or '')
            return str(path), validate_file(path)
        if source == 'path':
            raw = params.get('dataset') or ''
            path = Path(raw)
            if not path.is_file():
                raise PipelineError(f'本地文件不存在：{raw}')
            return str(path.resolve()), validate_file(path)
        raise PipelineError('未知数据来源')

    def _existing_dataset_path(self, raw: str) -> Path | None:
        if not raw or '://' in raw:
            return None
        path = Path(raw).expanduser()
        try:
            if path.is_file() or path.is_dir():
                return path.resolve()
        except OSError:
            return None
        return None

    def _directory_dataset_report(self, path: str) -> dict:
        return {
            'ok': True,
            'num_rows': None,
            'formats': {'dataset': 1},
            'errors': [],
            'error_count': 0,
            'warnings': [],
            'preview': [],
            'dataset': path,
        }

    def _resolve_public_one(self, raw: str, *, fetch: bool = True):
        raw = str(raw or '').strip()
        base, sample = split_dataset_sample(raw)
        existing = self._existing_dataset_path(base if sample else raw)
        if existing is not None:
            shown = str(existing)
            if sample:
                shown = f'{shown}#{sample}'
            if existing.is_file():
                report = validate_file(existing)
                report['dataset'] = shown
                return shown, report
            return shown, self._directory_dataset_report(shown)
        blocked = dataset_block_message(raw, kind='dataset')
        if blocked:
            raise PipelineError(blocked)
        cloneable = (model_source() == 'hope' and hope_dataset_cloneable(raw)) or (
            model_source() == 'local' and hf_cloneable(raw))
        if cloneable:
            repo = base if sample else raw
            status = self.fetcher.ensure('dataset', repo, background=True) if fetch else self.fetcher.status_of(
                'dataset', repo)
            self._note_fetch(status)
            if status.state == '就绪':
                shown = status.local_path
                if sample:
                    shown = f'{shown}#{sample}'
                local = Path(status.local_path)
                if local.is_file():
                    report = validate_file(local)
                    report['dataset'] = shown
                    return shown, report
                return shown, self._directory_dataset_report(shown)
            report = validate_public_id(repo)
            report['ok'] = False
            report['errors'] = [{'row': 0, 'error': status.message or '同步中'}]
            report['error_count'] = 1
            report['fetch_state'] = status.state
            report['fetch_repo'] = status.repo_id or repo
            return status.local_path, report
        report = validate_public_id(raw)
        return report.get('dataset') or '', report

    def _resolve_public(self, raw: str, *, fetch: bool = True):
        text = str(raw or '').strip()
        parts = [part for part in text.split() if part] if ' ' in text else [text]
        if len(parts) <= 1:
            return self._resolve_public_one(parts[0] if parts else '', fetch=fetch)
        resolved = []
        warnings = []
        errors = []
        syncing = None
        repos = []
        for part in parts:
            path, report = self._resolve_public_one(part, fetch=fetch)
            if report.get('fetch_state') == '同步中':
                repo = report.get('fetch_repo') or ''
                if repo and repo not in repos:
                    repos.append(repo)
                if syncing is None:
                    syncing = (path, report)
                continue
            if not report.get('ok'):
                errors.extend(report.get('errors') or [])
            warnings.extend(report.get('warnings') or [])
            resolved.append(path or part)
        if syncing is not None:
            path, report = syncing
            report = dict(report)
            report['fetch_repos'] = repos
            return path, report
        joined = ' '.join(resolved)
        return joined, {
            'ok': not errors,
            'num_rows': None,
            'formats': {'dataset': len(parts)},
            'errors': errors,
            'error_count': len(errors),
            'warnings': warnings,
            'preview': [],
            'dataset': joined,
        }

    def _require_dataset(self, dataset: str) -> None:
        path = Path(dataset)
        parts = [dataset] if path.is_file() or path.is_dir() else str(dataset).split()
        for part in parts:
            if not part:
                continue
            base, sample = split_dataset_sample(part)
            item = Path(base if sample else part)
            if item.is_file():
                report = validate_file(item)
                if not report['ok']:
                    raise PipelineError(_format_errors(report))
                continue
            if item.is_dir():
                continue
            blocked = dataset_block_message(part, kind='dataset')
            if blocked:
                raise PipelineError(blocked)
            report = validate_public_id(part)
            if not report['ok']:
                raise PipelineError(_format_errors(report))

    def _note_fetch(self, status) -> None:
        if status is None or status.state in {'保持', '缺少'}:
            return
        self.store.upsert_fetch(status.as_row())

    def _place_trained_checkpoint(self, merged: dict, run: dict) -> None:
        """Hand a LoRA checkpoint to swift as ``--adapters``, not ``--model``.

        A directory with ``adapter_config.json``, or without ``config.json``, is an
        adapter. The base ``--model`` comes from that checkpoint's ``args.json``.
        When args.json has no model, the run's base model is used. A full
        checkpoint (``config.json`` and no adapter file) stays ``--model``.
        A LoRA on ``adapters`` keeps that path and takes ``--model`` from the
        checkpoint ``args.json``. A full checkpoint remembered as
        ``trained_model`` becomes ``--model``. An ``adapters`` value that is
        not a LoRA directory is left as the form wrote it. When ``--model``
        itself is an adapter directory (the next round's base), it moves to
        ``--adapters``.
        """
        chosen = str(merged.get('adapters') or '').strip()
        trained = str(merged.get('trained_model') or '').strip()
        raw = str(merged.get('model') or '').strip()
        fallback = (run or {}).get('base_model') or ''
        if chosen:
            path = Path(chosen).expanduser()
            try:
                adapter = path.is_dir() and _is_adapter_checkpoint(path)
            except OSError:
                adapter = False
            if adapter:
                base = _adapter_base_model(path, fallback)
                if base:
                    merged['model'] = base
                self._pin_identity(merged, run)
                return
            model_path = Path(raw).expanduser() if raw else None
            model_is_adapter = False
            if model_path is not None:
                try:
                    model_is_adapter = model_path.is_dir() and _is_adapter_checkpoint(model_path)
                except OSError:
                    model_is_adapter = False
            if model_is_adapter or not raw:
                base = _adapter_base_model(model_path or path, fallback)
                if base:
                    merged['model'] = base
            self._pin_identity(merged, run)
            return
        if trained:
            path = Path(trained).expanduser()
            try:
                full = path.is_dir() and (path / 'config.json').is_file() and not (
                    path / 'adapter_config.json').is_file()
            except OSError:
                full = False
            if full:
                merged['model'] = str(path)
                merged['adapters'] = ''
                self._pin_identity(merged, run)
                return
        if not raw:
            return
        path = Path(raw).expanduser()
        try:
            is_dir = path.is_dir()
        except OSError:
            return
        if not is_dir or not _is_adapter_checkpoint(path):
            return
        base = _adapter_base_model(path, (run or {}).get('base_model') or '')
        merged['adapters'] = str(path)
        if base:
            merged['model'] = base
        else:
            merged['model'] = ''
        self._pin_identity(merged, run)

    def _lock_model_fields(self, merged: dict, defaults: dict) -> None:
        """Keep --model, --model_type, and --template on the values chosen at 新建实验."""
        for key in ('model', 'model_type', 'template'):
            if key in defaults:
                merged[key] = defaults[key]

    def _pin_identity(self, merged: dict, run: dict) -> None:
        model = str(merged.get('model') or '')
        meta = match_model(model) or {}
        if not merged.get('model_type'):
            merged['model_type'] = (run or {}).get('model_type') or meta.get('model_type') or ''
        if not merged.get('template'):
            merged['template'] = (run or {}).get('template') or meta.get('template') or ''

    def _rewrite_model(self, merged: dict, run: dict, side: dict, strict: bool, *, fetch: bool = True) -> None:
        if model_source() not in {'hope', 'local'}:
            return
        model = str(merged.get('model') or '')
        if not model or not is_repo_id(model):
            self._pin_identity(merged, run)
            return
        blocked = dataset_block_message(model)
        if blocked:
            raise PipelineError(blocked)
        self._pin_identity(merged, run)
        status = self.fetcher.ensure('model', model, background=True) if fetch else self.fetcher.status_of(
            'model', model)
        self._note_fetch(status)
        local = Path(status.local_path).expanduser() if status.local_path else None
        if local is not None and not is_repo_id(status.local_path):
            if not local.is_absolute():
                local = local.resolve()
            merged['model'] = str(local)
        else:
            merged['model'] = status.local_path
        if status.state == '同步中':
            side['fetch_state'] = '同步中'
            side['fetch_repo'] = status.repo_id
            repos = list(side.get('fetch_repos') or [])
            item = {'kind': 'model', 'repo_id': status.repo_id}
            if item not in repos:
                repos.append(item)
            side['fetch_repos'] = repos
        elif strict and status.state == '失败':
            raise PipelineError(status.message or '本地没有该文件')
        elif strict and not merged.get('model_type'):
            raise PipelineError('请填写模型类型和模板')

    def _block_hub_model(self, stage: str, merged: dict, side: dict) -> bool:
        """Hope and local must not start swift while --model is still a hub id."""
        if stage not in SWIFT_STAGES or model_source() not in {'hope', 'local'}:
            return False
        model = str(merged.get('model') or '')
        blocked = dataset_block_message(model)
        if blocked:
            raise PipelineError(blocked)
        if is_repo_id(model):
            side['fetch_state'] = '同步中'
            side.setdefault('fetch_repo', model)
            return True
        return False

    def _attach_eval_locals(self, merged: dict, strict: bool) -> None:
        if model_source() not in {'hope', 'local'}:
            return
        raw = str(merged.get('eval_dataset') or '')
        if not raw or raw == 'itops-heldout' or raw == 'general_qa':
            return
        names = [part for part in raw.split() if part]
        extra = merged.get('eval_dataset_args') or {}
        if isinstance(extra, str) and extra.strip():
            extra = json.loads(extra)
        if not isinstance(extra, dict):
            extra = {}
        missing = []
        for name in names:
            spec = dict(extra.get(name) or {})
            local = spec.get('local_path') or ''
            if local and Path(local).exists():
                extra[name] = spec
                continue
            candidate = benchmark_dir(self.home, name)
            if candidate.is_dir() and any(candidate.iterdir()):
                spec['local_path'] = str(candidate)
                extra[name] = spec
            else:
                missing.append(name)
        if extra:
            merged['eval_dataset_args'] = extra
        if strict and missing:
            raise PipelineError('评测集 ' + ', '.join(missing) + ' 需要放在本地后再运行')

    def validate_request(self, params: dict) -> dict:
        # Format checks must not start hope clone; that would race the stage log writer.
        _dataset, report = self.resolve_dataset(params or {}, fetch=False)
        return report

    def _sft_params(self, run_id: str, round_index: int) -> dict:
        step = self.store.latest_active_steps(run_id, round_index).get('sft')
        if not step or step['status'] != 'succeeded':
            return {}
        return _loads(step['params_json'])

    def _seed_dataset(self, preset: dict) -> None:
        """SFT keeps the data-stage source and split. The dataset name stays empty until chosen."""
        data = presets().get('data') or {}
        if not preset.get('source'):
            preset['source'] = data.get('source') or 'public'
        if preset.get('dataset') in (None, ''):
            preset['dataset'] = ''
        for key in ('split_dataset_ratio', 'padding_free'):
            if key not in preset and data.get(key) not in (None, ''):
                preset[key] = data[key]

    def _carry_dataset(self, preset: dict, run: dict, rnd: dict) -> None:
        if rnd['round_index'] <= 1:
            return
        previous = self.store.latest_active_steps(run['id'], rnd['round_index'] - 1)
        nxt = previous.get('next_round')
        reuse = True
        if nxt and nxt['status'] == 'succeeded':
            reuse = bool(_loads(nxt['params_json']).get('reuse_dataset', True))
        prev = previous.get('sft')
        if not reuse or not prev or prev['status'] != 'succeeded':
            return
        carried = _loads(prev['params_json'])
        for key in ('source', 'bundled', 'dataset', 'max_length', 'split_dataset_ratio', 'padding_free'):
            if carried.get(key) not in (None, ''):
                preset[key] = carried[key]
        dataset = str(preset.get('dataset') or '')
        if dataset and preset.get('source') == 'public' and ' ' not in dataset:
            try:
                if Path(dataset).expanduser().is_file():
                    preset['source'] = 'path'
            except OSError:
                pass

    def _remember_checkpoint(self, preset: dict, path: str) -> None:
        """Show the experiment model. A LoRA fills adapters; a full checkpoint is command-only."""
        preset['adapters'] = ''
        preset.pop('trained_model', None)
        if not path:
            return
        checkpoint = Path(path).expanduser()
        try:
            is_dir = checkpoint.is_dir()
        except OSError:
            return
        if not is_dir:
            return
        if _is_adapter_checkpoint(checkpoint):
            preset['adapters'] = str(path)
            return
        if (checkpoint / 'config.json').is_file():
            preset['trained_model'] = str(path)

    def _defaults_for(self, run: dict, rnd: dict, stage: str) -> dict:
        preset = dict(presets().get(stage) or {})
        if stage == 'sft':
            self._seed_dataset(preset)
            self._carry_dataset(preset, run, rnd)
            self._show_round_model(preset, run, rnd)
            preset['output_dir'] = str(self.stage_dir(run['id'], rnd['round_index'], 'sft'))
            return preset

        if stage == 'rl':
            self._show_round_model(preset, run, rnd)
            sft_ckpt = self.store.active_checkpoint(run['id'], rnd['round_index'], 'sft')
            self._remember_checkpoint(preset, sft_ckpt['path'] if sft_ckpt else '')
            sft_params = self._sft_params(run['id'], rnd['round_index'])
            if sft_params.get('dataset'):
                preset['dataset'] = sft_params['dataset']
            preset['output_dir'] = str(self.stage_dir(run['id'], rnd['round_index'], 'rl'))
            preset['external_plugins'] = _resolve_existing(preset.get('external_plugins'))
            return preset

        if stage in {'sft_eval', 'rl_eval'}:
            # Web-UI's eval dataset dropdown has no default. This loop's gate is the
            # bundled itops-heldout set, so the preset keeps that name.
            self._show_round_model(preset, run, rnd)
            kind = 'sft' if stage == 'sft_eval' else 'rl'
            checkpoint = self.store.active_checkpoint(run['id'], rnd['round_index'], kind)
            self._remember_checkpoint(preset, checkpoint['path'] if checkpoint else '')
            preset['eval_output_dir'] = str(self.stage_dir(run['id'], rnd['round_index'], stage) / 'eval_output')
            preset['custom_eval_config'] = _resolve_existing(preset.get('custom_eval_config'))
            return preset

        if stage == 'promote':
            self._show_round_model(preset, run, rnd)
            _candidates, recommendation = self.candidates(run, rnd)
            preset['choice'] = recommendation
            preset['reason'] = preset.get('reason') or ''
            return preset

        if stage == 'serve':
            preset.update(self._serve_targets(run, rnd))
            self._apply_model_meta(preset, run)
            if not str(preset.get('system') or '').strip():
                preset['system'] = template_default_system(str(preset.get('template') or ''))
            return preset
        self._show_round_model(preset, run, rnd)
        return preset

    def _show_round_model(self, preset: dict, run: dict, rnd: dict) -> None:
        preset['model'] = (rnd or {}).get('base_model') or (run or {}).get('base_model') or ''
        if (run or {}).get('model_type'):
            preset['model_type'] = run['model_type']
        elif not preset.get('model_type'):
            preset['model_type'] = ''
        if (run or {}).get('template'):
            preset['template'] = run['template']
        elif not preset.get('template'):
            preset['template'] = ''

    def _serve_checkpoint_path(self, run: dict, rnd: dict) -> tuple:
        """Promoted checkpoint, else this round's RL checkpoint, else SFT."""
        promo = self.store.latest_promotion(run['id'])
        if promo and promo.get('promoted_checkpoint'):
            return promo['promoted_checkpoint'], promo.get('promoted_kind') or 'ckpt', promo.get('round_index') or rnd.get(
                'round_index')
        round_index = (rnd or {}).get('round_index') or run.get('round_index')
        for kind in ('rl', 'sft'):
            row = self.store.active_checkpoint(run['id'], round_index, kind)
            if row and row.get('path'):
                return row['path'], kind, round_index
        return '', '', round_index

    def _serve_targets(self, run: dict, rnd: dict) -> dict:
        """Base ``--model`` plus LoRA ``--adapters`` for this round.

        An adapter directory keeps the run base (or the base recorded in
        ``args.json``) as ``--model``. A full checkpoint stays ``--model``.
        """
        path, kind, round_index = self._serve_checkpoint_path(run, rnd)
        base = str((run or {}).get('base_model') or '').strip()
        name = _slug(run.get('name') or '', round_index or 1, kind or 'model')
        if not path:
            return {'model': base, 'adapters': '', 'served_model_name': name, 'checkpoint': ''}
        checkpoint = Path(path).expanduser()
        try:
            is_adapter = checkpoint.is_dir() and _is_adapter_checkpoint(checkpoint)
            is_full = checkpoint.is_dir() and (checkpoint / 'config.json').is_file() and not (
                checkpoint / 'adapter_config.json').is_file()
        except OSError:
            is_adapter = False
            is_full = False
        if is_full:
            return {'model': str(path), 'adapters': '', 'served_model_name': name, 'checkpoint': str(path)}
        model = _adapter_base_model(checkpoint, base) if is_adapter else base
        return {
            'model': model or base,
            'adapters': str(path),
            'served_model_name': name,
            'checkpoint': str(path),
        }

    def _export_weights(self, run: dict, rnd: dict) -> tuple:
        """Latest LoRA or full checkpoint for this round, else a promoted directory."""
        round_index = (rnd or {}).get('round_index') or (run or {}).get('round_index')
        for kind in ('rl', 'sft'):
            row = self.store.active_checkpoint(run['id'], round_index, kind)
            if row and row.get('path'):
                return row['path'], kind
        promo = self.store.latest_promotion(run['id'])
        path = str((promo or {}).get('promoted_checkpoint') or '').strip()
        if path:
            try:
                if Path(path).expanduser().is_dir():
                    return path, (promo or {}).get('promoted_kind') or 'ckpt'
            except OSError:
                return '', ''
        return '', ''

    def _export_defaults(self, run: dict, rnd: dict) -> dict:
        """Locked experiment identity plus the weights this export should start from."""
        preset = {}
        self._show_round_model(preset, run, rnd)
        weights, _kind = self._export_weights(run, rnd)
        preset['adapters'] = weights
        preset['checkpoint'] = weights
        preset['merge_lora'] = False
        preset['device_map'] = ''
        preset['quant_bits'] = ''
        preset['quant_method'] = ''
        preset['quant_n_samples'] = 256
        preset['max_length'] = ''
        preset['dataset'] = ''
        preset['more_params'] = ''
        preset['gpu_id'] = '0'
        if weights:
            path = Path(weights).expanduser()
            try:
                adapter = path.is_dir() and _is_adapter_checkpoint(path)
            except OSError:
                adapter = False
            preset['merge_lora'] = bool(adapter)
            preset['output_dir'] = str(path.parent / f'{path.name}-merged')
        else:
            preset['output_dir'] = str(self.home / 'runs' / run['id'] / f'round_{rnd["round_index"]}' / 'export')
        return preset

    def _export_argv(self, merged: dict) -> list:
        """``swift export`` argv. A full checkpoint is ``--model``; a LoRA stays ``--adapters``."""
        params = dict(merged)
        weights = str(params.get('adapters') or '').strip()
        if weights:
            path = Path(weights).expanduser()
            try:
                full = path.is_dir() and (path / 'config.json').is_file() and not (path / 'adapter_config.json').is_file()
            except OSError:
                full = False
            if full:
                params['model'] = str(path)
                params['adapters'] = ''
                params.pop('merge_lora', None)
        if params.get('merge_lora') not in {True, 'true', 'True', 1, '1'}:
            params.pop('merge_lora', None)
        if not params.get('quant_method'):
            params.pop('quant_method', None)
            params.pop('quant_bits', None)
            if str(params.get('quant_n_samples') or '') in {'', '256'}:
                params.pop('quant_n_samples', None)
        return build_argv('export', params)

    def candidates(self, run: dict, rnd: dict):
        active = self.store.latest_active_steps(run['id'], rnd['round_index'])
        sft = self.store.active_checkpoint(run['id'], rnd['round_index'], 'sft')
        rl = self.store.active_checkpoint(run['id'], rnd['round_index'], 'rl')
        base_score = None
        if rnd['round_index'] > 1:
            previous = self.store.get_round(run['id'], rnd['round_index'] - 1)
            if previous and previous.get('promoted_score') is not None:
                base_score = previous['promoted_score']
        rows = [
            {
                'kind': 'base',
                'label': '当前基座',
                'path': rnd['base_model'],
                'score': base_score,
            },
            {
                'kind': 'sft',
                'label': 'SFT checkpoint',
                'path': sft['path'] if sft else None,
                'score': _metric(active.get('sft_eval'), 'accuracy'),
            },
            {
                'kind': 'rl',
                'label': 'RL checkpoint',
                'path': rl['path'] if rl else None,
                'score': _metric(active.get('rl_eval'), 'accuracy'),
            },
        ]
        preference = {'rl': 3, 'sft': 2, 'base': 1}
        best = None
        best_key = None
        for row in rows:
            if not row['path'] or row['score'] is None:
                continue
            key = (row['score'], preference[row['kind']])
            if best_key is None or key > best_key:
                best_key = key
                best = row['kind']
        if best is None:
            if rl:
                best = 'rl'
            elif sft:
                best = 'sft'
            else:
                best = 'base'
        return rows, best

    def _prepare(self, run, rnd, stage, user, strict: bool, *, fetch: bool = True):
        defaults = self._defaults_for(run, rnd, stage)
        merged = _merge(defaults, user)
        self._lock_model_fields(merged, defaults)
        if isinstance(merged.get('dataset'), list):
            merged['dataset'] = ' '.join(
                str(item).strip() for item in merged['dataset'] if str(item).strip())
        if 'more_params' in (user or {}):
            text = '' if user.get('more_params') is None else str(user.get('more_params'))
            merged['more_params'] = text
            try:
                merged = apply_more_params(merged)
            except ValueError as exc:
                raise PipelineError(str(exc)) from exc
            self._lock_model_fields(merged, defaults)
            if text.strip():
                merged['more_params'] = text
        side = {}
        if stage == 'sft' and merged.get('source'):
            public_blank = (
                str(merged.get('source')) == 'public' and not str(merged.get('dataset') or '').strip())
            if public_blank and strict:
                raise PipelineError('请先选择数据集')
            if public_blank:
                dataset, report = None, None
            else:
                try:
                    dataset, report = self.resolve_dataset(merged, fetch=fetch)
                except PipelineError:
                    if strict:
                        raise
                    dataset, report = None, None
            if dataset is None and report is None:
                pass
            else:
                merged['dataset'] = dataset
                side['validation'] = report
                if report.get('fetch_state') == '同步中':
                    dataset_repo = report.get('fetch_repo') or ''
                    names = list(report.get('fetch_repos') or ([] if not dataset_repo else [dataset_repo]))
                    side['fetch_repos'] = [
                        {'kind': 'dataset', 'repo_id': repo} for repo in names if repo]
                    self._rewrite_model(merged, run, side, strict, fetch=fetch)
                    if side.get('fetch_state') != '同步中':
                        side['fetch_state'] = '同步中'
                        side['fetch_repo'] = dataset_repo
                    return merged, side
                if strict and not report.get('ok'):
                    raise PipelineError(_format_errors(report))

        if stage in {'sft', 'rl'}:
            dataset = merged.get('dataset') or ''
            if strict:
                if not dataset:
                    raise PipelineError('请先选择数据集')
                self._require_dataset(str(dataset))
                if not merged.get('model'):
                    raise PipelineError('缺少模型路径')
            merged['output_dir'] = self._concrete_output(run['id'], rnd['round_index'], stage, merged.get('output_dir'),
                                                          False)
            if not dataset:
                merged['dataset'] = '<dataset>'
            self._rewrite_model(merged, run, side, strict, fetch=fetch)
            self._place_trained_checkpoint(merged, run)
            return merged, side

        if stage in {'sft_eval', 'rl_eval'}:
            custom = merged.get('custom_eval_config')
            if custom:
                path = Path(str(custom))
                if path.is_file():
                    report = validate_file(path)
                    if strict and not report['ok']:
                        raise PipelineError(_format_errors(report))
                    side['eval_rows'] = report.get('num_rows')
                    merged['custom_eval_config'] = str(path.resolve())
                elif strict:
                    raise PipelineError(f'自定义评测文件不存在：{custom}')
            merged['eval_output_dir'] = self._concrete_output(
                run['id'], rnd['round_index'], stage, merged.get('eval_output_dir'), True)
            if strict and not merged.get('model'):
                raise PipelineError('缺少待评测的 checkpoint')
            extra = merged.get('extra_eval_args') or {}
            if isinstance(extra, str) and extra.strip():
                extra = json.loads(extra)
            if not isinstance(extra, dict):
                extra = {}
            if self.backend.name == 'swift' and str(merged.get('eval_dataset') or 'itops-heldout') == 'itops-heldout':
                source = Path(str(merged.get('custom_eval_config') or ''))
                if not source.is_file():
                    source = EXAMPLES / 'itops_eval.jsonl'
                qa_dir, rows = materialize_general_qa(
                    source, self.stage_dir(run['id'], rnd['round_index'], stage) / 'general_qa')
                side['eval_rows'] = rows
                merged['eval_dataset'] = 'general_qa'
                merged['eval_dataset_args'] = {
                    'general_qa': {
                        'local_path': str(qa_dir),
                        'subset_list': ['itops'],
                    },
                }
                extra.pop('custom_eval_config', None)
            elif merged.get('custom_eval_config'):
                extra['custom_eval_config'] = merged['custom_eval_config']
            if extra:
                merged['extra_eval_args'] = extra
            else:
                merged.pop('extra_eval_args', None)
            if not merged.get('eval_dataset'):
                if strict:
                    raise PipelineError('请填写 eval_dataset')
                merged['eval_dataset'] = 'itops-heldout'
            self._rewrite_model(merged, run, side, strict, fetch=fetch)
            self._place_trained_checkpoint(merged, run)
            self._attach_eval_locals(merged, strict)
            return merged, side

        if stage == 'promote':
            rows, _recommendation = self.candidates(run, rnd)
            choice = (merged.get('choice') or '').strip()
            if strict and choice not in {'base', 'sft', 'rl'}:
                raise PipelineError('晋级选择必须是 base、sft 或 rl')
            picked = next((row for row in rows if row['kind'] == choice), None)
            if strict and (not picked or not picked.get('path')):
                raise PipelineError(f'没有可用的 {choice} 检查点')
            side['candidates'] = rows
            side['picked'] = picked
            return merged, side

        if stage == 'next_round':
            if strict and not rnd.get('promoted_checkpoint'):
                raise PipelineError('还没有晋级结果')
            return merged, side
        raise PipelineError('未知阶段')

    def _status_for(self, stage: str, active: dict, rnd: dict) -> str:
        step = active.get(stage)
        if step and step['status'] in {'succeeded', 'failed', 'running', 'syncing'}:
            return step['status']
        if stage_unlocked(stage, active, rnd):
            return 'ready'
        return 'locked'

    def _form_state(self, run, rnd, stage, active, recommendation, draft=None):
        defaults = self._defaults_for(run, rnd, stage)
        if stage == 'promote':
            defaults['choice'] = recommendation
        step = active.get(stage)
        step_params = None
        if step and step['status'] in {'succeeded', 'failed', 'running', 'syncing'}:
            step_params = _loads(step['params_json'])
        values = dict(defaults)
        if step_params:
            for key, value in step_params.items():
                if key in {'model', 'model_type', 'template', 'trained_model'}:
                    continue
                values[key] = value
        if draft:
            for key, value in draft.items():
                if key in {'model', 'model_type', 'template'} or isinstance(value, dict):
                    continue
                values[key] = value
        shown = {}
        self._show_round_model(shown, run, rnd)
        for key in ('model', 'model_type', 'template'):
            if shown.get(key) not in (None, ''):
                values[key] = shown[key]
        adapters_set = (
            (step_params is not None and 'adapters' in step_params) or (draft is not None and 'adapters' in draft))
        if not adapters_set and defaults.get('adapters') and not str(values.get('adapters') or '').strip():
            values['adapters'] = defaults['adapters']
        result = _loads(step['result_json']) if step and step.get('result_json') else None
        result = self._with_eval_report(run['id'], stage, result, values)
        status = self._status_for(stage, active, rnd)
        logs = step['logs'] if step else ''
        # Syncing/running keep writing hope + swift into the file. The step row is only a
        # snapshot from when status flipped; the panel must not roll back to that text.
        if step and status in {'syncing', 'running'}:
            live = self._live_form_logs(run['id'], rnd['round_index'], stage, step)
            if live:
                logs = live
        return {
            'status': status,
            'values': values,
            'preset': defaults,
            'result': result,
            'logs': logs,
            'command': step['command'] if step else '',
            'started_at': step['started_at'] if step else None,
            'ended_at': step['ended_at'] if step else None,
            'step_id': step['id'] if step else None,
            'backend': step['backend'] if step else None,
        }

    def _live_form_logs(self, run_id: str, round_index: int, stage: str, step: dict) -> str:
        """Latest normalized swift.log text for a stage that is still in flight."""
        path = self._stage_log_path(run_id, round_index, stage, step)
        if path is None or not path.is_file():
            return ''
        try:
            if path.stat().st_size <= 0:
                return ''
        except OSError:
            return ''
        return read_log_window(path, None, None).get('text') or ''

    def _with_eval_report(self, run_id: str, stage: str, result, params):
        if stage not in {'sft_eval', 'rl_eval'} or not isinstance(result, dict):
            return result
        metrics = merge_eval_report(
            result.get('metrics') or {}, (params or {}).get('eval_output_dir'), self.home / 'runs' / run_id)
        if metrics == (result.get('metrics') or {}):
            return result
        updated = dict(result)
        updated['metrics'] = metrics
        if metrics.get('report_html'):
            updated['report_path'] = metrics['report_html']
        return updated

    def eval_report_file(self, run_id: str, stage: str, round_index=None) -> Path:
        """HTML report for this stage, only if it sits inside the run directory."""
        if stage not in {'sft_eval', 'rl_eval'}:
            raise PipelineError('这一阶段没有评测报告', 404)
        run = self._require_run(run_id)
        if round_index is None:
            round_index = run['round_index']
        else:
            round_index = int(round_index)
        step = self.store.latest_active_steps(run_id, round_index).get(stage)
        if not step:
            raise PipelineError('还没有评测报告', 404)
        root = self.home / 'runs' / run_id
        result = _loads(step.get('result_json'))
        params = _loads(step.get('params_json'))
        shown = self._with_eval_report(run_id, stage, result, params) or {}
        path = inside_directory(root, shown.get('report_path') or (shown.get('metrics') or {}).get('report_html'))
        if path is None or path.suffix.lower() not in {'.html', '.htm'}:
            raise PipelineError('还没有评测报告', 404)
        return path

    def _touch_fetch_progress(self) -> None:
        """Update hope rows from local files. Clone progress is in the process log."""
        if model_source() != 'hope':
            return
        for row in self.store.list_fetches():
            if row.get('state') != '同步中' or not row.get('repo_id'):
                continue
            status = self.fetcher.status_of(row['kind'], row['repo_id'])
            if status.state != row.get('state') or status.message != row.get('message'):
                self._note_fetch(status)

    def view(self, run_id: str, round_index=None) -> dict:
        run = self._require_run(run_id)
        self._touch_fetch_progress()
        if round_index is None:
            round_index = run['round_index']
        else:
            round_index = int(round_index)
        rnd = self.store.get_round(run_id, round_index)
        if not rnd:
            raise PipelineError('轮次不存在', 404)
        active = self.store.latest_active_steps(run_id, round_index)
        candidates, recommendation = self.candidates(run, rnd)
        drafts = self.store.list_drafts(run_id, round_index)
        forms = {}
        board = []
        for meta in STAGE_META:
            stage = meta['id']
            state = self._form_state(run, rnd, stage, active, recommendation, drafts.get(stage))
            forms[stage] = state
            metric = None
            metrics = (state.get('result') or {}).get('metrics') or {}
            if 'accuracy' in metrics:
                metric = f"acc {metrics['accuracy']}"
            elif 'train_loss' in metrics:
                metric = f"loss {metrics['train_loss']}"
            elif 'reward' in metrics:
                metric = f"reward {metrics['reward']}"
            elif 'bleu_1' in metrics:
                metric = f"BLEU-1 {metrics['bleu_1']}"
            elif 'rouge_l_f' in metrics:
                metric = f"ROUGE-L {metrics['rouge_l_f']}"
            board.append({'id': stage, 'label': meta['label'], 'summary': meta['summary'], 'status': state['status'],
                          'metric': metric})
        service = self.store.latest_service(run_id)
        serve_fill = self._defaults_for(run, rnd, 'serve')
        saved_serve = _loads(service.get('params_json')) if service and service.get('params_json') else {}
        if service and service.get('status') == 'running' and saved_serve:
            serve_values = saved_serve
        else:
            serve_values = dict(serve_fill)
            remembered = saved_serve.get('more_params')
            if remembered not in (None, ''):
                serve_values['more_params'] = remembered
        for key, value in (drafts.get('serve') or {}).items():
            if key in {'model', 'model_type', 'template', 'adapters'} or isinstance(value, dict):
                continue
            serve_values[key] = value
        for key in ('model', 'model_type', 'template', 'adapters'):
            if key in serve_fill:
                serve_values[key] = serve_fill.get(key) if serve_fill.get(key) is not None else ''
        if service and service.get('api_key'):
            service = dict(service)
            service['api_key_set'] = True
            service['api_key'] = ''
        export_fill = self._export_defaults(run, rnd)
        export_step = active.get('export')
        saved_export = _loads(export_step.get('params_json')) if export_step else {}
        export_values = dict(export_fill)
        for key, value in saved_export.items():
            if key in {'model', 'model_type', 'template', 'adapters', 'checkpoint'}:
                continue
            if value in (None, ''):
                continue
            export_values[key] = value
        for key in ('model', 'model_type', 'template', 'adapters', 'checkpoint'):
            export_values[key] = export_fill.get(key) if export_fill.get(key) is not None else ''
        if 'merge_lora' in saved_export:
            export_values['merge_lora'] = saved_export.get('merge_lora') in {True, 'true', 'True', 1, '1'}
        for key, value in (drafts.get('export') or {}).items():
            if key in {'model', 'model_type', 'template'} or isinstance(value, dict):
                continue
            export_values[key] = value
        for key in ('model', 'model_type', 'template'):
            export_values[key] = export_fill.get(key) if export_fill.get(key) is not None else ''
        export_running = self._exports.running(run_id)
        export_status = 'running' if export_running else (export_step['status'] if export_step else 'idle')
        export_job = {
            'status': export_status,
            'command': (export_step or {}).get('command') or '',
            'output_dir': export_values.get('output_dir') or '',
        }
        steps = []
        for row in self.store.list_steps(run_id):
            item = dict(row)
            item['params'] = _loads(row['params_json'])
            item['result'] = self._with_eval_report(
                run_id, row.get('stage'), _loads(row['result_json']) if row.get('result_json') else None, item['params'])
            steps.append(item)
        return {
            'run': run,
            'round': rnd,
            'rounds': self.store.list_rounds(run_id),
            'mutable': (rnd['round_index'] == run['round_index'] and rnd['status'] == 'active'
                        and run.get('status') == 'active'),
            'board': board,
            'forms': forms,
            'candidates': candidates,
            'recommendation': recommendation,
            'serve_form': serve_values,
            'serve_fill': {
                'model': serve_fill.get('model') or '',
                'model_type': serve_fill.get('model_type') or '',
                'template': serve_fill.get('template') or '',
                'adapters': serve_fill.get('adapters') or '',
                'checkpoint': serve_fill.get('checkpoint') or '',
            },
            'export_form': export_values,
            'export_fill': {
                'model': export_fill.get('model') or '',
                'model_type': export_fill.get('model_type') or '',
                'template': export_fill.get('template') or '',
                'adapters': export_fill.get('adapters') or '',
                'checkpoint': export_fill.get('checkpoint') or '',
                'output_dir': export_fill.get('output_dir') or '',
                'merge_lora': bool(export_fill.get('merge_lora')),
            },
            'export_job': export_job,
            'steps': steps,
            'checkpoints': self.store.list_checkpoints(run_id),
            'service': service,
            'services': self._public_services(run_id),
            'fetches': self.store.list_fetches(),
            'backend': self.backend.name,
            'flow': assemble_flow(run, rnd, active, lambda stage: forms[stage]['status'],
                                  lambda stage: next((item['metric'] for item in board if item['id'] == stage), '')),
        }

    def _public_services(self, run_id: str) -> list:
        rows = []
        for service in self.store.list_services(run_id):
            item = dict(service)
            item['api_key_set'] = bool(item.get('api_key'))
            item['api_key'] = ''
            rows.append(item)
        return rows

    def _assert_can_run(self, run: dict, stage: str):
        if stage not in STAGE_ORDER:
            raise PipelineError('未知阶段')
        if run.get('status') != 'active':
            raise PipelineError('实验已结束')
        rnd = self.store.get_round(run['id'], run['round_index'])
        if not rnd or rnd['status'] != 'active':
            raise PipelineError('当前轮次已结束，不能再改')
        active = self.store.latest_active_steps(run['id'], run['round_index'])
        if not stage_unlocked(stage, active, rnd):
            raise PipelineError(blocked_message(stage, active, rnd))

    def refresh_fetch(self, run_id: str, body: dict) -> dict:
        self._require_run(run_id)
        body = body or {}
        jobs = []
        for kind, raw in (('model', body.get('model') or ''), ('dataset', body.get('dataset') or '')):
            if not raw or not is_repo_id(str(raw)):
                continue
            blocked = dataset_block_message(str(raw), kind=kind)
            if blocked:
                raise PipelineError(blocked)
            jobs.append((kind, str(raw)))
        for kind, raw in jobs:
            status = self.fetcher.refresh(kind, raw, background=True)
            self._note_fetch(status)
        return {'fetches': self.store.list_fetches(), 'view': self.view(run_id)}

    def decide(self, run_id: str, action: str) -> dict:
        run = self._require_run(run_id)
        if run.get('status') != 'active':
            raise PipelineError('实验已结束')
        rnd = self.store.get_round(run_id, run['round_index'])
        if not rnd or rnd['status'] != 'active':
            raise PipelineError('当前轮次已结束，不能再改')
        active = self.store.latest_active_steps(run_id, rnd['round_index'])
        allowed = {item['id'] for item in decision_actions(run, rnd, active)[0]}
        if action not in allowed:
            raise PipelineError('当前不能做这个选择')
        focus = {'rl': 'rl', 'next': 'promote', 'finish': 'finish'}[action]
        with self.store.transaction() as conn:
            if action == 'finish':
                self.store.close_round(conn, run_id, rnd['round_index'], '完成')
                self.store.set_run_status(conn, run_id, 'completed')
            self.store.set_current_stage(conn, run_id, focus)
            step_id = self.store.insert_step(conn, run_id, rnd['round_index'], 'decide', {'action': action}, 'rsi')
            self.store.finish_step(conn, step_id, 'succeeded', {'action': action}, ACTION_LABELS[action] + '\n', '')
        return {'ok': True, 'stage': focus, 'view': self.view(run_id)}

    def preview(self, run_id: str, stage: str, params: dict) -> dict:
        if stage == 'serve':
            run = self._require_run(run_id)
            merged = self._prepare_serve(run, params or {}, strict=False)
            return {'params': merged, 'command': format_shell(build_argv('serve', merged))}
        if stage not in STAGE_ORDER:
            raise PipelineError('未知阶段')
        run = self._require_run(run_id)
        rnd = self.store.get_round(run_id, run['round_index'])
        # Preview must not kick off hope; otherwise an inflight clone has no stage log.
        merged, _side = self._prepare(run, rnd, stage, params or {}, strict=False, fetch=False)
        command = stage_shell(stage, merged) if stage in STAGE_COMMAND else ''
        return {'params': merged, 'command': command}

    def stage_log(self, run_id: str, stage: str, round_index=None, offset=None, limit=None) -> dict:
        """Return a normalized slice of this stage's swift.log.

        Omit ``offset`` to read the last 64KB (what the panel shows on open).
        Pass ``offset`` to read forward from that byte. Carriage returns and ANSI
        are collapsed so a tqdm bar is one line. The path must stay inside this run.
        """
        if stage not in STAGE_ORDER:
            raise PipelineError('未知阶段')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id or ''):
            raise PipelineError('实验不存在', 404)
        run = self._require_run(run_id)
        if round_index is None:
            round_index = run['round_index']
        else:
            round_index = int(round_index)
        if round_index < 1:
            raise PipelineError('轮次不存在', 404)
        rnd = self.store.get_round(run_id, round_index)
        if not rnd:
            raise PipelineError('轮次不存在', 404)
        step = self.store.latest_active_steps(run_id, round_index).get(stage)
        path = self._stage_log_path(run_id, round_index, stage, step)
        window = read_log_window(path, offset, limit)
        if window['size'] == 0 and step and step.get('logs') and offset in (None, 0):
            window = slice_bytes((step['logs'] or '').encode('utf-8'), offset, limit)
        status = step['status'] if step else 'locked'
        return {
            'stage': stage,
            'round': round_index,
            'status': status,
            'done': status not in {'running', 'syncing'},
            'started_at': step.get('started_at') if step else None,
            'ended_at': step.get('ended_at') if step else None,
            **window,
        }

    def _stage_log_path(self, run_id: str, round_index: int, stage: str, step) -> Path | None:
        params = _loads(step['params_json']) if step else {}
        base = self.home / 'runs' / run_id / f'round_{int(round_index)}' / stage
        # While hope is still writing, the panel must read that file. A custom
        # output_dir that already has a swift.log would otherwise replace it.
        if step and step.get('status') == 'syncing':
            canonical = self.stage_log_dir(run_id, round_index, stage) / 'swift.log'
            owned = self._owned_log(run_id, canonical)
            if owned is not None and owned.is_file():
                return owned
        candidates = []
        if params.get('output_dir'):
            candidates.append(Path(str(params['output_dir'])) / 'swift.log')
        if params.get('eval_output_dir'):
            candidates.append(Path(str(params['eval_output_dir'])) / 'swift.log')
        candidates.append(base / 'swift.log')
        candidates.append(base / 'eval_output' / 'swift.log')
        owned = [path for path in (self._owned_log(run_id, item) for item in candidates) if path]
        for path in owned:
            if path.is_file():
                return path
        return owned[0] if owned else None

    def _owned_log(self, run_id: str, candidate: Path | None) -> Path | None:
        if candidate is None:
            return None
        root = (self.home / 'runs' / run_id).resolve()
        try:
            resolved = candidate.resolve()
        except OSError:
            return None
        if resolved.name != 'swift.log':
            return None
        if resolved != root and root not in resolved.parents:
            return None
        return resolved

    def _remember_user_draft(self, run_id: str, round_index: int, stage: str, params: dict) -> None:
        """Keep the raw form, including a run that never finishes, so refresh can refill it."""
        if stage not in set(STAGE_ORDER) | {'serve', 'export'}:
            return
        cleaned = _clean_draft(params or {})
        if not cleaned:
            return
        self.store.upsert_draft(run_id, round_index, stage, cleaned)

    def save_draft(self, run_id: str, stage: str, round_index, params: dict) -> dict:
        run = self._require_run(run_id)
        stage = str(stage or '').strip()
        if stage not in set(STAGE_ORDER) | {'serve', 'export'}:
            raise PipelineError('未知阶段')
        if round_index in (None, ''):
            round_index = run['round_index']
        try:
            round_index = int(round_index)
        except (TypeError, ValueError):
            raise PipelineError('轮次不存在', 404) from None
        if not self.store.get_round(run_id, round_index):
            raise PipelineError('轮次不存在', 404)
        self._remember_user_draft(run_id, round_index, stage, params if isinstance(params, dict) else {})
        stored = self.store.list_drafts(run_id, round_index).get(stage) or {}
        return {'ok': True, 'stage': stage, 'round': round_index, 'params': stored}

    def _sync_verdict(self, repos: list) -> tuple:
        """``ready`` when every listed repo is on disk, ``wait`` while one is still cloning, else ``fail``."""
        if not repos:
            return 'wait', ''
        for item in repos:
            kind = item.get('kind') or 'model'
            repo = item.get('repo_id') or ''
            if not repo:
                return 'wait', ''
            status = self.fetcher.status_of(kind, repo)
            if status.state == '失败':
                return 'fail', status.message or '拉取失败'
            if status.state != '就绪':
                return 'wait', ''
        return 'ready', ''

    def _continue_after_fetch(self) -> None:
        """Start a stage that was waiting on hope once every repo it asked for is local."""
        with self._resume_mu:
            if self._resume_busy:
                self._resume_again = True
                return
            self._resume_busy = True
        try:
            while True:
                self._resume_syncing_steps()
                with self._resume_mu:
                    if not self._resume_again:
                        self._resume_busy = False
                        return
                    self._resume_again = False
        except Exception:
            logger.exception('同步完成后继续阶段失败')
            with self._resume_mu:
                self._resume_busy = False
                self._resume_again = False

    def _resume_syncing_steps(self) -> None:
        for step in self.store.list_steps_with_status('syncing'):
            self._resume_one(step)

    def _resume_one(self, step: dict) -> None:
        run = self.store.get_run(step['run_id'])
        if not run or run.get('status') != 'active' or run.get('round_index') != step.get('round_index'):
            return
        stage = step.get('stage')
        if stage not in SWIFT_STAGES:
            return
        result = _loads(step.get('result_json'))
        repos = list((result.get('fetch') or {}).get('repos') or [])
        verdict, message = self._sync_verdict(repos)
        if verdict == 'wait':
            return
        if verdict == 'fail':
            note = (message or '拉取失败') + '\n'
            with self.store.transaction() as conn:
                current = conn.execute('SELECT status FROM steps WHERE id=?', (step['id'], )).fetchone()
                if current and current['status'] == 'syncing':
                    self.store.finish_step(conn, step['id'], 'failed', {
                        'error': message or '拉取失败',
                        'fetch': result.get('fetch') or {},
                    }, note, '')
            return
        params = result.get('resume_params')
        if not isinstance(params, dict):
            params = {}
        try:
            self.run_stage(step['run_id'], stage, params)
        except PipelineError as exc:
            logger.info('拉取完成后开训失败：%s', exc)
            with self.store.transaction() as conn:
                current = conn.execute('SELECT status FROM steps WHERE id=?', (step['id'], )).fetchone()
                if current and current['status'] == 'syncing':
                    self.store.finish_step(conn, step['id'], 'failed', {'error': str(exc)}, str(exc) + '\n', '')

    def run_stage(self, run_id: str, stage: str, params: dict) -> dict:
        run = self._require_run(run_id)
        self._assert_can_run(run, stage)
        rnd = self.store.get_round(run_id, run['round_index'])
        self._remember_user_draft(run_id, rnd['round_index'], stage, params or {})
        fetch_log = None
        if stage in SWIFT_STAGES:
            fetch_log = self.stage_log_dir(run['id'], rnd['round_index'], stage) / 'swift.log'
            prior = self.store.latest_active_steps(run['id'], rnd['round_index']).get(stage)
            # A click while the previous attempt is still syncing keeps the hope
            # lines already in the file. A fresh run or a re-run after success starts clean.
            if not prior or prior.get('status') != 'syncing':
                reset_stage_log(fetch_log)
            with use_stage_log(fetch_log):
                merged, side = self._prepare(run, rnd, stage, params or {}, strict=True)
        else:
            merged, side = self._prepare(run, rnd, stage, params or {}, strict=True)
        index = STAGE_ORDER.index(stage)
        with self.store.transaction() as conn:
            superseded = list(STAGE_ORDER[index:])
            if stage in {'sft', 'sft_eval', 'rl', 'rl_eval'}:
                superseded.append('decide')
            self.store.supersede(conn, run_id, rnd['round_index'], superseded)
            self.store.deactivate_checkpoints(conn, run_id, rnd['round_index'], CHECKPOINT_RESET.get(stage, []))
            if index <= STAGE_ORDER.index('promote') and stage != 'next_round':
                self.store.clear_promotion(conn, run_id, rnd['round_index'])
            self.store.set_current_stage(conn, run_id, stage)
            step_id = self.store.insert_step(conn, run_id, rnd['round_index'], stage, merged, self.backend.name)

        self._block_hub_model(stage, merged, side)
        if side.get('fetch_state') == '同步中':
            repo = side.get('fetch_repo') or ''
            repos = list(side.get('fetch_repos') or [])
            if not repos and repo:
                repos = [{'kind': 'model', 'repo_id': repo}]
            note = '同步中\n' + repo + '\n'
            if fetch_log is not None and fetch_log.is_file():
                written = fetch_log.read_text(encoding='utf-8', errors='replace')
                if written.strip():
                    note = written
            result = {
                'fetch': {'state': '同步中', 'repo_id': repo, 'repos': repos},
                'resume_params': params or {},
            }
            with self.store.transaction() as conn:
                self.store.finish_step(conn, step_id, 'syncing', result, note, '')
            self._continue_after_fetch()
            latest = self.store.latest_active_steps(run_id, rnd['round_index']).get(stage)
            if latest and latest.get('status') != 'syncing':
                return {
                    'ok': True,
                    'status': latest['status'],
                    'step_id': latest['id'],
                    'command': latest.get('command') or '',
                    'result': _loads(latest.get('result_json')),
                    'view': self.view(run_id),
                }
            return {
                'ok': True,
                'status': 'syncing',
                'step_id': step_id,
                'command': '',
                'result': {'fetch': result['fetch']},
                'view': self.view(run_id),
            }

        if stage in SWIFT_STAGES and self.backend.name != 'mock':
            # Hold the stage log until the training thread opens it for append, so a
            # concurrent re-click cannot truncate hope lines already on disk.
            if fetch_log is not None:
                activate_stage_log(fetch_log)
            thread = threading.Thread(
                target=self._run_swift_stage,
                args=(run_id, rnd['round_index'], stage, step_id, merged, side, fetch_log),
                daemon=True)
            thread.start()
            return {
                'ok': True,
                'status': 'running',
                'step_id': step_id,
                'command': stage_shell(stage, merged),
                'result': None,
                'view': self.view(run_id),
            }

        try:
            outcome = self._execute(run, rnd, stage, merged, side)
        except (PipelineError, BackendError) as exc:
            logs = getattr(exc, 'logs', '') or str(exc)
            with self.store.transaction() as conn:
                self.store.finish_step(conn, step_id, 'failed', {'error': str(exc), 'backend': self.backend.name}, logs,
                                       stage_shell(stage, merged) if stage in STAGE_COMMAND else '')
            raise PipelineError(str(exc)) from exc

        self._commit_success(run_id, rnd['round_index'], stage, step_id, merged, outcome)
        refreshed = self.store.get_run(run_id)
        return {
            'ok': True,
            'status': 'succeeded',
            'step_id': step_id,
            'command': outcome.get('command') or '',
            'result': outcome.get('result'),
            'view': self.view(refreshed['id'], refreshed['round_index'] if stage == 'next_round' else rnd['round_index']),
        }

    def _run_swift_stage(self, run_id, round_index, stage, step_id, params, side, fetch_log=None):
        run = self.store.get_run(run_id)
        rnd = self.store.get_round(run_id, round_index)
        try:
            outcome = self._execute(run, rnd, stage, params, side)
        except (PipelineError, BackendError) as exc:
            logs = getattr(exc, 'logs', '') or str(exc)
            with self.store.transaction() as conn:
                self.store.finish_step(conn, step_id, 'failed', {'error': str(exc), 'backend': self.backend.name}, logs,
                                       stage_shell(stage, params))
            return
        finally:
            if fetch_log is not None:
                deactivate_stage_log(fetch_log)
        self._commit_success(run_id, round_index, stage, step_id, params, outcome)

    def _execute(self, run, rnd, stage, params, side) -> dict:
        if stage in SWIFT_STAGES:
            log_dir = self.stage_log_dir(run['id'], rnd['round_index'], stage)
            context = {
                'round_index': rnd['round_index'],
                'run_id': run['id'],
                'workdir': str(log_dir),
                'eval_rows': side.get('eval_rows'),
                'scenario': run['scenario'],
            }
            produced = self.backend.execute(stage, params, context)
            metrics = merge_eval_report(
                produced.get('metrics') or {}, params.get('eval_output_dir'), self.home / 'runs' / run['id'])
            result = {
                'backend': self.backend.name,
                'mock': produced.get('mock', self.backend.name == 'mock'),
                'metrics': metrics,
                'checkpoint': produced.get('checkpoint'),
            }
            if metrics.get('report_html'):
                result['report_path'] = metrics['report_html']
            if side.get('validation'):
                result['validation'] = side['validation']
                result['dataset'] = params.get('dataset')
            outcome = {
                'command': produced.get('command') or '',
                'logs': produced.get('logs') or '',
                'result': result,
            }
            if produced.get('checkpoint'):
                outcome['checkpoint'] = produced['checkpoint']
                outcome['checkpoint_kind'] = 'sft' if stage == 'sft' else 'rl' if stage == 'rl' else None
                outcome['metrics'] = produced.get('metrics') or {}
            return outcome

        if stage == 'promote':
            picked = side['picked']
            score = picked.get('score')
            score_text = '未评测' if score is None else str(score)
            reason = (params.get('reason') or '').strip()
            logs = (f"[rsi] 晋级选择 {picked['kind']}，分数 {score_text}。\n"
                    f"[rsi] 路径：{picked['path']}\n")
            if reason:
                logs += f'[rsi] 说明：{reason}\n'
            return {
                'command': '',
                'logs': logs,
                'result': {
                    'backend': self.backend.name,
                    'mock': self.backend.name == 'mock',
                    'choice': picked['kind'],
                    'promoted_checkpoint': picked['path'],
                    'score': score,
                    'candidates': side['candidates'],
                },
                'promotion': {
                    'checkpoint': picked['path'],
                    'kind': picked['kind'],
                    'score': score,
                },
            }

        if stage == 'next_round':
            nxt = rnd['round_index'] + 1
            logs = (f"[rsi] 第 {rnd['round_index']} 轮结束。下一轮基座：{rnd['promoted_checkpoint']}\n")
            if params.get('note'):
                logs += f"[rsi] 备注：{params['note']}\n"
            return {
                'command': '',
                'logs': logs,
                'result': {
                    'backend': self.backend.name,
                    'mock': self.backend.name == 'mock',
                    'next_round': nxt,
                    'base_model': rnd['promoted_checkpoint'],
                    'reuse_dataset': bool(params.get('reuse_dataset', True)),
                },
                'next_base': rnd['promoted_checkpoint'],
                'note': params.get('note') or '',
            }
        raise PipelineError('未知阶段')

    def _commit_success(self, run_id, round_index, stage, step_id, params, outcome):
        with self.store.transaction() as conn:
            self.store.finish_step(conn, step_id, 'succeeded', outcome.get('result'), outcome.get('logs'),
                                   outcome.get('command') or '')
            if outcome.get('checkpoint') and outcome.get('checkpoint_kind'):
                self.store.add_checkpoint(conn, run_id, round_index, outcome['checkpoint_kind'], outcome['checkpoint'],
                                          outcome.get('metrics') or {})
            if stage == 'promote' and outcome.get('promotion'):
                promo = outcome['promotion']
                self.store.set_promotion(conn, run_id, round_index, promo['checkpoint'], promo['kind'], promo['score'])
                self.store.set_current_stage(conn, run_id, 'next_round')
            elif stage == 'next_round':
                self.store.close_round_and_open_next(conn, run_id, round_index, outcome.get('note') or '',
                                                     outcome['next_base'])
            else:
                focus = {'sft': 'sft_eval', 'rl': 'rl_eval'}.get(stage, stage)
                self.store.set_current_stage(conn, run_id, focus)

    def _prepare_serve(self, run: dict, user: dict, strict: bool) -> dict:
        rnd = self.store.get_round(run['id'], run['round_index']) or {
            'round_index': run['round_index'],
            'base_model': run['base_model'],
        }
        user = user or {}
        # Keep the typed 「更多参数」 text. apply_more_params merges the flags
        # into the swift command and drops the box, which is not a swift flag.
        typed_more = user.get('more_params') if 'more_params' in user else None
        defaults = self._defaults_for(run, rnd, 'serve')
        merged = _merge(defaults, user)
        self._lock_model_fields(merged, defaults)
        if 'adapters' in user:
            merged['adapters'] = user.get('adapters') or ''
        try:
            merged = apply_more_params(merged)
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc
        self._lock_model_fields(merged, defaults)
        if strict and not merged.get('model') and not merged.get('adapters'):
            raise PipelineError('还没有可部署的模型')
        if not merged.get('served_model_name'):
            merged['served_model_name'] = _slug(run['name'], run['round_index'], 'model')
        try:
            port = int(merged.get('port') or 0)
        except (TypeError, ValueError) as exc:
            raise PipelineError('端口必须是整数') from exc
        if port == 0:
            port = find_free_port()
        merged['port'] = port
        merged['host'] = merged.get('host') or '127.0.0.1'
        side = {}
        self._rewrite_model(merged, run, side, strict)
        self._place_trained_checkpoint(merged, run)
        if side.get('fetch_state') == '同步中' and strict:
            raise PipelineError('同步中')
        if 'more_params' in user:
            text = '' if typed_more is None else str(typed_more)
            if text.strip():
                merged['more_params'] = text
            else:
                merged.pop('more_params', None)
        return merged

    def _serve_backend_name(self) -> str:
        import os
        return os.environ.get('RSI_SERVE_BACKEND', self.backend.name).strip().lower() or 'mock'

    def start_serve(self, run_id: str, params: dict) -> dict:
        run = self._require_run(run_id)
        self._remember_user_draft(run_id, run['round_index'], 'serve', params or {})
        merged = self._prepare_serve(run, params or {}, strict=True)
        host = merged['host']
        port = int(merged['port'])
        previous = self.store.latest_service(run_id)
        if previous and previous['status'] == 'running':
            self.serve.stop(run_id)
            with self.store.transaction() as conn:
                self.store.stop_service_row(conn, previous['id'], (previous.get('logs') or '') + '\n[rsi] 被新的部署替换。\n')
        if not port_is_free(host, port):
            raise PipelineError(f'端口 {port} 已被占用，换一个端口再启动')
        command = format_shell(build_argv('serve', merged))
        trained = str(merged.get('adapters') or merged.get('model') or '')
        backend_name = self._serve_backend_name()
        if backend_name == 'swift':
            self._start_swift_serve(run, merged, command)
            logs = f'推理服务已启动。\n{command}\n'
        else:
            try:
                self.serve.start_mock(run_id, host, port, merged['served_model_name'], merged.get('api_key') or '',
                                      run['scenario'], trained)
            except TimeoutError as exc:
                raise PipelineError(f'推理服务没有在预期时间内起来：{exc}') from exc
            logs = f'推理服务已监听 http://{loopback(host)}:{port}/v1\n{command}\n'
        result = {
            'backend': backend_name,
            'mock': backend_name != 'swift',
            'base_url': f'http://{loopback(host)}:{port}/v1',
            'served_model_name': merged['served_model_name'],
            'checkpoint': trained,
        }
        with self.store.transaction() as conn:
            self.store.insert_service(
                conn, {
                    'run_id': run_id,
                    'checkpoint': trained,
                    'served_model_name': merged['served_model_name'],
                    'host': host,
                    'port': port,
                    'api_key': merged.get('api_key') or '',
                    'backend': backend_name,
                    'params': merged,
                    'logs': logs,
                    'command': command,
                })
            step_id = self.store.insert_step(conn, run_id, run['round_index'], 'serve', merged, backend_name)
            self.store.finish_step(conn, step_id, 'succeeded', result, logs, command)
        return {'ok': True, 'result': result, 'command': command, 'view': self.view(run_id)}

    def _start_swift_serve(self, run, params, command):
        import subprocess
        logical = build_argv('serve', params)
        argv, env = swift_launch(logical)
        try:
            visible = cuda_visible_devices(params.get('gpu_id'))
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc
        if visible is not None:
            if visible == '':
                env.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                env['CUDA_VISIBLE_DEVICES'] = visible
        log_dir = self.stage_dir(run['id'], run['round_index'], 'serve')
        log_path = log_dir / 'deploy.log'
        announce_log(f'[serve {run["id"]}]', log_path)
        log_file = log_path.open('w', encoding='utf-8')
        log_file.write(command + '\n')
        log_file.write('exec: ' + ' '.join(argv) + '\n')
        log_file.flush()
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        self.serve.track(run['id'], process)

    def stop_serve(self, run_id: str) -> dict:
        run = self._require_run(run_id)
        service = self.store.latest_service(run_id)
        if not service or service['status'] != 'running':
            raise PipelineError('没有正在运行的推理服务')
        self.serve.stop(run_id)
        with self.store.transaction() as conn:
            self.store.stop_service_row(conn, service['id'], (service.get('logs') or '') + '\n[rsi] 服务已停止。\n')
            step_id = self.store.insert_step(conn, run_id, run['round_index'], 'serve_stop',
                                             {'service_id': service['id']}, service['backend'])
            self.store.finish_step(conn, step_id, 'succeeded', {'stopped': service['id']}, '[rsi] 推理服务已停止。\n', '')
        return {'ok': True, 'view': self.view(run_id)}

    def serve_log(self, run_id: str) -> dict:
        """Deploy log for the 运行时 panel. The file wins; otherwise the service row."""
        run = self._require_run(run_id)
        service = self.store.latest_service(run_id)
        path = self.home / 'runs' / run_id / f'round_{int(run["round_index"])}' / 'serve' / 'deploy.log'
        text = ''
        owned = self._owned_named_log(run_id, path, 'deploy.log')
        if owned and owned.is_file():
            text = owned.read_bytes()[-65536:].decode('utf-8', errors='replace')
        elif service:
            text = service.get('logs') or ''
        return {
            'log': text,
            'running': bool(service and service.get('status') == 'running'),
            'command': (service or {}).get('command') or '',
        }

    def _owned_named_log(self, run_id: str, candidate: Path, name: str) -> Path | None:
        root = (self.home / 'runs' / run_id).resolve()
        try:
            resolved = candidate.resolve()
        except OSError:
            return None
        if resolved.name != name:
            return None
        if resolved != root and root not in resolved.parents:
            return None
        return resolved

    def _chat_request(self, run_id: str, messages: list, stream: bool, temperature=None, **kwargs):
        self._require_run(run_id)
        service = self.store.latest_service(run_id)
        if not service or service['status'] != 'running':
            raise PipelineError('推理服务未启动')
        if not messages:
            raise PipelineError('messages 不能为空')
        import urllib.request
        body = chat_request_body(service['served_model_name'], messages, temperature=temperature, stream=stream,
                                 **kwargs)
        host = loopback(service['host'])
        request = urllib.request.Request(
            f"http://{host}:{service['port']}/v1/chat/completions",
            data=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )
        if service.get('api_key'):
            request.add_header('Authorization', f"Bearer {service['api_key']}")
        return request

    def chat(self, run_id: str, messages: list, temperature=None, **kwargs) -> dict:
        import urllib.error
        import urllib.request
        request = self._chat_request(run_id, messages, False, temperature, **kwargs)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise PipelineError(f'推理服务返回 {exc.code}: {detail}', exc.code if exc.code in {401, 404} else 502) from exc
        except urllib.error.URLError as exc:
            raise PipelineError(f'无法连接推理服务：{exc.reason}', 502) from exc

    def iter_chat_sse(self, run_id: str, messages: list, temperature=None, **kwargs):
        """Proxy the OpenAI stream. A JSON backend becomes one delta, then [DONE]."""
        import urllib.error
        import urllib.request
        request = self._chat_request(run_id, messages, True, temperature, **kwargs)
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise PipelineError(f'推理服务返回 {exc.code}: {detail}', exc.code if exc.code in {401, 404} else 502) from exc
        except urllib.error.URLError as exc:
            raise PipelineError(f'无法连接推理服务：{exc.reason}', 502) from exc

        def generate():
            try:
                content_type = response.headers.get('Content-Type', '')
                if 'text/event-stream' in content_type:
                    while True:
                        chunk = response.read(2048)
                        if not chunk:
                            break
                        yield chunk
                    return
                payload = json.loads(response.read().decode('utf-8'))
                choice = ((payload.get('choices') or [{}])[0]) or {}
                text = ((choice.get('message') or {}).get('content')) or choice.get('text') or ''
                event = {'choices': [{'delta': {'content': text}}]}
                yield f'data: {json.dumps(event, ensure_ascii=False)}\n\n'.encode()
                yield b'data: [DONE]\n\n'
            finally:
                response.close()

        return generate()

    def _prepare_export(self, run: dict, user: dict, strict: bool) -> dict:
        rnd = self.store.get_round(run['id'], run['round_index']) or {
            'round_index': run['round_index'],
            'base_model': run['base_model'],
        }
        user = user or {}
        typed_more = user.get('more_params') if 'more_params' in user else None
        defaults = self._export_defaults(run, rnd)
        merged = _merge(defaults, user)
        self._lock_model_fields(merged, defaults)
        if 'adapters' in user:
            merged['adapters'] = user.get('adapters') or ''
        if 'merge_lora' in user:
            merged['merge_lora'] = user.get('merge_lora') in {True, 'true', 'True', 1, '1'}
        try:
            merged = apply_more_params(merged)
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc
        self._lock_model_fields(merged, defaults)
        if 'adapters' in user:
            merged['adapters'] = user.get('adapters') or ''
        if 'more_params' in user:
            text = '' if typed_more is None else str(typed_more)
            if text.strip():
                merged['more_params'] = text
            else:
                merged.pop('more_params', None)
        if strict and not str(merged.get('model') or '').strip() and not str(merged.get('adapters') or '').strip():
            raise PipelineError('还没有可导出的模型')
        try:
            cuda_visible_devices(merged.get('gpu_id'))
        except ValueError as exc:
            if strict:
                raise PipelineError(str(exc)) from exc
        return merged

    def preview_export(self, run_id: str, params: dict) -> dict:
        run = self._require_run(run_id)
        merged = self._prepare_export(run, params or {}, strict=False)
        return {'params': merged, 'command': self.preview_export_command(merged)}

    def _read_export_log(self, run_id: str, round_index: int) -> str:
        path = self.home / 'runs' / run_id / f'round_{int(round_index)}' / 'export' / 'export.log'
        owned = self._owned_named_log(run_id, path, 'export.log')
        if owned and owned.is_file():
            return owned.read_bytes()[-65536:].decode('utf-8', errors='replace')
        return ''

    def export_log(self, run_id: str) -> dict:
        run = self._require_run(run_id)
        rnd = self.store.get_round(run_id, run['round_index'])
        step = self.store.latest_active_steps(run_id, run['round_index']).get('export')
        self._settle_export(run_id, step)
        step = self.store.latest_active_steps(run_id, run['round_index']).get('export')
        text = self._read_export_log(run_id, (rnd or run)['round_index'])
        if not text and step:
            text = step.get('logs') or ''
        running = self._exports.running(run_id) or bool(step and step.get('status') == 'running')
        status = 'running' if running else (step['status'] if step else 'idle')
        return {
            'log': text,
            'running': running,
            'status': status,
            'command': (step or {}).get('command') or '',
        }

    def _settle_export(self, run_id: str, step) -> None:
        """Mark a finished process if the waiter has not written the step yet."""
        if not step or step.get('status') != 'running':
            return
        if self._exports.running(run_id):
            return
        text = self._read_export_log(run_id, step['round_index'])
        if not text:
            return
        status = 'failed'
        if '模型导出已完成' in text or text.rstrip().endswith('exit 0'):
            status = 'succeeded'
        with self.store.transaction() as conn:
            self.store.finish_step(conn, step['id'], status, {'output_dir': ''}, text, step.get('command') or '')

    def preview_export_command(self, merged: dict) -> str:
        return format_shell(self._export_argv(merged), env_prefix(merged))

    def start_model_export(self, run_id: str, params: dict) -> dict:
        run = self._require_run(run_id)
        if self._exports.running(run_id):
            raise PipelineError('这个实验已经有一个导出在跑')
        self._remember_user_draft(run_id, run['round_index'], 'export', params or {})
        merged = self._prepare_export(run, params or {}, strict=True)
        command = self.preview_export_command(merged)
        argv = self._export_argv(merged)
        output_dir = str(merged.get('output_dir') or '')
        log_dir = self.stage_dir(run['id'], run['round_index'], 'export')
        log_path = log_dir / 'export.log'
        backend_name = 'mock' if self.backend.name == 'mock' else 'swift'
        with self.store.transaction() as conn:
            step_id = self.store.insert_step(conn, run_id, run['round_index'], 'export', merged, backend_name)
        if backend_name == 'mock':
            log_path.write_text(f'[rsi] 模型导出已完成。\n{command}\n输出目录：{output_dir}\n', encoding='utf-8')
            if output_dir:
                folder = Path(output_dir)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'export_ok.txt').write_text('mock export\n', encoding='utf-8')
            text = log_path.read_text(encoding='utf-8')
            with self.store.transaction() as conn:
                self.store.finish_step(conn, step_id, 'succeeded', {'output_dir': output_dir, 'mock': True}, text,
                                       command)
            return {
                'ok': True,
                'status': 'succeeded',
                'command': command,
                'output_dir': output_dir,
                'log': text,
                'view': self.view(run_id),
            }
        announce_log(f'[export {run_id}]', log_path)
        try:
            launched, env = swift_launch(argv)
            apply_process_env(env, merged)
        except ValueError as exc:
            with self.store.transaction() as conn:
                self.store.finish_step(conn, step_id, 'failed', {}, str(exc), command)
            raise PipelineError(str(exc)) from exc
        if str(merged.get('quant_method') or '') == 'gptq':
            env['OMP_NUM_THREADS'] = '14'
        log_file = log_path.open('w', encoding='utf-8')
        log_file.write(command + '\n')
        log_file.write('exec: ' + ' '.join(launched) + '\n')
        log_file.flush()
        import subprocess
        try:
            process = subprocess.Popen(launched, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        except OSError as exc:
            log_file.write(f'\n[rsi] 无法启动导出：{exc}\n')
            log_file.close()
            text = log_path.read_text(encoding='utf-8', errors='replace')
            with self.store.transaction() as conn:
                self.store.finish_step(conn, step_id, 'failed', {}, text, command)
            raise PipelineError(f'无法启动导出：{exc}') from exc
        self._exports.track(run_id, process)

        def _wait():
            code = process.wait()
            log_file.write(f'\nexit {code}\n')
            log_file.flush()
            log_file.close()
            text = log_path.read_text(encoding='utf-8', errors='replace')
            status = 'succeeded' if code == 0 else 'failed'
            with self.store.transaction() as conn:
                row = conn.execute('SELECT status FROM steps WHERE id=?', (step_id, )).fetchone()
                if row and row['status'] == 'running':
                    self.store.finish_step(conn, step_id, status, {'output_dir': output_dir, 'returncode': code}, text,
                                           command)

        threading.Thread(target=_wait, daemon=True).start()
        return {
            'ok': True,
            'status': 'running',
            'command': command,
            'output_dir': output_dir,
            'log': self._read_export_log(run_id, run['round_index']),
            'view': self.view(run_id),
        }

    def stop_model_export(self, run_id: str) -> dict:
        run = self._require_run(run_id)
        step = self.store.latest_active_steps(run_id, run['round_index']).get('export')
        stopped = self._exports.stop(run_id)
        if not stopped and (not step or step.get('status') != 'running'):
            raise PipelineError('没有正在运行的导出')
        note = '\n[rsi] 导出已停止。\n'
        text = self._read_export_log(run_id, run['round_index']) + note
        if step and step.get('status') == 'running':
            with self.store.transaction() as conn:
                self.store.finish_step(conn, step['id'], 'failed', {'stopped': True}, text, step.get('command') or '')
        return {'ok': True, 'view': self.view(run_id)}
