# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stage machine for one RSI run.

The trunk is SFT → SFT eval. Dataset settings live on SFT. After that eval
the user chooses RL, another round, or an end. RL eval offers the same kind
of choice. A failed eval cannot start RL.
"""
import json
import logging
import os
import re
import shutil
import signal
import threading
import uuid
from pathlib import Path

import yaml

from rsi.backends import BackendError, swift_launch
from rsi.eval_report import (
    base_eval_output_dir,
    compact_eval_metrics,
    inside_directory,
    merge_eval_report,
)
from rsi.commands import (
    QUANT_FORM_KEYS,
    STAGE_COMMAND,
    apply_more_params,
    apply_process_env,
    build_argv,
    env_prefix,
    format_shell,
    grpo_generation_batch_issue,
    is_rl_slider_midpoint,
    stage_shell,
    strip_form_quant,
    strip_rl_slider_midpoints,
)
from rsi.llm_export_spec import CALIB_QUANT_METHODS, EXPORT_QUANT_KEYS, export_quant_available
from rsi.llm_infer_spec import chat_request_body, cuda_visible_devices, template_default_system
from rsi.dataset_archive import (
    archive_max_bytes,
    extract_archive,
    is_archive_filename,
    process_extracted_dataset,
    text_upload_max_bytes,
)
from rsi.dataset_validate import validate_file, validate_public_id
from rsi.flow import ACTION_LABELS, assemble_flow, blocked_message, decision_actions, stage_unlocked
from rsi.logtail import (
    LIVE_TAIL_BYTES,
    announce_log,
    normalize_log,
    read_log_window,
    slice_bytes,
    truncate_log_text,
)
from rsi.metrics import STAGE_PLOTS, collect_stage_metrics
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
}


def _terminate_proc(proc, timeout: float = 5.0) -> None:
    """SIGTERM then SIGKILL. Prefer the process group when ``start_new_session`` was used."""
    if proc is None or proc.poll() is not None:
        return

    def _signal(sig):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.send_signal(sig)
            except (ProcessLookupError, PermissionError, OSError):
                return

    _signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
        return
    except Exception:
        pass
    _signal(signal.SIGKILL)
    try:
        proc.wait(timeout=timeout)
    except Exception:
        pass


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
        _terminate_proc(proc)
        return True


class _StageProcs:
    """In-flight training/eval ``swift`` processes, one per experiment.

    Cancellation is per step. A stop that arrives before ``Popen`` still kills
    the child when it starts, and a later retry of the same experiment is a new step.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._procs = {}
        self._cancel = set()

    def track(self, run_id: str, step_id: int, proc) -> None:
        kill_now = False
        previous = None
        with self._lock:
            if step_id in self._cancel:
                kill_now = True
            else:
                previous = self._procs.get(run_id)
                self._procs[run_id] = (step_id, proc)
        if kill_now:
            _terminate_proc(proc)
            return
        if previous is not None and previous[1] is not proc and previous[1].poll() is None:
            _terminate_proc(previous[1])

    def clear(self, run_id: str, step_id: int) -> None:
        with self._lock:
            self._cancel.discard(step_id)
            current = self._procs.get(run_id)
            if current is not None and current[0] == step_id:
                self._procs.pop(run_id, None)

    def running(self, run_id: str) -> bool:
        with self._lock:
            current = self._procs.get(run_id)
        return current is not None and current[1].poll() is None

    def stop(self, run_id: str, step_id: int) -> bool:
        with self._lock:
            self._cancel.add(step_id)
            current = self._procs.get(run_id)
            proc = None
            if current is not None and current[0] == step_id:
                proc = self._procs.pop(run_id)[1]
        if proc is None or proc.poll() is not None:
            return False
        _terminate_proc(proc)
        return True


class PipelineError(Exception):

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _loads(text, fallback=None):
    if not text:
        return {} if fallback is None else fallback
    return json.loads(text)


def _export_error_snippet(logs: str, limit: int = 8) -> str:
    """Last non-empty log lines for a failed export status card."""
    lines = [line.strip() for line in normalize_log(logs or '').splitlines() if line.strip()]
    if not lines:
        return ''
    return '\n'.join(lines[-limit:])


def _export_step_output(step) -> tuple:
    """Return ``(output_dir, returncode)`` from an export step row."""
    if not step:
        return '', None
    result = _loads(step.get('result_json')) if step.get('result_json') else {}
    params = _loads(step.get('params_json')) if step.get('params_json') else {}
    if not isinstance(result, dict):
        result = {}
    if not isinstance(params, dict):
        params = {}
    output_dir = str(result.get('output_dir') or params.get('output_dir') or '').strip()
    code = result.get('returncode')
    if code is None and step.get('status') == 'succeeded':
        code = 0
    return output_dir, code


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


def _clean_draft(params: dict, stage: str = '') -> dict:
    """One stage snapshot. Locked identity stays on the experiment, not in the draft."""
    cleaned = {}
    for key, value in (params or {}).items():
        if not isinstance(key, str) or not key or key.startswith('_'):
            continue
        if key in {'model', 'model_type', 'template', 'trained_model'}:
            continue
        # SFT no longer exposes quant widgets; drop sticky draft bnb values.
        if stage == 'sft' and key in QUANT_FORM_KEYS:
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
    if stage == 'rl':
        cleaned = strip_rl_slider_midpoints(cleaned)
    if stage == 'export':
        # Merge-only drafts must not keep awq/bits from an HTML select first-option bug.
        method = str(cleaned.get('quant_method') or '').strip()
        if not method or method == 'none':
            for key in EXPORT_QUANT_KEYS:
                cleaned.pop(key, None)
    return cleaned


def _metric(step, name):
    if not step or step.get('status') != 'succeeded' or not step.get('result_json'):
        return None
    metrics = _loads(step['result_json']).get('metrics') or {}
    value = metrics.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _flag(value, default: bool = True) -> bool:
    """Checkbox / YAML / form string → bool. Missing value keeps ``default``."""
    if value is None or value == '':
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


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


def _bundled_eval_match(params: dict) -> tuple[str, int] | None:
    """Held-out file that pairs with a bundled training set, plus its row count."""
    params = params or {}
    name = ''
    if str(params.get('source') or '') == 'bundled':
        name = str(params.get('bundled') or '')
    if not name:
        dataset = str(params.get('dataset') or '')
        file = Path(dataset)
        try:
            file.expanduser().resolve().relative_to(EXAMPLES.resolve())
        except (OSError, ValueError):
            return None
        stem = file.name
        if stem.endswith('.jsonl'):
            name = stem[:-len('.jsonl')]
    if name == 'itops_mix':
        eval_id = 'itops_eval'
    elif name.endswith('_train'):
        eval_id = name[:-len('_train')] + '_eval'
    elif name:
        eval_id = f'{name}_eval'
    else:
        return None
    path = EXAMPLES / f'{eval_id}.jsonl'
    if not path.is_file():
        return None
    with path.open(encoding='utf-8') as handle:
        count = sum(1 for line in handle if line.strip())
    if not count:
        return None
    return str(path.resolve()), count


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
        self._stages = _StageProcs()

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
        runs = self.store.list_runs()
        for run in runs:
            if run.get('list_status') == 'running':
                continue
            # A live training or export process wins over a stale step row.
            if self._stages.running(run['id']) or self._exports.running(run['id']):
                run['list_status'] = 'running'
        return runs

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
        name = filename or 'data.jsonl'
        if is_archive_filename(name):
            limit = archive_max_bytes()
            if len(data) > limit:
                raise PipelineError(f'压缩包超过 {limit // (1024 * 1024)}MB')
            dest_dir = self.home / 'uploads' / uuid.uuid4().hex
            try:
                extract_archive(name, data, dest_dir)
                resolved, media_warnings = process_extracted_dataset(dest_dir)
            except ValueError as exc:
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise PipelineError(str(exc)) from exc
            except Exception:
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise
            report = validate_file(resolved)
            if media_warnings:
                warnings = list(report.get('warnings') or [])
                for item in media_warnings[:20]:
                    warnings.append(item)
                report['warnings'] = warnings
            return {'path': str(resolved), 'filename': name, 'validation': report}

        limit = text_upload_max_bytes()
        if len(data) > limit:
            raise PipelineError(f'文件超过 {limit // (1024 * 1024)}MB')
        suffix = Path(name).suffix.lower()
        if suffix not in {'.jsonl', '.json', '.txt'}:
            suffix = '.jsonl'
        dest_dir = self.home / 'uploads'
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f'{uuid.uuid4().hex}{suffix}'
        dest.write_bytes(data)
        report = validate_file(dest)
        return {'path': str(dest), 'filename': name, 'validation': report}

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
            name = str(params.get('bundled') or 'itops_mix')
            if name.startswith('wb:'):
                from rsi.workbench import resolve_generated
                path = resolve_generated(self.home, name)
                return str(path.resolve()), validate_file(path)
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

    def _apply_eval_choice(self, preset: dict, run: dict, rnd: dict, stage: str) -> None:
        """SFT 评测跟上本轮训练集；RL 评测跟上已经跑过的 SFT 评测。"""
        if stage == 'rl_eval':
            carried = self._succeeded_sft_eval(run, rnd)
            if carried:
                if carried.get('path'):
                    preset['eval_dataset'] = 'itops-heldout'
                    preset['custom_eval_config'] = carried['path']
                elif carried.get('dataset'):
                    preset['eval_dataset'] = carried['dataset']
                    preset['custom_eval_config'] = ''
                if carried.get('limit') not in (None, ''):
                    preset['eval_limit'] = carried['limit']
                return
        paired = _bundled_eval_match(self._sft_params(run['id'], rnd['round_index']))
        if not paired:
            return
        path, count = paired
        preset['eval_dataset'] = 'itops-heldout'
        preset['custom_eval_config'] = path
        preset['eval_limit'] = count

    def _succeeded_sft_eval(self, run: dict, rnd: dict) -> dict | None:
        step = self.store.latest_active_steps(run['id'], rnd['round_index']).get('sft_eval')
        if not step or step['status'] != 'succeeded':
            return None
        params = _loads(step['params_json'])
        dataset = str(params.get('eval_dataset') or '')
        custom = str(params.get('custom_eval_config') or '')
        limit = params.get('eval_limit')
        if custom and Path(custom).is_file() and dataset in {'', 'itops-heldout', 'general_qa'}:
            return {'path': custom, 'limit': limit}
        if dataset and dataset not in {'itops-heldout', 'general_qa'}:
            return {'dataset': dataset, 'limit': limit}
        return None

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
        # Merged「开启下一轮」stores reuse_dataset on promote; older runs used next_round.
        gate = previous.get('promote') or previous.get('next_round')
        reuse = True
        if gate and gate['status'] == 'succeeded':
            reuse = bool(_loads(gate['params_json']).get('reuse_dataset', True))
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
            return strip_form_quant(preset)

        if stage == 'rl':
            self._show_round_model(preset, run, rnd)
            sft_ckpt = self.store.active_checkpoint(run['id'], rnd['round_index'], 'sft')
            self._remember_checkpoint(preset, sft_ckpt['path'] if sft_ckpt else '')
            sft_params = self._sft_params(run['id'], rnd['round_index'])
            if sft_params.get('dataset'):
                preset['dataset'] = sft_params['dataset']
            preset['output_dir'] = str(self.stage_dir(run['id'], rnd['round_index'], 'rl'))
            preset['external_plugins'] = _resolve_existing(preset.get('external_plugins'))
            # Keep empty-meaning quant off the RL form defaults (HTML select first-option bug).
            return strip_form_quant(preset)

        if stage in {'sft_eval', 'rl_eval'}:
            # Web-UI's eval dataset dropdown has no default. This loop's gate is the
            # bundled itops-heldout set, so the preset keeps that name.
            self._show_round_model(preset, run, rnd)
            kind = 'sft' if stage == 'sft_eval' else 'rl'
            checkpoint = self.store.active_checkpoint(run['id'], rnd['round_index'], kind)
            self._remember_checkpoint(preset, checkpoint['path'] if checkpoint else '')
            preset['eval_output_dir'] = str(self.stage_dir(run['id'], rnd['round_index'], stage) / 'eval_output')
            preset['custom_eval_config'] = _resolve_existing(preset.get('custom_eval_config'))
            self._apply_eval_choice(preset, run, rnd, stage)
            if 'eval_base' not in preset:
                preset['eval_base'] = False
            else:
                preset['eval_base'] = _flag(preset.get('eval_base'), False)
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

    def _export_artifacts(self, run_id: str, round_index: int) -> list:
        """Recent finished export steps for this round (path, time, success)."""
        rows = []
        for row in self.store.list_steps(run_id, round_index):
            if row.get('stage') != 'export' or row.get('status') not in {'succeeded', 'failed'}:
                continue
            output_dir, code = _export_step_output(row)
            rows.append({
                'id': row['id'],
                'status': row['status'],
                'output_dir': output_dir,
                'ended_at': row.get('ended_at') or '',
                'returncode': code,
            })
        return rows[-8:]

    def _export_job_payload(self, run_id: str, round_index: int, export_step, export_values: dict,
                            running: bool) -> dict:
        """Runtime status for the export tab: live job plus last finished result."""
        status = 'running' if running else (export_step['status'] if export_step else 'idle')
        output_dir, code = _export_step_output(export_step)
        if not output_dir:
            output_dir = str((export_values or {}).get('output_dir') or '').strip()
        error = ''
        if status == 'failed' and export_step:
            error = _export_error_snippet(export_step.get('logs') or '')
        return {
            'status': status,
            'command': (export_step or {}).get('command') or '',
            'output_dir': output_dir,
            'returncode': code,
            'started_at': (export_step or {}).get('started_at'),
            'ended_at': (export_step or {}).get('ended_at'),
            'error': error,
            'artifacts': self._export_artifacts(run_id, round_index),
        }

    def _known_export_output(self, run_id: str, path: str) -> str:
        """Return a succeeded export output_dir for this run, or ''."""
        want = str(path or '').strip()
        if not want:
            return ''
        for row in self.store.list_steps(run_id):
            if row.get('stage') != 'export' or row.get('status') != 'succeeded':
                continue
            output_dir, _code = _export_step_output(row)
            if output_dir and Path(output_dir).expanduser() == Path(want).expanduser():
                return output_dir
        return ''

    def _resolved_export_deploy(self, run: dict, user: dict) -> str:
        """Allow 推理服务 to deploy a successful export as full ``--model`` (no adapters)."""
        user = user or {}
        flag = user.get('deploy_export') in {True, 'true', 'True', 1, '1'}
        path = str(user.get('model') or '').strip()
        if not flag and not path:
            return ''
        if flag:
            known = self._known_export_output(run['id'], path)
            if known:
                return known
            # CTA without an explicit path: latest successful export this round.
            for row in reversed(self.store.list_steps(run['id'], run['round_index'])):
                if row.get('stage') == 'export' and row.get('status') == 'succeeded':
                    output_dir, _code = _export_step_output(row)
                    if output_dir:
                        return output_dir
            return ''
        if user.get('adapters') not in ('', None):
            return ''
        return self._known_export_output(run['id'], path)

    def _export_defaults(self, run: dict, rnd: dict) -> dict:
        """Locked experiment identity plus the weights this export should start from.

        ``--model`` matches serve / train / eval: LoRA checkpoints take the base
        from ``args.json`` (local path after hope fetch), else the run base, then
        hope/local rewrites a hub id to the ready directory when available.
        """
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
            if adapter:
                base = _adapter_base_model(path, preset.get('model') or '')
                if base:
                    preset['model'] = base
        else:
            preset['output_dir'] = str(self.home / 'runs' / run['id'] / f'round_{rnd["round_index"]}' / 'export')
        # Same local rewrite train/eval use, without starting a new clone for the form.
        self._rewrite_model(preset, run, {}, strict=False, fetch=False)
        return preset

    def _normalize_export_quant(self, merged: dict) -> None:
        """Empty quant = merge-only. Drop orphan bits/samples so they never reach swift."""
        method = str(merged.get('quant_method') or '').strip()
        if not method or method == 'none':
            merged['quant_method'] = ''
            merged['quant_bits'] = ''
            # Keep the Web-UI sample count as a form default; argv strips it when idle.
            if str(merged.get('quant_n_samples') or '') in {'', '256'}:
                merged['quant_n_samples'] = 256
            return
        merged['quant_method'] = method
        bits = merged.get('quant_bits')
        if bits in (None, '', 'none'):
            merged['quant_bits'] = ''
        else:
            merged['quant_bits'] = bits

    def _validate_export_quant(self, merged: dict) -> None:
        """Block missing calib data / packages before merge+quant dies mid-run."""
        method = str(merged.get('quant_method') or '').strip()
        if not method or method == 'none':
            return
        if method != 'fp8' and merged.get('quant_bits') in (None, '', 'none'):
            raise PipelineError('已选择量化方法，请填写量化比特数(--quant_bits)，或不量化时将量化方法留空。')
        if method in CALIB_QUANT_METHODS:
            dataset = merged.get('dataset')
            if isinstance(dataset, list):
                dataset = ' '.join(str(item).strip() for item in dataset if str(item).strip())
            if not str(dataset or '').strip():
                raise PipelineError(
                    'AWQ/GPTQ 量化需要校准数据集。请填写「校准数据集(--dataset)」，'
                    '或将「量化方法」留空，仅合并 LoRA。')
        missing = export_quant_available(method)
        if missing:
            raise PipelineError(missing)

    def _validate_export_output_dir(self, merged: dict) -> None:
        """swift export defaults to exist_ok=False; warn before a long merge/quant run."""
        if str(merged.get('exist_ok') or '').lower() in {'1', 'true', 'yes'}:
            return
        if merged.get('exist_ok') in {True, 1}:
            return
        out = str(merged.get('output_dir') or '').strip()
        if not out:
            return
        path = Path(out).expanduser()
        try:
            exists = path.exists()
        except OSError:
            return
        if exists:
            raise PipelineError(
                f'输出目录已存在：{path}。请改「输出路径(--output_dir)」，'
                '或在更多参数里加 --exist_ok true（不会自动删除已有文件）。')

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
        method = str(params.get('quant_method') or '').strip()
        if not method or method == 'none':
            params.pop('quant_method', None)
            params.pop('quant_bits', None)
            params.pop('quant_n_samples', None)
            params.pop('dataset', None)
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
        # 「base」is this round's starting weights (round.base_model), not always the
        # experiment foundation. Round 1 → hub / full model; later rounds → the
        # LoRA (or full ckpt) promoted into this round. Choice is which single
        # path becomes next round's starting adapters — foundation stays on --model.
        foundation = str((run or {}).get('base_model') or '').strip()
        round_base = str((rnd or {}).get('base_model') or '').strip()
        base_label = '实验底座' if (not round_base or round_base == foundation) else '沿用本轮起点'
        rows = [
            {
                'kind': 'base',
                'label': base_label,
                'path': rnd['base_model'],
                'score': base_score,
            },
            {
                'kind': 'sft',
                'label': '本轮 SFT checkpoint',
                'path': sft['path'] if sft else None,
                'score': _metric(active.get('sft_eval'), 'accuracy'),
            },
            {
                'kind': 'rl',
                'label': '本轮 RL checkpoint',
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
        # SFT form/draft must not reintroduce QLoRA; enable only via extra_yaml / 更多参数.
        incoming = strip_form_quant(user) if stage == 'sft' else dict(user or {})
        if stage == 'rl':
            incoming = strip_rl_slider_midpoints(incoming)
        merged = _merge(defaults, incoming)
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

        if stage == 'rl':
            raw = str(merged.get('dataset') or '')
            resolved = _resolve_existing(raw)
            if isinstance(resolved, str) and resolved and Path(resolved).is_file():
                merged['dataset'] = resolved
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
            if stage == 'rl':
                issue = grpo_generation_batch_issue(merged)
                if issue:
                    raise PipelineError(issue)
            return merged, side

        if stage in {'sft_eval', 'rl_eval'}:
            custom = merged.get('custom_eval_config')
            if custom:
                path = Path(str(custom))
                if not path.is_file():
                    path = Path(str(_resolve_existing(str(custom))))
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
            merged['eval_base'] = _flag(merged.get('eval_base'), False)
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
                if stage == 'rl' and is_rl_slider_midpoint(key, value):
                    continue
                values[key] = value
        if draft:
            for key, value in draft.items():
                if key in {'model', 'model_type', 'template'} or isinstance(value, dict):
                    continue
                if stage == 'sft' and key in QUANT_FORM_KEYS:
                    continue
                if stage == 'rl' and is_rl_slider_midpoint(key, value):
                    continue
                values[key] = value
        if stage == 'sft':
            for key in QUANT_FORM_KEYS:
                values.pop(key, None)
        elif stage == 'rl':
            # Empty-meaning: do not keep a sticky method that was never intentionally set.
            method = str(values.get('quant_method') or '').strip()
            if not method or method == 'none':
                for key in QUANT_FORM_KEYS:
                    values.pop(key, None)
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
        # Live panels poll /stages/{stage}/log (tailed). Never embed swift.log here —
        # a training run can be tens of MB and would bloat every view poll.
        if status in {'syncing', 'running'}:
            logs = ''
        else:
            logs = truncate_log_text(step['logs'] if step else '')
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

    def _with_eval_report(self, run_id: str, stage: str, result, params):
        if stage not in {'sft_eval', 'rl_eval'} or not isinstance(result, dict):
            return result
        root = self.home / 'runs' / run_id
        metrics = merge_eval_report(result.get('metrics') or {}, (params or {}).get('eval_output_dir'), root)
        updated = dict(result)
        changed = metrics != (result.get('metrics') or {})
        if changed:
            updated['metrics'] = metrics
            if metrics.get('report_html'):
                updated['report_path'] = metrics['report_html']
        compare = list(result.get('compare') or [])
        refreshed = []
        for row in compare:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            out_dir = item.get('eval_output_dir')
            if out_dir:
                row_metrics = merge_eval_report(item.get('metrics') or {}, out_dir, root)
                if row_metrics != (item.get('metrics') or {}):
                    item['metrics'] = compact_eval_metrics(row_metrics)
                    changed = True
                if row_metrics.get('report_html'):
                    item['report_path'] = row_metrics['report_html']
                    changed = True
            refreshed.append(item)
        if refreshed != compare:
            updated['compare'] = refreshed
            changed = True
        return updated if changed else result

    def _eval_step_by_id(self, run_id: str, stage: str, round_index: int, step_id: int) -> dict | None:
        for row in self.store.list_steps(run_id, round_index):
            if int(row['id']) == int(step_id) and row.get('stage') == stage:
                return row
        return None

    def eval_report_file(self, run_id: str, stage: str, round_index=None, kind: str | None = None,
                         step_id=None) -> Path:
        """HTML report for this stage, only if it sits inside the run directory.

        ``kind`` selects a compare row (``base`` / ``sft`` / ``rl``). Empty kind
        serves the stage checkpoint report (current SFT or RL eval).
        ``step_id`` selects a historical (possibly superseded) step instead of
        the latest active one — used by the flexible report picker.
        """
        if stage not in {'sft_eval', 'rl_eval'}:
            raise PipelineError('这一阶段没有评测报告', 404)
        run = self._require_run(run_id)
        if round_index is None:
            round_index = run['round_index']
        else:
            round_index = int(round_index)
        if step_id is not None:
            step = self._eval_step_by_id(run_id, stage, round_index, int(step_id))
        else:
            step = self.store.latest_active_steps(run_id, round_index).get(stage)
        if not step:
            raise PipelineError('还没有评测报告', 404)
        root = self.home / 'runs' / run_id
        result = _loads(step.get('result_json'))
        params = _loads(step.get('params_json'))
        shown = self._with_eval_report(run_id, stage, result, params) or {}
        role = (kind or '').strip().lower()
        path = None
        if role in {'base', 'sft', 'rl'}:
            for row in shown.get('compare') or []:
                if isinstance(row, dict) and row.get('role') == role:
                    path = inside_directory(root, row.get('report_path') or (row.get('metrics') or {}).get('report_html'))
                    break
            if path is None and role == ('sft' if stage == 'sft_eval' else 'rl'):
                path = inside_directory(
                    root, shown.get('report_path') or (shown.get('metrics') or {}).get('report_html'))
        else:
            path = inside_directory(root, shown.get('report_path') or (shown.get('metrics') or {}).get('report_html'))
        if path is None or path.suffix.lower() not in {'.html', '.htm'}:
            raise PipelineError('还没有评测报告', 404)
        return path

    _EVAL_STAGE_SHORT = {'sft_eval': 'SFT评测', 'rl_eval': 'RL评测'}
    _EVAL_ROLE_SHORT = {'base': '基座', 'sft': 'SFT', 'rl': 'RL'}

    def _eval_report_label(self, round_index: int, stage: str, role: str, status: str, step_id: int) -> str:
        stage_name = self._EVAL_STAGE_SHORT.get(stage, stage)
        role_name = self._EVAL_ROLE_SHORT.get(role, role or '报告')
        label = f'第{round_index}轮 · {stage_name}/{role_name}'
        if status == 'superseded':
            label = f'{label} · 历史#{step_id}'
        return label

    def _eval_report_entry(self, run_id: str, step: dict, role: str, label: str, metrics: dict,
                           report_path: str | None, eval_output_dir: str = '') -> dict | None:
        """One picker row. Skip empty shells with neither scores nor an HTML report."""
        compact = compact_eval_metrics(metrics or {})
        html = report_path or (metrics or {}).get('report_html') or ''
        if not compact and not html:
            return None
        stage = step['stage']
        step_id = int(step['id'])
        round_index = int(step['round_index'])
        role_key = (role or '').strip().lower() or ('sft' if stage == 'sft_eval' else 'rl')
        query = f'round={round_index}&step={step_id}'
        if role_key:
            query += f'&kind={role_key}'
        entry = {
            'id': f'{step_id}:{role_key}',
            'step_id': step_id,
            'round_index': round_index,
            'stage': stage,
            'role': role_key,
            'label': label,
            'status': step.get('status') or '',
            'metrics': compact,
            'eval_output_dir': eval_output_dir or '',
            'report_url': f'/api/runs/{run_id}/stages/{stage}/report?{query}',
            'has_report': bool(html),
        }
        if html:
            entry['report_path'] = html
        return entry

    def list_eval_reports(self, run_id: str, round_index=None) -> dict:
        """Comparable eval reports for this run (picker UI). Does not bloat ``/api/runs``.

        Walks ``sft_eval`` / ``rl_eval`` steps (including superseded history). Each
        compare row (基座 / SFT / RL) becomes one entry; steps without compare still
        contribute the stage checkpoint. Metrics are compact scores only.
        """
        self._require_run(run_id)
        if round_index is not None:
            round_index = int(round_index)
        reports = []
        for step in self.store.list_steps(run_id, round_index):
            stage = step.get('stage')
            if stage not in {'sft_eval', 'rl_eval'}:
                continue
            if step.get('status') not in {'succeeded', 'superseded', 'failed'}:
                continue
            if not step.get('result_json'):
                continue
            params = _loads(step.get('params_json')) or {}
            shown = self._with_eval_report(
                run_id, stage, _loads(step.get('result_json')), params) or {}
            if not isinstance(shown, dict):
                continue
            compare = [row for row in (shown.get('compare') or []) if isinstance(row, dict)]
            if compare:
                for row in compare:
                    role = str(row.get('role') or '').strip().lower()
                    if not role:
                        continue
                    label = self._eval_report_label(
                        int(step['round_index']), stage, role, step.get('status') or '', int(step['id']))
                    entry = self._eval_report_entry(
                        run_id, step, role, label,
                        row.get('metrics') or {},
                        row.get('report_path'),
                        row.get('eval_output_dir') or '',
                    )
                    if entry:
                        reports.append(entry)
            else:
                role = 'sft' if stage == 'sft_eval' else 'rl'
                label = self._eval_report_label(
                    int(step['round_index']), stage, role, step.get('status') or '', int(step['id']))
                entry = self._eval_report_entry(
                    run_id, step, role, label,
                    shown.get('metrics') or {},
                    shown.get('report_path'),
                    params.get('eval_output_dir') or '',
                )
                if entry:
                    reports.append(entry)
        return {'reports': reports}

    def _want_base_eval(self, run: dict, params: dict) -> bool:
        if not _flag((params or {}).get('eval_base'), False):
            return False
        base = str((run or {}).get('base_model') or '').strip()
        if not base:
            return False
        model = str((params or {}).get('model') or '').strip()
        adapters = str((params or {}).get('adapters') or '').strip()
        # Already evaluating the bare base — a second pass would be identical.
        if not adapters and model == base:
            return False
        return True

    def _base_eval_params(self, run: dict, params: dict) -> dict:
        """Same eval knobs as the stage checkpoint, but ``model=run.base_model`` and no adapters."""
        base = dict(params or {})
        base_model = str((run or {}).get('base_model') or '').strip()
        base['model'] = base_model
        base['adapters'] = ''
        base.pop('trained_model', None)
        out = base_eval_output_dir(base.get('eval_output_dir'))
        if out is not None:
            base['eval_output_dir'] = str(out)
        self._pin_identity(base, run)
        return base

    def _compare_from_prior_sft(self, run_id: str, round_index: int, root: Path) -> dict | None:
        """Reuse this round's SFT eval metrics when building an RL compare table (no re-run)."""
        step = self.store.latest_active_steps(run_id, round_index).get('sft_eval')
        if not step or step.get('status') != 'succeeded':
            return None
        result = self._with_eval_report(run_id, 'sft_eval', _loads(step.get('result_json')), _loads(step.get('params_json')))
        if not isinstance(result, dict):
            return None
        metrics = compact_eval_metrics(result.get('metrics') or {})
        if not metrics:
            return None
        report = result.get('report_path') or (result.get('metrics') or {}).get('report_html')
        row = {
            'role': 'sft',
            'label': 'SFT',
            'metrics': metrics,
            'eval_output_dir': (_loads(step.get('params_json')) or {}).get('eval_output_dir') or '',
        }
        if report and inside_directory(root, report) is not None:
            row['report_path'] = report
        return row

    def _build_eval_compare(self, run, rnd, stage, params, adapter_metrics, base_bundle=None) -> list:
        """Ordered rows for the form UI: 基座 → (SFT) → current stage checkpoint."""
        root = self.home / 'runs' / run['id']
        rows = []
        if base_bundle:
            base_metrics = compact_eval_metrics(base_bundle.get('metrics') or {})
            base_row = {
                'role': 'base',
                'label': '基座',
                'metrics': base_metrics,
                'eval_output_dir': base_bundle.get('eval_output_dir') or '',
                'command': base_bundle.get('command') or '',
            }
            report = base_bundle.get('report_path') or base_metrics.get('report_html')
            if report:
                base_row['report_path'] = report
            rows.append(base_row)
        if stage == 'rl_eval':
            prior = self._compare_from_prior_sft(run['id'], rnd['round_index'], root)
            if prior:
                rows.append(prior)
        current_role = 'sft' if stage == 'sft_eval' else 'rl'
        current_label = 'SFT' if stage == 'sft_eval' else 'RL'
        current = {
            'role': current_role,
            'label': current_label,
            'metrics': compact_eval_metrics(adapter_metrics or {}),
            'eval_output_dir': (params or {}).get('eval_output_dir') or '',
        }
        report = (adapter_metrics or {}).get('report_html')
        if report:
            current['report_path'] = report
        rows.append(current)
        return rows

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
            # Do not sticky-restore quant from a previous (often failed) export step;
            # default form stays merge-only unless the live draft opts in.
            if key in EXPORT_QUANT_KEYS:
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
        # Empty-meaning / sticky-bug quant: force merge-only display.
        method = str(export_values.get('quant_method') or '').strip()
        if not method or method == 'none':
            export_values['quant_method'] = ''
            export_values['quant_bits'] = ''
            export_values['quant_n_samples'] = 256
        export_running = self._exports.running(run_id)
        export_job = self._export_job_payload(
            run_id, rnd['round_index'], export_step, export_values, export_running)
        steps = []
        for row in self.store.list_steps(run_id):
            item = dict(row)
            # Cap audit-trail logs the same way as forms; full text is on disk.
            item['logs'] = truncate_log_text(row.get('logs') or '')
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
        status = run.get('status')
        if status not in {'active', 'completed'}:
            raise PipelineError('实验已结束')
        rnd = self.store.get_round(run_id, run['round_index'])
        if not rnd:
            raise PipelineError('当前轮次已结束，不能再改')
        # After 「完成」, only 「开启下一轮」may reopen this round; other actions stay closed.
        if status == 'completed':
            if action != 'next' or rnd.get('status') != 'closed':
                raise PipelineError('实验已结束')
        elif rnd.get('status') != 'active':
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
            elif action == 'next' and status == 'completed':
                self.store.reopen_round(conn, run_id, rnd['round_index'])
                self.store.set_run_status(conn, run_id, 'active')
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
        payload = {'params': merged, 'command': command}
        if stage in {'sft_eval', 'rl_eval'} and self._want_base_eval(run, merged):
            base_params = self._base_eval_params(run, merged)
            payload['base_params'] = base_params
            payload['base_command'] = stage_shell(stage, base_params)
        return payload

    def stage_log(self, run_id: str, stage: str, round_index=None, offset=None, limit=None) -> dict:
        """Return a normalized slice of this stage's swift.log.

        Omit ``offset`` to read the live tail (``LIVE_TAIL_BYTES``) for the panel.
        Pass ``offset`` to read forward (follow) or to page earlier history.
        Windows that do not start at byte 0 are aligned to the next newline so a
        mid‑tqdm cut does not paint a broken first line. Huge files stay on disk;
        the UI pages with「加载更早」instead of embedding the full log in
        ``/api/runs/{id}``. Carriage returns and ANSI are collapsed so a tqdm bar
        is one line. The path must stay inside this run.
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
        # First paint uses a larger live tail; follow/earlier pass explicit offset/limit.
        live_tail = None if offset is not None else LIVE_TAIL_BYTES
        window = read_log_window(path, offset, limit, tail=live_tail)
        if window['size'] == 0 and step and step.get('logs') and offset in (None, 0):
            window = slice_bytes((step['logs'] or '').encode('utf-8'), offset, limit, tail=live_tail)
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

    def stage_metrics(self, run_id: str, stage: str, round_index=None) -> dict:
        """TensorBoard scalars for SFT/RL charts (Web-UI Runtime.plot equivalent).

        Looks under ``logging_dir``, ``output_dir/runs``, and ``output_dir``. Paths
        must stay inside this experiment. Payload is a tailed summary only.
        """
        if stage not in STAGE_PLOTS:
            raise PipelineError('该阶段没有训练曲线')
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
        params = _loads(step['params_json']) if step else {}
        roots = self._metrics_search_roots(run_id, round_index, stage, params)
        payload = collect_stage_metrics(stage, roots)
        status = step['status'] if step else 'locked'
        return {
            'stage': stage,
            'round': round_index,
            'status': status,
            'done': status not in {'running', 'syncing'},
            'event_file': payload.get('event_file'),
            'series': payload.get('series') or [],
        }

    def _metrics_search_roots(self, run_id: str, round_index: int, stage: str, params: dict) -> list[Path]:
        """Candidate TensorBoard dirs owned by this run."""
        owned: list[Path] = []
        seen: set[Path] = set()

        def add(raw):
            path = self._owned_dir(run_id, Path(str(raw)) if raw else None)
            if path is None or path in seen:
                return
            seen.add(path)
            owned.append(path)

        add(params.get('logging_dir'))
        add(params.get('output_dir'))
        add(self.stage_dir(run_id, round_index, stage))
        return owned

    def _owned_dir(self, run_id: str, candidate: Path | None) -> Path | None:
        if candidate is None:
            return None
        root = (self.home / 'runs' / run_id).resolve()
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            return None
        if resolved != root and root not in resolved.parents:
            return None
        return resolved

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
            base_out = base_eval_output_dir(params.get('eval_output_dir'))
            if base_out is not None:
                candidates.append(base_out / 'swift.log')
        candidates.append(base / 'swift.log')
        candidates.append(base / 'eval_output' / 'swift.log')
        candidates.append(base / 'eval_output' / 'adapter' / 'swift.log')
        candidates.append(base / 'eval_output' / 'base' / 'swift.log')
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
        cleaned = _clean_draft(params or {}, stage)
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
            if index <= STAGE_ORDER.index('promote'):
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
            'view': self.view(refreshed['id'], refreshed['round_index'] if stage == 'promote' else rnd['round_index']),
        }

    def _run_swift_stage(self, run_id, round_index, stage, step_id, params, side, fetch_log=None):
        run = self.store.get_run(run_id)
        rnd = self.store.get_round(run_id, round_index)
        try:
            outcome = self._execute(run, rnd, stage, params, side, step_id=step_id)
        except (PipelineError, BackendError) as exc:
            logs = getattr(exc, 'logs', '') or str(exc)
            with self.store.transaction() as conn:
                row = conn.execute('SELECT status FROM steps WHERE id=?', (step_id, )).fetchone()
                if row and row['status'] == 'running':
                    self.store.finish_step(conn, step_id, 'failed', {'error': str(exc), 'backend': self.backend.name},
                                           logs, stage_shell(stage, params))
            return
        finally:
            self._stages.clear(run_id, step_id)
            if fetch_log is not None:
                deactivate_stage_log(fetch_log)
        self._commit_success(run_id, round_index, stage, step_id, params, outcome)

    def stop_stage(self, run_id: str, stage: str) -> dict:
        """Stop a running SFT / eval / RL stage. The step is marked failed so it can be retried."""
        if stage not in SWIFT_STAGES:
            raise PipelineError('这一阶段不能停止')
        run = self._require_run(run_id)
        step = self.store.latest_active_steps(run_id, run['round_index']).get(stage)
        if not step or step.get('status') != 'running':
            raise PipelineError('没有正在运行的阶段')
        self._stages.stop(run_id, step['id'])
        note = '\n[rsi] 本阶段已停止。\n'
        log_path = self.stage_log_dir(run_id, run['round_index'], stage) / 'swift.log'
        try:
            with log_path.open('a', encoding='utf-8') as handle:
                handle.write(note)
        except OSError:
            pass
        with self.store.transaction() as conn:
            row = conn.execute('SELECT status, command, logs FROM steps WHERE id=?', (step['id'], )).fetchone()
            # The worker may already have stored the signal exit. This stop is the reason.
            if row and row['status'] in {'running', 'failed'}:
                logs = note if row['status'] == 'running' else ((row['logs'] or '') + note)
                self.store.finish_step(conn, step['id'], 'failed', {
                    'stopped': True,
                    'error': '本阶段已停止',
                }, logs, row['command'] or '')
        return {'ok': True, 'view': self.view(run_id)}

    def _execute(self, run, rnd, stage, params, side, step_id=None) -> dict:
        if stage in SWIFT_STAGES:
            log_dir = self.stage_log_dir(run['id'], rnd['round_index'], stage)
            context = {
                'round_index': rnd['round_index'],
                'run_id': run['id'],
                'workdir': str(log_dir),
                'eval_rows': side.get('eval_rows'),
                'scenario': run['scenario'],
            }
            if step_id is not None:
                context['on_proc'] = lambda proc, rid=run['id'], sid=step_id: self._stages.track(rid, sid, proc)
            produced = self.backend.execute(stage, params, context)
            run_root = self.home / 'runs' / run['id']
            metrics = merge_eval_report(
                produced.get('metrics') or {}, params.get('eval_output_dir'), run_root)
            result = {
                'backend': self.backend.name,
                'mock': produced.get('mock', self.backend.name == 'mock'),
                'metrics': metrics,
                'checkpoint': produced.get('checkpoint'),
            }
            if metrics.get('report_html'):
                result['report_path'] = metrics['report_html']
            logs = produced.get('logs') or ''
            command = produced.get('command') or ''
            if stage in {'sft_eval', 'rl_eval'} and self._want_base_eval(run, params):
                base_params = self._base_eval_params(run, params)
                base_dir = Path(base_params['eval_output_dir'])
                base_dir.mkdir(parents=True, exist_ok=True)
                base_context = dict(context)
                base_context['workdir'] = str(base_dir)
                # Base eval reuses the same held-out rows / dataset args as the stage run.
                # Adapter already succeeded (incl. soft-exit-with-report); keep those
                # metrics even if the base pass hard-fails.
                try:
                    base_produced = self.backend.execute(stage, base_params, base_context)
                except BackendError as exc:
                    result['compare'] = self._build_eval_compare(run, rnd, stage, params, metrics)
                    result['base_error'] = str(exc)
                    note = f'\n\n[rsi] —— 基座评测失败（本阶段检查点结果已保留）——\n{exc}\n'
                    logs = (logs.rstrip() + note)
                    base_cmd = stage_shell(stage, base_params)
                    if base_cmd:
                        result['base_command'] = base_cmd
                        if command:
                            command = command.rstrip() + '\n\n# 基座评测（失败）\n' + base_cmd
                else:
                    base_metrics = merge_eval_report(
                        base_produced.get('metrics') or {}, base_params.get('eval_output_dir'), run_root)
                    base_bundle = {
                        'metrics': base_metrics,
                        'eval_output_dir': base_params.get('eval_output_dir') or '',
                        'command': base_produced.get('command') or '',
                        'report_path': base_metrics.get('report_html') or '',
                    }
                    result['compare'] = self._build_eval_compare(
                        run, rnd, stage, params, metrics, base_bundle=base_bundle)
                    result['base_command'] = base_bundle['command']
                    base_logs = base_produced.get('logs') or ''
                    if base_logs:
                        logs = (logs.rstrip() + '\n\n[rsi] —— 基座评测 ——\n' + base_logs.lstrip())
                    if command and base_bundle['command']:
                        command = command.rstrip() + '\n\n# 基座评测\n' + base_bundle['command']
            elif stage in {'sft_eval', 'rl_eval'}:
                # Still surface a one-column compare when dual-eval is off / skipped.
                result['compare'] = self._build_eval_compare(run, rnd, stage, params, metrics)
            if side.get('validation'):
                result['validation'] = side['validation']
                result['dataset'] = params.get('dataset')
            outcome = {
                'command': command,
                'logs': logs,
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
            nxt = rnd['round_index'] + 1
            logs = (f"[rsi] 晋级选择 {picked['kind']}，分数 {score_text}。\n"
                    f"[rsi] 路径：{picked['path']}\n"
                    f"[rsi] 第 {rnd['round_index']} 轮结束。下一轮起点：{picked['path']}\n")
            if reason:
                logs += f'[rsi] 说明：{reason}\n'
            if params.get('note'):
                logs += f"[rsi] 备注：{params['note']}\n"
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
                    'next_round': nxt,
                    'base_model': picked['path'],
                    'reuse_dataset': bool(params.get('reuse_dataset', True)),
                },
                'promotion': {
                    'checkpoint': picked['path'],
                    'kind': picked['kind'],
                    'score': score,
                },
                'next_base': picked['path'],
                'note': params.get('note') or '',
            }
        raise PipelineError('未知阶段')

    def _commit_success(self, run_id, round_index, stage, step_id, params, outcome):
        with self.store.transaction() as conn:
            row = conn.execute('SELECT status FROM steps WHERE id=?', (step_id, )).fetchone()
            # Stop marks the step failed while the worker is still unwinding.
            if not row or row['status'] != 'running':
                return
            self.store.finish_step(conn, step_id, 'succeeded', outcome.get('result'), outcome.get('logs'),
                                   outcome.get('command') or '')
            if outcome.get('checkpoint') and outcome.get('checkpoint_kind'):
                self.store.add_checkpoint(conn, run_id, round_index, outcome['checkpoint_kind'], outcome['checkpoint'],
                                          outcome.get('metrics') or {})
            if stage == 'promote' and outcome.get('promotion'):
                promo = outcome['promotion']
                self.store.set_promotion(conn, run_id, round_index, promo['checkpoint'], promo['kind'], promo['score'])
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
        export_model = self._resolved_export_deploy(run, user)
        defaults = self._defaults_for(run, rnd, 'serve')
        merged = _merge(defaults, user)
        self._lock_model_fields(merged, defaults)
        if 'adapters' in user:
            merged['adapters'] = user.get('adapters') or ''
        if export_model:
            # Merged/quantized export is a full model: deploy --model only.
            merged['model'] = export_model
            merged['adapters'] = ''
            merged.pop('trained_model', None)
        try:
            merged = apply_more_params(merged)
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc
        self._lock_model_fields(merged, defaults)
        if export_model:
            merged['model'] = export_model
            merged['adapters'] = ''
            merged.pop('trained_model', None)
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
        if export_model:
            merged['model'] = export_model
            merged['adapters'] = ''
        else:
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
        # Match serve: hub id -> local path, LoRA base from checkpoint args.json.
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
        if isinstance(merged.get('dataset'), list):
            merged['dataset'] = ' '.join(
                str(item).strip() for item in merged['dataset'] if str(item).strip())
        self._normalize_export_quant(merged)
        if strict and not str(merged.get('model') or '').strip() and not str(merged.get('adapters') or '').strip():
            raise PipelineError('还没有可导出的模型')
        # Always validate (preview and run) so AWQ without dataset / autoawq
        # never merges then dies with an English ModuleNotFoundError.
        self._validate_export_quant(merged)
        if strict:
            self._validate_export_output_dir(merged)
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
        round_index = (rnd or run)['round_index']
        step = self.store.latest_active_steps(run_id, round_index).get('export')
        self._settle_export(run_id, step)
        step = self.store.latest_active_steps(run_id, round_index).get('export')
        text = self._read_export_log(run_id, round_index)
        if not text and step:
            text = step.get('logs') or ''
        running = self._exports.running(run_id) or bool(step and step.get('status') == 'running')
        params = _loads(step.get('params_json')) if step else {}
        job = self._export_job_payload(run_id, round_index, step, params or {}, running)
        return {
            'log': text,
            'running': running,
            'status': job['status'],
            'command': job['command'],
            'output_dir': job['output_dir'],
            'returncode': job['returncode'],
            'ended_at': job['ended_at'],
            'error': job['error'],
            'artifacts': job['artifacts'],
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
        output_dir, _code = _export_step_output(step)
        code = 0 if status == 'succeeded' else 1
        match = re.search(r'(?m)^exit (-?\d+)\s*$', text)
        if match:
            try:
                code = int(match.group(1))
            except ValueError:
                pass
            status = 'succeeded' if code == 0 else 'failed'
        with self.store.transaction() as conn:
            self.store.finish_step(
                conn, step['id'], status, {'output_dir': output_dir, 'returncode': code}, text,
                step.get('command') or '')

    def preview_export_command(self, merged: dict) -> str:
        return format_shell(self._export_argv(merged), env_prefix(merged))

    def start_model_export(self, run_id: str, params: dict) -> dict:
        run = self._require_run(run_id)
        if self._exports.running(run_id):
            raise PipelineError('这个实验已经有一个导出在跑')
        # Validate first so a blocked AWQ attempt does not sticky-save quant into the draft.
        merged = self._prepare_export(run, params or {}, strict=True)
        self._remember_user_draft(run_id, run['round_index'], 'export', params or {})
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
