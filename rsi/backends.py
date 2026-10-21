# Copyright (c) ModelScope Contributors. All rights reserved.
"""Training backends. MsSwiftBackend is the default and shells out to this repo's swift CLI."""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from rsi.commands import apply_process_env, build_argv, stage_shell
from rsi.eval_report import eval_artifacts_usable, merge_eval_report
from rsi.logtail import announce_log, normalize_log, run_and_tee, truncate_log_text
from rsi.model_fetch import apply_runtime_env, configured_train_python

REPO_ROOT = Path(__file__).resolve().parents[1]

# nvidia-smi can take tens–hundreds of ms (or timeout under load). Cache the last
# good result so schema cache keys do not flip on probe jitter.
_GPU_TTL = 30.0
_gpu_cache: dict = {'at': float('-inf'), 'value': None}
_gpu_lock = threading.Lock()


class BackendError(Exception):

    def __init__(self, message: str, logs: str = ''):
        super().__init__(message)
        self.logs = logs or message


def parse_metrics(text: str) -> dict:
    metrics = {}
    for match in re.finditer(
            r'["\'](accuracy|train_loss|eval_loss|reward|kl|token_acc|rouge-l-f)["\']\s*:\s*([0-9.]+)', text):
        key = 'accuracy' if match.group(1) == 'rouge-l-f' else match.group(1)
        metrics[key] = float(match.group(2))
        if match.group(1) == 'rouge-l-f':
            metrics['score_name'] = 'rouge-l-f'
    losses = re.findall(r'["\']loss["\']\s*:\s*([0-9.]+)', text)
    if losses and 'train_loss' not in metrics:
        metrics['train_loss'] = float(losses[-1])
    return metrics


def detect_gpu() -> dict:
    """Report NVIDIA devices without importing torch. Empty when this machine has no GPU."""
    if not shutil.which('nvidia-smi'):
        return {'available': False, 'devices': []}
    try:
        completed = subprocess.run(['nvidia-smi', '-L'], check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return {'available': False, 'devices': []}
    devices = [line.strip() for line in (completed.stdout or '').splitlines() if line.strip().startswith('GPU')]
    return {'available': bool(devices), 'devices': devices}


def clear_gpu_cache() -> None:
    with _gpu_lock:
        _gpu_cache['value'] = None
        _gpu_cache['at'] = float('-inf')


def cached_gpu(ttl: float = _GPU_TTL) -> dict:
    """TTL-cached ``detect_gpu`` with stale-on-empty: keep the last successful device list."""
    now = time.monotonic()
    last = _gpu_cache['value']
    if last is not None and now - _gpu_cache['at'] < ttl:
        return last
    with _gpu_lock:
        now = time.monotonic()
        last = _gpu_cache['value']
        if last is not None and now - _gpu_cache['at'] < ttl:
            return last
        fresh = detect_gpu()
        if not fresh.get('devices') and last and last.get('devices'):
            fresh = last
        _gpu_cache['value'] = fresh
        _gpu_cache['at'] = now
        return fresh


def training_python() -> str:
    """Interpreter for ``python -m swift.cli.main``.

    ``RSI_TRAIN_PYTHON`` wins. Otherwise ``train_python`` in fetch.yaml when that
    file exists and is executable. Otherwise the current interpreter.
    """
    explicit = os.environ.get('RSI_TRAIN_PYTHON', '').strip()
    if explicit:
        return explicit
    return configured_train_python() or sys.executable


def swift_launch(argv: list) -> tuple:
    """Turn a logical `swift <cmd> ...` argv into a process that runs this checkout on GPU 0.

    `RSI_SWIFT_BIN` replaces the launcher (used by tests). Otherwise the command is
    ``<training python> -m swift.cli.main``.
    """
    if not argv:
        return [], os.environ.copy()
    override = os.environ.get('RSI_SWIFT_BIN', '').strip()
    if override:
        command = [override, *argv[1:]]
    else:
        command = [training_python(), '-m', 'swift.cli.main', *argv[1:]]
    env = os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    root = str(REPO_ROOT)
    current = env.get('PYTHONPATH', '')
    parts = [part for part in current.split(':') if part]
    if root not in parts:
        env['PYTHONPATH'] = ':'.join([root, *parts]) if parts else root
    apply_runtime_env(env)
    return command, env


_VERSION_DIR = re.compile(r'^v(\d+)-(\d{8}-\d{6})$')
_LAST_CHECKPOINT = re.compile(r'last_model_checkpoint:\s*(\S+)')


def _checkpoint_step(path: Path) -> int:
    match = re.search(r'(\d+)$', path.name)
    return int(match.group(1)) if match else -1


def _version_sort_key(name: str):
    match = _VERSION_DIR.match(name)
    if match:
        return (1, int(match.group(1)), match.group(2))
    return (0, 0, name)


def _checkpoints_under(output_dir: Path):
    """checkpoint-* in output_dir and in Swift version folders such as v1-20261020-122840."""
    if not output_dir.is_dir():
        return
    for path in output_dir.glob('checkpoint-*'):
        if path.is_dir():
            yield path
    try:
        children = list(output_dir.iterdir())
    except OSError:
        return
    for child in children:
        if not child.is_dir() or not re.match(r'^v\d+-', child.name):
            continue
        for path in child.glob('checkpoint-*'):
            if path.is_dir():
                yield path


def _checkpoint_from_log(logs: str, output_dir: Path):
    found = _LAST_CHECKPOINT.findall(logs or '')
    if not found:
        return None
    raw = found[-1].strip().strip('\'"')
    if raw in {'', 'None'}:
        return None
    path = Path(raw)
    try:
        if path.exists() or path.is_symlink():
            path = path.resolve()
    except OSError:
        return None
    if not path.is_dir():
        return None
    try:
        path.resolve().relative_to(output_dir.resolve())
    except (OSError, ValueError):
        return None
    return path


def find_checkpoint(output_dir: Path, logs: str = ''):
    """Latest checkpoint under output_dir, including ``v<n>-<timestamp>/checkpoint-*``.

    ``last_model_checkpoint`` in the swift log wins when that directory exists
    inside output_dir. Otherwise the highest step in the newest version folder.
    """
    output_dir = Path(output_dir)
    logged = _checkpoint_from_log(logs, output_dir)
    if logged is not None:
        return logged
    candidates = list(_checkpoints_under(output_dir))
    if not candidates:
        return None

    def sort_key(path: Path):
        parent = path.parent
        version = _version_sort_key(parent.name) if parent != output_dir else (0, 0, '')
        return (version[0], version[1], version[2], _checkpoint_step(path))

    return sorted(candidates, key=sort_key)[-1]


def _mock_eval_metrics(accuracy: float) -> dict:
    """Accuracy plus the headline NLG scores the compare table expects."""
    bleu_1 = round(max(0.01, accuracy - 0.18), 4)
    rouge_1_f = round(max(0.01, accuracy - 0.08), 4)
    rouge_l_f = round(max(0.01, accuracy - 0.05), 4)
    return {
        'accuracy': accuracy,
        'mock_score': accuracy,
        'bleu_1': bleu_1,
        'rouge_l_f': rouge_l_f,
        'scores': {
            'bleu-1': bleu_1,
            'rouge-1-f': rouge_1_f,
            'rouge-l-f': rouge_l_f,
        },
    }


def _write_mock_eval_report(eval_output_dir, metrics: dict) -> None:
    """Drop a tiny evalscope-shaped report so mock runs get HTML links + merge_eval_report scores."""
    if not eval_output_dir:
        return
    root = Path(str(eval_output_dir))
    reports = root / 'native' / 'mock' / 'reports'
    dataset = reports / 'mock-model'
    dataset.mkdir(parents=True, exist_ok=True)
    scores = (metrics or {}).get('scores') or {}
    payload = {
        'dataset_name': 'general_qa',
        'model_name': 'mock-model',
        'metrics': [
            {'name': name, 'score': score, 'categories': []}
            for name, score in scores.items()
            if isinstance(score, (int, float))
        ],
    }
    if metrics.get('accuracy') is not None:
        payload['metrics'].insert(0, {
            'name': 'accuracy',
            'score': metrics['accuracy'],
            'categories': [],
        })
    (dataset / 'general_qa.json').write_text(json.dumps(payload), encoding='utf-8')
    (reports / 'report.html').write_text(
        '<html><body><h1>mock eval report</h1></body></html>',
        encoding='utf-8',
    )


def _mock_metrics(stage: str, round_index: int) -> dict:
    bump = (round_index - 1) * 0.05
    if stage == 'sft':
        loss = round(max(0.22, 1.12 - 0.16 * round_index), 4)
        return {'train_loss': loss, 'eval_loss': round(loss + 0.08, 4), 'token_acc': round(0.55 + bump, 4)}
    if stage == 'sft_eval':
        return _mock_eval_metrics(round(0.51 + bump, 4))
    if stage == 'rl':
        return {
            'reward': round(0.48 + bump, 4),
            'kl': 0.03,
            'loss': round(max(0.1, 0.42 - 0.05 * round_index), 4),
        }
    if stage == 'rl_eval':
        return _mock_eval_metrics(round(0.58 + bump, 4))
    return {}


def resolve_log_dir(context: dict, params: dict) -> Path:
    """Absolute directory for this stage's swift.log.

    A missing path, ``.``, or any relative path is refused. Training output must
    not land in the process working directory.
    """
    for raw in (context.get('workdir'), params.get('output_dir'), params.get('eval_output_dir')):
        if raw is None:
            continue
        text = str(raw).strip()
        if text in {'', '.'}:
            continue
        path = Path(text).expanduser()
        if path.is_absolute():
            return path
    raise BackendError('缺少实验目录，训练日志不会写到当前工作目录')


def _append_after_fetch(path: Path) -> bool:
    """True when swift.log already holds hope text that the stage command should follow.

    A previous training-only log starts with ``swift `` / ``exec:`` and has no hope
    section markers; that file is replaced. Hope login/clone lines are kept, and
    training is appended under them.
    """
    try:
        from rsi.model_fetch import stage_log_is_active
        if stage_log_is_active(path):
            return True
    except Exception:
        pass
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        head = path.read_bytes()[:4096]
    except OSError:
        return False
    sample = head.lstrip()
    if not sample:
        return False
    hope_markers = ('[模型]'.encode('utf-8'), '[数据集]'.encode('utf-8'), '登录'.encode('utf-8'), b'hope ')
    if any(marker in head for marker in hope_markers):
        return True
    if sample.startswith(b'swift ') or b'\nexec:' in head:
        return False
    return True


def _write_stage_log(path: Path, logs: str) -> str:
    """Write this run's text. Keep a fetch preamble that is already in the file."""
    if _append_after_fetch(path):
        existing = path.read_bytes()
        prefix = b'' if existing.endswith(b'\n') else b'\n'
        with path.open('ab') as handle:
            handle.write(prefix)
            handle.write(logs.encode('utf-8'))
        return path.read_text(encoding='utf-8', errors='replace')
    path.write_text(logs, encoding='utf-8')
    return logs


def _absolute_output(raw, fallback: Path) -> Path:
    text = str(raw or '').strip()
    if text in {'', '.'}:
        return fallback
    path = Path(text).expanduser()
    if not path.is_absolute():
        return fallback
    return path


def _pin_output_params(stage: str, params: dict, workdir: Path) -> dict:
    pinned = dict(params)
    if stage in {'sft', 'rl'}:
        pinned['output_dir'] = str(_absolute_output(pinned.get('output_dir'), workdir))
    if stage in {'sft_eval', 'rl_eval'}:
        pinned['eval_output_dir'] = str(_absolute_output(pinned.get('eval_output_dir'), workdir))
    return pinned


def _write_mock_checkpoint(output_dir: Path, params: dict, stage: str, round_index: int, metrics: dict) -> Path:
    checkpoint = output_dir / 'checkpoint-final'
    checkpoint.mkdir(parents=True, exist_ok=True)
    args = {
        'model': params.get('model'),
        'template': params.get('template'),
        'tuner_type': params.get('tuner_type', 'lora'),
        'torch_dtype': params.get('torch_dtype'),
        'dataset': params.get('dataset'),
        'system': params.get('system'),
        'rsi_mock': True,
        'rsi_stage': stage,
        'rsi_round': round_index,
        'metrics': metrics,
    }
    (checkpoint / 'args.json').write_text(
        __import__('json').dumps(args, ensure_ascii=False, indent=2), encoding='utf-8')
    (checkpoint / 'rsi_mock.json').write_text(
        __import__('json').dumps({'stage': stage, 'metrics': metrics}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    return checkpoint


class MockBackend:
    """Records the swift command and returns deterministic metrics. No GPU calls."""

    name = 'mock'

    def execute(self, stage: str, params: dict, context: dict) -> dict:
        log_dir = resolve_log_dir(context, params)
        params = _pin_output_params(stage, params, log_dir)
        command = stage_shell(stage, params)
        round_index = int(context.get('round_index') or 1)
        metrics = _mock_metrics(stage, round_index)
        lines = [command] if command else []
        checkpoint = None
        if stage in {'sft', 'rl'}:
            output_dir = Path(params['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            loss_key = 'loss' if stage == 'rl' else 'train_loss'
            start = 1.8 if stage == 'sft' else 0.9
            end = metrics.get(loss_key, metrics.get('loss', 0.4))
            for step, loss in ((1, start), (2, round((start + end) / 2, 4)), (3, end)):
                if stage == 'rl':
                    lines.append(
                        f"{{'loss': {loss}, 'reward': {metrics['reward']}, 'kl': {metrics['kl']}, 'epoch': {step * 0.33:.2f}, 'step': {step}}}"
                    )
                else:
                    lines.append(
                        f"{{'loss': {loss}, 'token_acc': {metrics['token_acc']}, 'learning_rate': {params.get('learning_rate')}, 'epoch': {step * 0.33:.2f}}}"
                    )
            checkpoint_path = _write_mock_checkpoint(output_dir, params, stage, round_index, metrics)
            checkpoint = str(checkpoint_path)
            lines.append(f'检查点：{checkpoint}')
        elif stage in {'sft_eval', 'rl_eval'}:
            rows = context.get('eval_rows')
            # Empty adapters → base-model eval; keep scores below the adapter run so
            # the RSI compare table has a visible gap in mock mode.
            if not str(params.get('adapters') or '').strip():
                metrics = _mock_eval_metrics(round(max(0.01, float(metrics.get('accuracy') or 0.5) - 0.12), 4))
            lines.append(f"eval_dataset={params.get('eval_dataset')} eval_limit={params.get('eval_limit')} "
                         f'heldout_rows={rows} adapters={params.get("adapters") or ""} '
                         f'model={params.get("model") or ""}')
            lines.append(str({'accuracy': metrics.get('accuracy')}))
            if rows is not None:
                metrics = dict(metrics)
                metrics['eval_rows'] = rows
            _write_mock_eval_report(params.get('eval_output_dir'), metrics)
        logs = '\n'.join(lines) + '\n'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'swift.log'
        logs = _write_stage_log(log_path, logs)
        announce_log(f'[{stage} {context.get("run_id") or "-"}]', log_path)
        return {
            'ok': True,
            'logs': logs,
            'command': command,
            'metrics': metrics,
            'checkpoint': checkpoint,
            'log_path': str(log_path),
            'mock': True,
        }


class MsSwiftBackend:
    """Run the real ms-swift CLI. Blocking; the pipeline calls it off the request thread."""

    name = 'swift'

    def execute(self, stage: str, params: dict, context: dict) -> dict:
        workdir = resolve_log_dir(context, params)
        params = _pin_output_params(stage, params, workdir)
        logical = build_argv(stage, params)
        if not logical:
            raise BackendError(f'{stage} 没有对应的 swift 子命令')
        argv, env = swift_launch(logical)
        try:
            apply_process_env(env, params)
        except ValueError as exc:
            raise BackendError(str(exc)) from exc
        workdir.mkdir(parents=True, exist_ok=True)
        command = stage_shell(stage, params)
        log_path = workdir / 'swift.log'
        prefix = f'[{stage} {context.get("run_id") or "-"}]'
        announce_log(prefix, log_path)
        append = _append_after_fetch(log_path)
        needs_nl = False
        if append:
            try:
                with log_path.open('rb') as existing:
                    existing.seek(0, os.SEEK_END)
                    if existing.tell():
                        existing.seek(-1, os.SEEK_END)
                        needs_nl = existing.read(1) != b'\n'
            except OSError:
                needs_nl = False
        with log_path.open('ab' if append else 'wb') as log_file:
            if needs_nl:
                log_file.write(b'\n')
            log_file.write((command + '\nexec: ' + ' '.join(argv) + '\n').encode('utf-8'))
            log_file.flush()
            returncode = run_and_tee(
                argv, env, log_file, prefix, echo=False,
                on_start=context.get('on_proc'), new_session=True)
        # Parse metrics from the full file; only a short tail is stored/returned in
        # the step row so /api/runs/{id} stays small. Live UI tails swift.log itself.
        full = normalize_log(log_path.read_text(encoding='utf-8', errors='replace'))
        logs = truncate_log_text(full)
        metrics = parse_metrics(full)
        soft_exit = False
        if returncode != 0:
            # Evalscope often finishes (metrics + report.html) then the process
            # still exits 1 on uvicorn / leaked-semaphore cleanup. Prefer the
            # on-disk report over a hard failure so dual-eval (基座) can continue.
            if stage in {'sft_eval', 'rl_eval'}:
                disk = eval_artifacts_usable(params.get('eval_output_dir'))
                if disk:
                    soft_exit = True
                    note = (
                        f'\n[rsi] swift 退出码 {returncode}，但 eval_output 下已有评测报告/指标，'
                        f'按成功处理以便继续基座评测。\n')
                    logs = truncate_log_text((full.rstrip() + note).lstrip())
                    metrics = merge_eval_report(metrics, params.get('eval_output_dir'))
                else:
                    raise BackendError(f'swift 退出码 {returncode}', logs=logs)
            else:
                raise BackendError(f'swift 退出码 {returncode}', logs=logs)
        checkpoint = None
        if stage in {'sft', 'rl'}:
            found = find_checkpoint(Path(params['output_dir']), full)
            if found is None:
                raise BackendError('swift 已结束，但 output_dir 里没有 checkpoint-*（也查了 v*-* 版本目录）', logs=logs)
            checkpoint = str(found)
        if stage in {'sft_eval', 'rl_eval'} and context.get('eval_rows') is not None:
            metrics = dict(metrics)
            metrics['eval_rows'] = context['eval_rows']
        result = {
            'ok': True,
            'logs': logs,
            'command': command,
            'metrics': metrics,
            'checkpoint': checkpoint,
            'mock': False,
        }
        if soft_exit:
            result['soft_exit_code'] = returncode
        return result


def build_backend():
    kind = os.environ.get('RSI_BACKEND', 'swift').strip().lower() or 'swift'
    if kind == 'mock':
        return MockBackend()
    if kind == 'swift':
        return MsSwiftBackend()
    raise RuntimeError(f'未知 RSI_BACKEND={kind}，可选 swift 或 mock')
