# Copyright (c) ModelScope Contributors. All rights reserved.
"""Training backends. MsSwiftBackend is the default and shells out to this repo's swift CLI."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from rsi.commands import build_argv, format_shell

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def swift_launch(argv: list) -> tuple:
    """Turn a logical `swift <cmd> ...` argv into a process that runs this checkout on GPU 0.

    `RSI_SWIFT_BIN` replaces the launcher (used by tests). Otherwise the command is
    `python -m swift.cli.main`, which is the same entry point as the `swift` script.
    """
    if not argv:
        return [], os.environ.copy()
    override = os.environ.get('RSI_SWIFT_BIN', '').strip()
    if override:
        command = [override, *argv[1:]]
    else:
        command = [sys.executable, '-m', 'swift.cli.main', *argv[1:]]
    env = os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    root = str(REPO_ROOT)
    current = env.get('PYTHONPATH', '')
    parts = [part for part in current.split(':') if part]
    if root not in parts:
        env['PYTHONPATH'] = ':'.join([root, *parts]) if parts else root
    return command, env


def find_checkpoint(output_dir: Path):
    if not output_dir.is_dir():
        return None
    candidates = [path for path in output_dir.glob('checkpoint-*') if path.is_dir()]
    if not candidates:
        return None

    def sort_key(path: Path):
        match = re.search(r'(\d+)$', path.name)
        return int(match.group(1)) if match else -1

    return sorted(candidates, key=sort_key)[-1]


def _mock_metrics(stage: str, round_index: int) -> dict:
    bump = (round_index - 1) * 0.05
    if stage == 'sft':
        loss = round(max(0.22, 1.12 - 0.16 * round_index), 4)
        return {'train_loss': loss, 'eval_loss': round(loss + 0.08, 4), 'token_acc': round(0.55 + bump, 4)}
    if stage == 'sft_eval':
        accuracy = round(0.51 + bump, 4)
        return {'accuracy': accuracy, 'mock_score': accuracy}
    if stage == 'rl':
        return {
            'reward': round(0.48 + bump, 4),
            'kl': 0.03,
            'loss': round(max(0.1, 0.42 - 0.05 * round_index), 4),
        }
    if stage == 'rl_eval':
        accuracy = round(0.58 + bump, 4)
        return {'accuracy': accuracy, 'mock_score': accuracy}
    return {}


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
        command = format_shell(build_argv(stage, params))
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
            lines.append(f"eval_dataset={params.get('eval_dataset')} eval_limit={params.get('eval_limit')} "
                         f'heldout_rows={rows}')
            lines.append(str({'accuracy': metrics.get('accuracy')}))
            if rows is not None:
                metrics = dict(metrics)
                metrics['eval_rows'] = rows
        logs = '\n'.join(lines) + '\n'
        return {
            'ok': True,
            'logs': logs,
            'command': command,
            'metrics': metrics,
            'checkpoint': checkpoint,
            'mock': True,
        }


class MsSwiftBackend:
    """Run the real ms-swift CLI. Blocking; the pipeline calls it off the request thread."""

    name = 'swift'

    def execute(self, stage: str, params: dict, context: dict) -> dict:
        logical = build_argv(stage, params)
        if not logical:
            raise BackendError(f'{stage} 没有对应的 swift 子命令')
        argv, env = swift_launch(logical)
        workdir = Path(context.get('workdir') or params.get('output_dir') or '.')
        workdir.mkdir(parents=True, exist_ok=True)
        command = format_shell(logical)
        log_path = workdir / 'swift.log'
        with log_path.open('w', encoding='utf-8') as log_file:
            log_file.write(command + '\n')
            log_file.write('exec: ' + ' '.join(argv) + '\n')
            log_file.flush()
            completed = subprocess.run(argv, stdout=log_file, stderr=subprocess.STDOUT, env=env, check=False)
        logs = log_path.read_text(encoding='utf-8', errors='replace')
        if completed.returncode != 0:
            raise BackendError(f'swift 退出码 {completed.returncode}', logs=logs)
        metrics = parse_metrics(logs)
        checkpoint = None
        if stage in {'sft', 'rl'}:
            found = find_checkpoint(Path(params['output_dir']))
            if found is None:
                raise BackendError('swift 已结束，但 output_dir 里没有 checkpoint-*', logs=logs)
            checkpoint = str(found)
        if stage in {'sft_eval', 'rl_eval'} and context.get('eval_rows') is not None:
            metrics = dict(metrics)
            metrics['eval_rows'] = context['eval_rows']
        return {
            'ok': True,
            'logs': logs,
            'command': command,
            'metrics': metrics,
            'checkpoint': checkpoint,
            'mock': False,
        }


def build_backend():
    kind = os.environ.get('RSI_BACKEND', 'swift').strip().lower() or 'swift'
    if kind == 'mock':
        return MockBackend()
    if kind == 'swift':
        return MsSwiftBackend()
    raise RuntimeError(f'未知 RSI_BACKEND={kind}，可选 swift 或 mock')
