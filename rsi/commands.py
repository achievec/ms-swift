# Copyright (c) ModelScope Contributors. All rights reserved.
"""Build the swift CLI that a stage would run."""
import json
import shlex
from pathlib import Path

# Passed through as repeated CLI values when they contain whitespace,
# matching `swift sft --dataset a b` / `--reward_funcs accuracy format`.
REPEAT_KEYS = {'dataset', 'val_dataset', 'eval_dataset', 'reward_funcs', 'adapters'}

# Stored on the step, never forwarded to swift.
INTERNAL_KEYS = {
    'source',
    'bundled',
    'extra_yaml',
    'choice',
    'reason',
    'note',
    'reuse_dataset',
    'validation',
    'warnings',
    'custom_eval_config',
    'served_from',
}

STAGE_COMMAND = {
    'sft': 'sft',
    'sft_eval': 'eval',
    'rl': 'rlhf',
    'rl_eval': 'eval',
    'serve': 'deploy',
}


def _repeat_value(key, value):
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item) != '']
    if isinstance(value, str) and key in REPEAT_KEYS and ' ' in value.strip():
        if key in {'dataset', 'val_dataset', 'adapters'} and Path(value).exists():
            return [value]
        return [part for part in value.split() if part]
    return None


def build_argv(stage: str, params: dict) -> list:
    subcommand = STAGE_COMMAND.get(stage)
    if not subcommand:
        return []
    argv = ['swift', subcommand]
    for key, value in params.items():
        if key in INTERNAL_KEYS or key.startswith('_'):
            continue
        if value is None or value == '' or value == 'none':
            continue
        if key == 'quant_method' and value in {'', 'none'}:
            continue
        if key == 'quant_bits' and not params.get('quant_method'):
            continue
        flag = f'--{key}'
        if isinstance(value, bool):
            argv.extend([flag, 'true' if value else 'false'])
            continue
        if isinstance(value, dict):
            argv.extend([flag, json.dumps(value, ensure_ascii=False)])
            continue
        repeated = _repeat_value(key, value)
        if repeated is not None:
            if not repeated:
                continue
            argv.append(flag)
            argv.extend(repeated)
            continue
        argv.extend([flag, str(value)])
    return argv


def format_shell(argv: list, env_prefix: str = 'CUDA_VISIBLE_DEVICES=0') -> str:
    if not argv:
        return ''
    parts = [shlex.quote(token) for token in argv]
    lines = [f'{env_prefix} \\']
    head = parts[0] if len(parts) == 1 else f'{parts[0]} {parts[1]}'
    if len(parts) <= 2:
        lines.append(head)
        return '\n'.join(lines)
    lines.append(head + ' \\')
    index = 2
    while index < len(parts):
        if parts[index].startswith("'--") or parts[index].startswith('--'):
            token = parts[index]
            if index + 1 < len(parts) and not parts[index + 1].startswith('--') and not parts[index + 1].startswith("'--"):
                token = f'{parts[index]} {parts[index + 1]}'
                index += 2
            else:
                index += 1
        else:
            token = parts[index]
            index += 1
        suffix = ' \\' if index < len(parts) else ''
        lines.append(f'    {token}{suffix}')
    return '\n'.join(lines)
