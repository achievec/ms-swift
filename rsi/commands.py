# Copyright (c) ModelScope Contributors. All rights reserved.
"""Build the swift CLI that a stage would run."""
import json
import shlex
from pathlib import Path

# Passed through as repeated CLI values when they contain whitespace,
# matching `swift sft --dataset a b` / `--reward_funcs accuracy format`.
REPEAT_KEYS = {
    'dataset',
    'val_dataset',
    'eval_dataset',
    'reward_funcs',
    'reward_model',
    'reward_model_type',
    'adapters',
    'report_to',
    'target_modules',
    'model_name',
    'model_author',
}

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
    'passed',
    'pass_threshold',
    'more_params',
    'checkpoint',
    'trained_model',
    'gpu_id',
    'infer_model_type',
    'train_stage',
    'use_ddp',
    'ddp_num',
    'envs',
    'dry_run',
}

STAGE_COMMAND = {
    'sft': 'sft',
    'sft_eval': 'eval',
    'rl': 'rlhf',
    'rl_eval': 'eval',
    'serve': 'deploy',
    'export': 'export',
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
    if stage == 'sft' and str((params or {}).get('train_stage') or '').strip() == 'pt':
        subcommand = 'pt'
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


def apply_more_params(params: dict) -> dict:
    """Merge the Web-UI 「更多参数」 box into deploy params.

    JSON object keys are copied. A ``--key value`` string is parsed the same way.
    ``more_params`` itself is not a swift flag.
    """
    merged = dict(params or {})
    raw = merged.pop('more_params', None)
    if raw is None or str(raw).strip() == '':
        return merged
    text = str(raw).strip()
    if text[0] in '{[':
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'更多参数 JSON 无法解析：{exc}') from exc
        if not isinstance(loaded, dict):
            raise ValueError('更多参数的 JSON 必须是对象')
        for key, value in loaded.items():
            if value is not None:
                merged[str(key).replace('-', '_')] = value
        return merged
    tokens = shlex.split(text)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith('--') or token == '--':
            raise ValueError('更多参数请用 JSON，或 --key value')
        key = token[2:].replace('-', '_')
        if not key:
            raise ValueError('更多参数请用 JSON，或 --key value')
        if index + 1 < len(tokens) and not tokens[index + 1].startswith('--'):
            merged[key] = tokens[index + 1]
            index += 2
        else:
            merged[key] = True
            index += 1
    return merged


def env_prefix(params: dict) -> str:
    """Shell prefix matching the Web-UI GPU / DDP / env boxes. Missing GPU stays on device 0."""
    from rsi.llm_infer_spec import cuda_visible_devices

    params = params or {}
    parts = []
    gpu = params.get('gpu_id')
    if gpu is None or gpu == '':
        parts.append('CUDA_VISIBLE_DEVICES=0')
    else:
        try:
            visible = cuda_visible_devices(gpu)
        except ValueError:
            visible = None
        if visible:
            parts.append(f'CUDA_VISIBLE_DEVICES={visible}')
        elif visible is None:
            parts.append('CUDA_VISIBLE_DEVICES=0')
    use_ddp = params.get('use_ddp')
    if use_ddp in {True, 'true', 'True', 1, '1'}:
        parts.append(f"NPROC_PER_NODE={params.get('ddp_num') or 1}")
    extra = str(params.get('envs') or '').strip()
    if extra:
        parts.append(extra)
    return ' '.join(parts)


def apply_process_env(env: dict, params: dict) -> dict:
    """Copy GPU, DDP, and ``envs`` onto the process environment."""
    from rsi.llm_infer_spec import cuda_visible_devices

    params = params or {}
    gpu = params.get('gpu_id')
    if gpu not in (None, ''):
        visible = cuda_visible_devices(gpu)
        if visible == '':
            env.pop('CUDA_VISIBLE_DEVICES', None)
        elif visible:
            env['CUDA_VISIBLE_DEVICES'] = visible
    if params.get('use_ddp') in {True, 'true', 'True', 1, '1'}:
        env['NPROC_PER_NODE'] = str(params.get('ddp_num') or 1)
    for token in str(params.get('envs') or '').split():
        if '=' not in token:
            continue
        key, value = token.split('=', 1)
        if key:
            env[key] = value
    return env


def stage_shell(stage: str, params: dict) -> str:
    return format_shell(build_argv(stage, params), env_prefix(params))


def format_shell(argv: list, env_prefix: str = 'CUDA_VISIBLE_DEVICES=0') -> str:
    if not argv:
        return ''
    parts = [shlex.quote(token) for token in argv]
    lines = [f'{env_prefix} \\'] if env_prefix else []
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
