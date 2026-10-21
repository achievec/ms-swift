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

# Same maps as swift.ui LLMTrain.remove_useless_args. A filled AdaLoRA / VeRA / embedding
# tab must not ride along on a LoRA causal-LM command.
_TUNER_TABS = {
    'lora': ['lora_rank', 'lora_alpha', 'lora_dropout', 'lora_dtype', 'use_rslora', 'use_dora'],
    'llamapro': ['llamapro_num_new_blocks', 'llamapro_num_groups'],
    'lisa': ['lisa_activated_layers', 'lisa_step_interval'],
    'adalora': [
        'adalora_target_r', 'adalora_init_r', 'adalora_tinit', 'adalora_tfinal', 'adalora_deltaT', 'adalora_beta1',
        'adalora_beta2', 'adalora_orth_reg_weight'
    ],
    'lora_ga': [
        'lora_ga_batch_size', 'lora_ga_iters', 'lora_ga_max_length', 'lora_ga_direction', 'lora_ga_scale',
        'lora_ga_stable_gamma'
    ],
    'reft': ['reft_layers', 'reft_rank', 'reft_intervention_type'],
    'vera': ['vera_rank', 'vera_projection_prng_key', 'vera_dropout', 'vera_d_initial'],
    'boft': ['boft_block_size', 'boft_block_num', 'boft_dropout'],
    'fourierft': ['fourier_n_frequency', 'fourier_scaling'],
}
_OPTIMIZER_TABS = {
    'galore': ['use_galore', 'galore_with_embedding', 'galore_rank', 'galore_update_proj_gap'],
    'lorap': ['lorap_lr_ratio'],
    'multimodal': ['vit_lr', 'aligner_lr'],
    'muon': ['use_muon'],
}
_TASK_TABS = {
    'embedding': ['loss_type'],
    'seq_cls': ['num_labels', 'use_chat_template'],
}
_VLLM_TABS = {
    'colocate': [
        'vllm_enable_prefix_caching', 'vllm_gpu_memory_utilization', 'vllm_tensor_parallel_size',
        'vllm_max_model_len', 'sleep_level', 'offload_model', 'offload_optimizer'
    ],
    'server': ['async_generate', 'vllm_server_host', 'vllm_server_port', 'vllm_server_timeout'],
    'llm_rollout': [
        'tensor_parallel_size', 'data_parallel_size', 'max_model_len', 'gpu_memory_utilization', 'port'
    ],
}
# One visible GPU cannot run DeepSpeed, FSDP, or sequence parallel. Passing --deepspeed
# still imports the package, including the zero0 preset.
_MULTI_GPU_ARGS = ('deepspeed', 'sequence_parallel_size', 'fsdp', 'deepspeed_autotp_size', 'zero_hpz_partition_size')

# bitsandbytes / QLoRA knobs only apply when the user picked a load-time quant method.
_BNB_ARGS = (
    'bnb_4bit_compute_dtype',
    'bnb_4bit_quant_type',
    'bnb_4bit_use_double_quant',
    'bnb_4bit_quant_storage',
)


def _repeat_value(key, value):
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item) != '']
    if isinstance(value, str) and key in REPEAT_KEYS and ' ' in value.strip():
        if key in {'dataset', 'val_dataset', 'adapters'} and Path(value).exists():
            return [value]
        return [part for part in value.split() if part]
    return None


def _whole_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number == int(number) else None


def selected_gpu_count(params: dict) -> int:
    """How many CUDA devices this command will see. Missing GPU stays on device 0."""
    gpu = (params or {}).get('gpu_id')
    if gpu is None or gpu == '':
        return 1
    if isinstance(gpu, (list, tuple)):
        devices = [str(item).strip() for item in gpu if str(item).strip() and str(item).strip() != 'cpu']
        return len(devices)
    text = str(gpu).strip()
    if text == 'cpu':
        return 0
    return len([part for part in text.replace(',', ' ').split() if part and part != 'cpu'])


def _drop_inactive_args(params: dict, stage: str) -> None:
    """Drop tuner / optimizer / task args the Web-UI would omit for the selected tab."""
    if stage not in {'sft', 'rl'}:
        return
    relations = {'tuner_type': _TUNER_TABS}
    if stage == 'sft':
        relations['optimizer'] = _OPTIMIZER_TABS
        relations['task_type'] = _TASK_TABS
    else:
        relations['vllm_mode'] = _VLLM_TABS
    for target, tabs in relations.items():
        current = params.get(target)
        if target == 'tuner_type' and not current:
            current = 'lora'
        elif target == 'vllm_mode' and not current:
            current = 'colocate'
        elif target == 'optimizer':
            current = None
            if params.get('use_galore'):
                current = 'galore'
            if params.get('lorap_lr_ratio'):
                current = 'lorap'
            if params.get('vit_lr') or params.get('aligner_lr'):
                current = 'multimodal'
            if params.get('use_muon'):
                current = 'muon'
        for tab, args in tabs.items():
            if tab == 'lora' and current in {'longlora', 'adalora'}:
                continue
            if tab == 'lisa' and current == 'full' and params.get('lisa_activated_layers'):
                continue
            if tab == 'lora_ga' and current == 'lora' and params.get('init_weights') == 'lora-ga':
                continue
            if tab == current:
                continue
            for arg in args:
                params.pop(arg, None)


def _drop_single_gpu_args(params: dict) -> None:
    size = _whole_number(params.get('sequence_parallel_size'))
    if size is not None and size <= 1:
        params.pop('sequence_parallel_size', None)
    if selected_gpu_count(params) > 1:
        return
    for key in _MULTI_GPU_ARGS:
        params.pop(key, None)


def _drop_inactive_quant_args(params: dict) -> None:
    """Omit load-time quant flags unless the user chose a method (avoid forcing bitsandbytes)."""
    method = str(params.get('quant_method') or '').strip()
    if not method or method == 'none':
        params.pop('quant_method', None)
        params.pop('quant_bits', None)
        for key in _BNB_ARGS:
            params.pop(key, None)
        return
    if method != 'bnb':
        for key in _BNB_ARGS:
            params.pop(key, None)


def _command_params(stage: str, params: dict) -> dict:
    cleaned = dict(params or {})
    _drop_inactive_args(cleaned, stage)
    _drop_single_gpu_args(cleaned)
    _drop_inactive_quant_args(cleaned)
    return cleaned


def build_argv(stage: str, params: dict) -> list:
    subcommand = STAGE_COMMAND.get(stage)
    if not subcommand:
        return []
    params = _command_params(stage, params)
    if stage == 'sft' and str(params.get('train_stage') or '').strip() == 'pt':
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
    if selected_gpu_count(params) > 1 and params.get('use_ddp') in {True, 'true', 'True', 1, '1'}:
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
    if selected_gpu_count(params) > 1 and params.get('use_ddp') in {True, 'true', 'True', 1, '1'}:
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
