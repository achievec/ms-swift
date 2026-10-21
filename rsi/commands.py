# Copyright (c) ModelScope Contributors. All rights reserved.
"""Build the swift CLI that a stage would run."""
import json
import math
import shlex
from functools import lru_cache
from pathlib import Path

# Passed through as repeated CLI values when they contain whitespace,
# matching `swift sft --dataset a b` / `--reward_funcs accuracy format` /
# `--reward_weights 1.0 0.5` (nargs list of floats — never one quoted token).
REPEAT_KEYS = {
    'dataset',
    'val_dataset',
    'eval_dataset',
    'reward_funcs',
    'reward_weights',
    'reward_model',
    'reward_model_type',
    'loss_weights',
    'interleave_prob',
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
    # RSI-only: run a second eval on the experiment base model (no adapters).
    'eval_base',
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

# Form / draft keys for load-time quant. RSI SFT/RL no longer expose these widgets;
# sticky drafts used to keep quant_method=bnb from an HTML <select> first-option bug.
QUANT_FORM_KEYS = frozenset(('quant_method', 'quant_bits', *_BNB_ARGS))


def strip_form_quant(params: dict) -> dict:
    """Drop load-time quant fields from a form/draft snapshot (extra_yaml may re-add)."""
    cleaned = dict(params or {})
    for key in QUANT_FORM_KEYS:
        cleaned.pop(key, None)
    return cleaned


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


def _same_number(value, expected) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isclose(number, float(expected), rel_tol=0.0, abs_tol=1e-9)


def slider_midpoint(minimum, maximum, step=None):
    """Value an empty ``<input type="range">`` submits after step snapping.

    HTML's default is ``min + (max - min) / 2``. The browser then rounds to the
    nearest step (ties toward +infinity). ``min=1, max=256, step=2`` → 129;
    ``min=0, max=20, step=0.5`` → 10.
    """
    low = float(minimum)
    high = float(maximum)
    raw = low + (high - low) / 2.0
    if step is None:
        return int(raw) if raw == int(raw) else raw
    stride = float(step)
    if stride == 0:
        return int(raw) if raw == int(raw) else raw
    steps_from_base = (raw - low) / stride
    lower_n = math.floor(steps_from_base)
    upper_n = math.ceil(steps_from_base)
    candidates = []
    for count in (lower_n, upper_n):
        snapped = low + count * stride
        if low - 1e-9 <= snapped <= high + 1e-9:
            candidates.append(snapped)
    if not candidates:
        chosen = raw
    else:
        # Nearest; a tie goes toward +infinity (the larger candidate).
        chosen = min(candidates, key=lambda item: (abs(item - raw), -item))
    if math.isclose(chosen, round(chosen), rel_tol=0.0, abs_tol=1e-9):
        return int(round(chosen))
    return chosen


@lru_cache(maxsize=1)
def rl_empty_slider_midpoints() -> dict:
    """RL Gradio sliders that have no explicit value, mapped to their empty-range midpoint.

    A filled default (GRPO's gradient accumulation starts at 1) is not an empty
    slider, so a user-chosen value there is left alone.
    """
    from rsi.webui_stage_spec import stage_layout

    layout = stage_layout('rl', 1)
    found = {}
    for item in (layout.get('by_id') or {}).values():
        if item.get('widget') != 'Slider':
            continue
        if item.get('default') is not None:
            continue
        minimum = item.get('minimum')
        maximum = item.get('maximum')
        if minimum is None or maximum is None:
            continue
        found[item['id']] = slider_midpoint(minimum, maximum, item.get('step'))
    return found


def is_rl_slider_midpoint(key, value) -> bool:
    midpoint = rl_empty_slider_midpoints().get(key)
    if midpoint is None:
        return False
    return _same_number(value, midpoint)


def strip_rl_slider_midpoints(params: dict) -> dict:
    """Drop empty-slider midpoints so the RL preset (eval batch 1, NEFTune 0) can apply."""
    cleaned = dict(params or {})
    for key in list(cleaned):
        if is_rl_slider_midpoint(key, cleaned[key]):
            cleaned.pop(key, None)
    return cleaned


def grpo_generation_batch_issue(params: dict) -> str | None:
    """Chinese error when GRPO's generation batch is not divisible by ``num_generations``.

    Mirrors ``RolloutTrainerArgumentsMixin._init_generation_batch_params`` for the
    default path (neither ``generation_batch_size`` nor ``steps_per_generation`` set):

    ``generation_batch_size = per_device_train_batch_size * world_size * gradient_accumulation_steps``.

    Swift's "Valid values" are the divisors of that batch that ``num_generations``
    may take — not legal values for the batch itself. This does not rewrite
    ``gradient_accumulation_steps``: bumping it grows the rollout batch.
    """
    params = params or {}
    rlhf = str(params.get('rlhf_type') or 'grpo').strip() or 'grpo'
    if rlhf not in {'grpo', 'gkd'}:
        return None
    if params.get('num_generations') in (None, ''):
        return None
    num_generations = _whole_number(params.get('num_generations'))
    if num_generations is None or num_generations <= 1:
        return None
    per_device = _whole_number(params.get('per_device_train_batch_size'))
    if per_device is None:
        per_device = 1
    world_size = selected_gpu_count(params)
    if world_size <= 0:
        world_size = 1
    global_batch = per_device * world_size
    gas = _whole_number(params.get('gradient_accumulation_steps'))
    if gas is None:
        gas = 1
    generation_batch = _whole_number(params.get('generation_batch_size'))
    steps = _whole_number(params.get('steps_per_generation'))
    if generation_batch is None and steps is None:
        steps = gas
        generation_batch = global_batch * steps
    elif generation_batch is not None and steps is None:
        if global_batch <= 0 or generation_batch % global_batch != 0:
            return (f'generation_batch_size（{generation_batch}）必须能被'
                    f'每卡训练 batch（{per_device}）× GPU 数（{world_size}）整除。')
        steps = generation_batch // global_batch
    elif generation_batch is None and steps is not None:
        generation_batch = global_batch * steps
    else:
        expected = global_batch * steps
        if generation_batch != expected:
            return (f'generation_batch_size（{generation_batch}）必须等于'
                    f'每卡训练 batch × GPU 数 × steps_per_generation（{expected}）。')
    if steps is None or steps <= 0 or generation_batch is None:
        return f'steps_per_generation 必须大于 0，当前是 {steps}。'
    if generation_batch % num_generations == 0:
        return None
    valid = [n for n in range(2, generation_batch + 1) if generation_batch % n == 0]
    suggestion = _next_accumulation(global_batch, gas, num_generations)
    hint = f'请把梯度累积调成 {num_generations} 的倍数（例如 {suggestion}），或点「恢复预设」。'
    return (f'GRPO 的 generation_batch_size（{generation_batch}）必须能被 num_generations（{num_generations}）整除。'
            f'当前为每卡训练 batch（{per_device}）× GPU 数（{world_size}）× 梯度累积（{gas}）= {generation_batch}。'
            f'在这个 batch 下，num_generations 只能取：{valid}。{hint}')


def _next_accumulation(global_batch: int, gas: int, num_generations: int) -> int:
    """Smallest accumulation >= the next multiple of ``num_generations`` that divides."""
    start = max(gas, 1)
    multiple = ((start + num_generations - 1) // num_generations) * num_generations
    if multiple == start:
        multiple += num_generations
    if global_batch <= 0:
        return multiple
    candidate = multiple
    # global_batch may share factors with num_generations; walk a few multiples.
    for _ in range(num_generations + 1):
        if (global_batch * candidate) % num_generations == 0:
            return candidate
        candidate += num_generations
    return multiple


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
    if stage == 'rl':
        _drop_lora_ref_model(params)


def _drop_lora_ref_model(params: dict) -> None:
    """LoRA / adapter GRPO must not pass --ref_model (ms-swift raises); sticky UI junk used to."""
    tuner = str(params.get('tuner_type') or 'lora').strip() or 'lora'
    if tuner == 'full':
        return
    params.pop('ref_model', None)
    params.pop('ref_model_type', None)
    params.pop('ref_model_revision', None)


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
    if stage == 'rl':
        cleaned = strip_rl_slider_midpoints(cleaned)
    _drop_inactive_args(cleaned, stage)
    _drop_single_gpu_args(cleaned)
    _drop_inactive_quant_args(cleaned)
    return cleaned


def build_argv(stage: str, params: dict) -> list:
    subcommand = STAGE_COMMAND.get(stage)
    if not subcommand:
        return []
    params = _command_params(stage, params)
    if stage == 'rl':
        issue = grpo_generation_batch_issue(params)
        if issue:
            raise ValueError(issue)
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
