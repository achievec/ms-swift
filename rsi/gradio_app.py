# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI closed loop as a Gradio tab, using the same controls as the SWIFT Web-UI."""
import os
from pathlib import Path

import gradio as gr

from rsi.pipeline import Pipeline, PipelineError
from rsi.schemas import STAGE_META, presets, public_schema
from rsi.webui_catalog import load_catalog, match_model
from rsi.webui_labels import control_label

OTHER_TABS = ['LLM预训练/微调', 'LLM人类对齐', 'LLM GRPO', 'LLM推理', 'LLM导出', 'LLM评测', 'LLM采样']
DEFAULT_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
COMMAND_STAGES = {'sft', 'sft_eval', 'rl', 'rl_eval', 'serve'}
STATUS_ZH = {
    'succeeded': '已完成',
    'failed': '失败',
    'running': '运行中',
    'locked': '未解锁',
    'ready': '可运行',
}
RUN_LABEL = {
    'data': '🚀 运行本阶段',
    'sft': '🚀 开始训练',
    'sft_eval': '开始评测',
    'rl': '🚀 开始训练',
    'rl_eval': '开始评测',
    'promote': '确认晋级',
    'next_round': '进入下一轮',
}
SLIDERS = {
    'per_device_train_batch_size': (1, 256, 1),
    'per_device_eval_batch_size': (1, 256, 1),
    'gradient_accumulation_steps': (1, 256, 1),
    'max_length': (32, 32768, 1),
    'split_dataset_ratio': (0, 1, 0.01),
    'lora_rank': (1, 512, 1),
    'lora_alpha': (1, 512, 1),
    'neftune_noise_alpha': (0, 20, 0.5),
    'warmup_ratio': (0, 1, 0.01),
    'num_generations': (1, 64, 1),
    'max_completion_length': (16, 8192, 1),
    'beta': (0, 5, 0.01),
    'vllm_gpu_memory_utilization': (0.05, 0.95, 0.01),
    'eval_limit': (1, 10000, 1),
    'max_new_tokens': (1, 8192, 1),
}
DROPDOWN_KEYS = {
    'model', 'model_type', 'template', 'dataset', 'eval_dataset', 'ref_model', 'deepspeed', 'quant_method',
    'tuner_type', 'torch_dtype', 'attn_impl', 'rlhf_type', 'eval_backend', 'infer_backend', 'vllm_mode',
}
CUSTOM_KEYS = {'model', 'model_type', 'template', 'dataset', 'eval_dataset', 'ref_model', 'deepspeed', 'quant_method'}
SECTIONS = {
    'data': [
        ('数据集设置', True, ['source', 'bundled', 'dataset', 'split_dataset_ratio', 'max_length', 'padding_free',
                          'val_dataset'], 'dataset'),
        ('资源', False, ['dataset_num_proc', 'dataloader_num_workers'], None),
    ],
    'sft': [
        ('模型设置', True, ['model', 'model_type', 'template', 'system'], 'model'),
        ('数据集设置', True, ['dataset', 'split_dataset_ratio', 'max_length', 'padding_free', 'val_dataset'], 'dataset'),
        ('训练参数设置', True,
         ['tuner_type', 'seed', 'torch_dtype', 'use_liger_kernel', 'deepspeed', 'sequence_parallel_size',
          'gradient_checkpointing'], None),
        ('超参数设置(更多参数->其他参数设置)', False, [
            'per_device_train_batch_size', 'per_device_eval_batch_size', 'learning_rate', 'num_train_epochs',
            'gradient_accumulation_steps', 'eval_steps', 'save_steps', 'output_dir', 'attn_impl', 'neftune_noise_alpha'
        ], None),
        ('LoRA参数设置', False,
         ['lora_rank', 'lora_alpha', 'lora_dropout', 'use_rslora', 'use_dora', 'target_modules'], None),
        ('量化参数设置', False, ['quant_method', 'quant_bits'], None),
        ('高级参数设置', False, [
            'warmup_ratio', 'lr_scheduler_type', 'weight_decay', 'logging_steps', 'save_total_limit',
            'dataset_num_proc', 'dataloader_num_workers', 'load_from_cache_file'
        ], None),
    ],
    'sft_eval': [
        ('模型设置', True, ['model', 'model_type', 'template'], 'model'),
        ('评测参数设置', True, [
            'eval_backend', 'eval_dataset', 'eval_limit', 'eval_output_dir', 'custom_eval_config', 'infer_backend',
            'temperature', 'max_new_tokens', 'eval_num_proc', 'verbose'
        ], None),
    ],
    'rl': [
        ('模型设置', True, ['model', 'model_type', 'template', 'system'], 'model'),
        ('数据集设置', True, ['dataset'], 'dataset'),
        ('对齐参数设置', True, ['rlhf_type', 'beta', 'ref_model', 'loss_scale', 'max_completion_length'], None),
        ('奖励模型设置', True, ['reward_funcs', 'reward_weights', 'external_plugins'], None),
        ('采样参数设置', True, [
            'num_generations', 'temperature', 'top_p', 'top_k', 'use_vllm', 'vllm_mode', 'vllm_gpu_memory_utilization',
            'vllm_tensor_parallel_size', 'vllm_max_model_len', 'sleep_level'
        ], None),
        ('训练参数设置', True, [
            'tuner_type', 'seed', 'torch_dtype', 'deepspeed', 'gradient_checkpointing', 'learning_rate',
            'num_train_epochs', 'per_device_train_batch_size', 'gradient_accumulation_steps'
        ], None),
        ('LoRA参数设置', False,
         ['lora_rank', 'lora_alpha', 'target_modules', 'vllm_enable_lora', 'vllm_max_lora_rank'], None),
        ('显存', False, ['offload_model', 'offload_optimizer', 'quant_method', 'quant_bits'], None),
        ('高级参数设置', False, [
            'epsilon', 'epsilon_high', 'num_iterations', 'max_length', 'logging_steps', 'save_steps',
            'save_total_limit', 'output_dir', 'warmup_ratio', 'dataset_num_proc', 'dataloader_num_workers',
            'split_dataset_ratio', 'load_from_cache_file', 'log_completions'
        ], None),
    ],
    'rl_eval': [
        ('模型设置', True, ['model', 'model_type', 'template'], 'model'),
        ('评测参数设置', True, [
            'eval_backend', 'eval_dataset', 'eval_limit', 'eval_output_dir', 'custom_eval_config', 'infer_backend',
            'temperature', 'max_new_tokens', 'eval_num_proc', 'verbose'
        ], None),
    ],
    'promote': [
        ('晋级', True, ['choice', 'reason'], None),
    ],
    'next_round': [
        ('下一轮', True, ['reuse_dataset', 'note'], None),
    ],
    'serve': [
        ('模型设置', True, [
            'model', 'model_type', 'template', 'infer_backend', 'served_model_name', 'host', 'port', 'api_key',
            'torch_dtype'
        ], 'model'),
        ('生成参数', False, ['max_new_tokens', 'temperature', 'vllm_gpu_memory_utilization', 'vllm_max_model_len'], None),
    ],
}

_pipeline = None


def resolve_server():
    host = os.environ.get('WEBUI_SERVER') or os.environ.get('RSI_HOST') or '127.0.0.1'
    port = int(os.environ.get('WEBUI_PORT') or os.environ.get('RSI_PORT') or '7860')
    return host, port


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        from rsi.backends import build_backend
        home = Path(os.environ.get('RSI_HOME', Path(__file__).resolve().parent / 'workspace'))
        _pipeline = Pipeline(home, build_backend())
    return _pipeline


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _display(field, raw):
    key = field['key']
    if field['type'] == 'checkbox':
        return _as_bool(raw)
    if key in SLIDERS:
        low, high, _step = SLIDERS[key]
        if raw in (None, ''):
            return low
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return low
        return min(high, max(low, number))
    if field['type'] == 'select' or key in DROPDOWN_KEYS:
        return None if raw in (None, '') else raw
    if raw is None:
        return ''
    if isinstance(raw, bool):
        return 'true' if raw else 'false'
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    return str(raw)


def _choices_for(key, field, catalog):
    if key in {'model', 'ref_model'}:
        choices = list(catalog['models'])
        if DEFAULT_MODEL not in choices:
            choices.insert(0, DEFAULT_MODEL)
        return choices
    if key == 'model_type':
        return list(catalog['model_types'])
    if key == 'template':
        return list(catalog['templates'])
    if key in {'dataset', 'eval_dataset'}:
        return list(catalog['datasets'])
    if key == 'deepspeed':
        return ['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload']
    if key == 'tuner_type':
        return ['lora', 'full', 'longlora', 'adalora', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft', 'bone']
    if key == 'attn_impl':
        return ['sdpa', 'eager', 'flash_attention_2', 'flash_attention_3', 'flash_attention_4']
    if key == 'torch_dtype':
        return ['bfloat16', 'float16', 'float32']
    pairs = []
    for choice in field.get('choices') or []:
        if isinstance(choice, dict):
            pairs.append((choice.get('label') or str(choice.get('value')), choice.get('value')))
        elif choice not in (None, ''):
            pairs.append(str(choice))
    return pairs


def _dropdown(field) -> bool:
    return field['type'] == 'select' or field['key'] in DROPDOWN_KEYS


def _build_widget(field, value, catalog, scale=4):
    label = control_label(field['label'], field.get('arg') or '')
    info = field.get('help') or None
    key = field['key']
    shown = _display(field, value)
    if field['type'] == 'checkbox':
        return gr.Checkbox(label=label, info=info, value=shown, scale=scale)
    if key in SLIDERS:
        low, high, step = SLIDERS[key]
        return gr.Slider(label=label, info=info, minimum=low, maximum=high, step=step, value=shown, scale=scale)
    if _dropdown(field):
        return gr.Dropdown(
            label=label,
            info=info,
            choices=_choices_for(key, field, catalog),
            value=shown,
            allow_custom_value=key in CUSTOM_KEYS,
            scale=scale,
        )
    lines = 1
    if key in {'reason', 'note'} or field['type'] == 'textarea':
        lines = 4
    if key == 'system':
        lines = 4 if field.get('_stage') == 'rl' else 1
    return gr.Textbox(label=label, info=info, value=shown if isinstance(shown, str) else '' if shown is None else str(shown),
                      lines=lines, scale=scale)


class _Binder:
    def __init__(self):
        self.outputs = []
        self.fillers = []
        self.slots = {}

    def add(self, component, filler, slot=None):
        if slot:
            self.slots[slot] = len(self.outputs)
        self.outputs.append(component)
        self.fillers.append(filler)
        return component

    def values(self, view, runs):
        return [filler(view, runs) for filler in self.fillers]


def _run_choices(runs):
    return [(f"{run['name']} · 第 {run['round_index']} 轮 · {run['id']}", run['id']) for run in runs or []]


def _load(run_id):
    pipeline = get_pipeline()
    runs = pipeline.list_runs()
    if run_id:
        try:
            return pipeline.view(run_id), runs
        except PipelineError:
            pass
    if runs:
        return pipeline.view(runs[0]['id']), runs
    return None, runs


def _hint(model_id) -> str:
    if not model_id or match_model(str(model_id)):
        return ''
    return '无法识别model_type和template,请手动选择'


def _raw(view, stage, key, preset):
    if not view:
        raw = preset.get(stage, {}).get(key, '')
        if stage == 'sft' and key == 'model' and not raw:
            return DEFAULT_MODEL
        if stage == 'sft' and key in {'model_type', 'template'} and not raw:
            return (match_model(DEFAULT_MODEL) or {}).get(key) or ''
        return raw
    if stage == 'serve':
        return (view.get('serve_form') or {}).get(key, preset.get(stage, {}).get(key, ''))
    return ((view.get('forms') or {}).get(stage) or {}).get('values', {}).get(key, preset.get(stage, {}).get(key, ''))


def _logs(view, stage):
    if not view:
        return ''
    if stage == 'serve':
        service = view.get('service') or {}
        return service.get('logs') or ''
    state = (view.get('forms') or {}).get(stage) or {}
    return state.get('logs') or ''


def _command(view, stage):
    if not view:
        return ''
    if stage == 'serve':
        service = view.get('service') or {}
        return service.get('command') or ''
    state = (view.get('forms') or {}).get(stage) or {}
    return state.get('command') or ''


def _board(view) -> str:
    if not view:
        return '还没有实验。先在上面的模型设置里确认基座，再创建。'
    run = view['run']
    rnd = view['round']
    lines = [
        f"实验 **{run['name']}** · 第 {rnd['round_index']} 轮 · 基座 `{rnd['base_model']}` · 后端 `{view['backend']}`",
        '',
    ]
    for item in view['board']:
        metric = f" · {item['metric']}" if item.get('metric') else ''
        lines.append(f"- **{item['label']}** · {STATUS_ZH.get(item['status'], item['status'])}{metric}")
    service = view.get('service')
    if service:
        lines.append('')
        lines.append(
            f"推理服务 · {service.get('status')} · http://{service.get('host')}:{service.get('port')}/v1"
            f" · model `{service.get('served_model_name')}`"
        )
    return '\n'.join(lines)


def _normalize(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if value is None:
        return ''
    return value


def _params(keys, values):
    params = {}
    for key, value in zip(keys, values):
        if key == 'extra_yaml':
            params[key] = value or ''
        else:
            params[key] = _normalize(value)
    return params


def _on_model(model_id, model_type, template):
    meta = match_model(model_id or '')
    if not meta:
        return model_type, template, _hint(model_id)
    return meta.get('model_type') or model_type, meta.get('template') or template, ''


def build_rsi_tab(app):
    """Build the RSI tab. Call this inside an existing ``gr.Tabs()``."""
    schema = public_schema()
    catalog = load_catalog()
    preset = presets()
    fields = {stage: {item['key']: item for item in items} for stage, items in schema['fields'].items()}
    hardware = (schema.get('hardware') or {}).get('label') or ''
    binder = _Binder()
    stage_inputs = {}
    upload = None

    with gr.TabItem('RSI闭环'):
        gr.Markdown(
            '数据 → SFT → SFT 评测 → 在线 RL → RL 评测 → 晋级 → 下一轮 → 推理。'
            f' 参数名与 SWIFT Web-UI / 命令行一致。{hardware}。'
            ' 默认 Mock，不占 GPU。`RSI_BACKEND=swift` 时改为调用真实 swift 命令。'
        )
        with gr.Accordion('新建实验', open=True):
            with gr.Row():
                name = gr.Textbox(label='实验名称', value='IT 运维助手', scale=10)
                scenario = gr.Textbox(
                    label='场景说明',
                    value='企业内部 IT 工单分诊与处理建议。目标是从通用基座迭代出能按步骤处理工单的模型。',
                    lines=2,
                    scale=20,
                )
            create_meta = match_model(DEFAULT_MODEL) or {}
            with gr.Accordion('模型设置', open=True):
                with gr.Row(equal_height=True):
                    create_model = gr.Dropdown(
                        label='模型id或路径(--model)',
                        info='实际的模型id',
                        choices=_choices_for('model', {'key': 'model'}, catalog),
                        value=DEFAULT_MODEL,
                        allow_custom_value=True,
                        scale=20,
                    )
                    create_type = gr.Dropdown(
                        label='模型类型(--model_type)',
                        info='SWIFT已支持的模型类型',
                        choices=catalog['model_types'],
                        value=create_meta.get('model_type') or 'qwen2',
                        allow_custom_value=True,
                        scale=20,
                    )
                    create_template = gr.Dropdown(
                        label='模型Prompt模板类型(--template)',
                        info='选择匹配模型的Prompt模板',
                        choices=catalog['templates'],
                        value=create_meta.get('template') or 'qwen2_5',
                        allow_custom_value=True,
                        scale=20,
                    )
                create_system = gr.Textbox(
                    label='System字段(--system)',
                    info='选择system字段的内容',
                    value=preset.get('sft', {}).get('system') or '',
                    lines=1,
                )
                create_hint = gr.Markdown('')
            create_btn = gr.Button('创建并进入数据阶段', variant='primary')
        create_model.change(_on_model, [create_model, create_type, create_template],
                            [create_type, create_template, create_hint])

        with gr.Row():
            run_dd = gr.Dropdown(label='实验', choices=[], value=None, scale=20, allow_custom_value=False)
            refresh_btn = gr.Button('刷新', scale=2)
        binder.add(gr.Markdown('还没有实验。先在上面的模型设置里确认基座，再创建。'), lambda view, _runs: _board(view),
                   slot='board')

        def bind_widget(component, stage, key, field):
            def filler(view, _runs, stage=stage, key=key, field=field):
                if key == 'choice':
                    raw = _raw(view, stage, key, preset) or (view or {}).get('recommendation')
                    pairs = []
                    for row in (view or {}).get('candidates') or []:
                        score = row.get('score')
                        label = row.get('label') or row.get('kind')
                        if isinstance(score, (int, float)):
                            label = f'{label} · {score}'
                        if row.get('kind') == (view or {}).get('recommendation'):
                            label = f'{label} · 建议'
                        pairs.append((label, row.get('kind')))
                    if pairs:
                        return gr.update(choices=pairs, value=raw or None)
                    return _display(field, raw)
                return _display(field, _raw(view, stage, key, preset))
            return binder.add(component, filler, slot=f'{stage}.{key}')

        pending = []

        def refresh_values(run_id):
            view, runs = _load(run_id)
            selected = view['run']['id'] if view else None
            return [gr.update(choices=_run_choices(runs), value=selected), *binder.values(view, runs)]

        def error_values(run_id, stage_name, message):
            view, runs = _load(run_id)
            values = binder.values(view, runs)
            slot = binder.slots.get(f'{stage_name}.logs')
            if slot is not None:
                values[slot] = message
            elif 'board' in binder.slots:
                values[binder.slots['board']] = message
            selected = view['run']['id'] if view else (run_id or None)
            return [gr.update(choices=_run_choices(runs), value=selected), *values]

        with gr.Tabs():
            for meta_stage in STAGE_META:
                stage = meta_stage['id']
                stage_fields = fields[stage]
                with gr.TabItem(meta_stage['label']):
                    gr.Markdown(meta_stage['summary'])
                    widgets = {}
                    sections = list(SECTIONS.get(stage, []))
                    known = {key for _title, _opened, keys, _kind in sections for key in keys}
                    leftover = [item['key'] for item in schema['fields'][stage] if item['key'] not in known]
                    if leftover:
                        sections.append(('其他参数设置', False, leftover, None))
                    for title, opened, keys, kind in sections:
                        present = [key for key in keys if key in stage_fields]
                        if not present:
                            continue
                        with gr.Accordion(title, open=opened):
                            if kind == 'model':
                                with gr.Row(equal_height=True):
                                    for key in [item for item in present if item != 'system']:
                                        field = dict(stage_fields[key])
                                        field['_stage'] = stage
                                        widgets[key] = bind_widget(
                                            _build_widget(field, _raw(None, stage, key, preset), catalog, scale=20),
                                            stage, key, field)
                                if 'system' in present:
                                    field = dict(stage_fields['system'])
                                    field['_stage'] = stage
                                    widgets['system'] = bind_widget(
                                        _build_widget(field, _raw(None, stage, 'system', preset), catalog, scale=20),
                                        stage, 'system', field)
                                hint = binder.add(
                                    gr.Markdown(''),
                                    lambda view, _runs, stage=stage: _hint(_raw(view, stage, 'model', preset)))
                                model_inputs = [widgets['model']]
                                model_outputs = []
                                if 'model_type' in widgets:
                                    model_inputs.append(widgets['model_type'])
                                    model_outputs.append(widgets['model_type'])
                                else:
                                    model_inputs.append(gr.State(''))
                                if 'template' in widgets:
                                    model_inputs.append(widgets['template'])
                                    model_outputs.append(widgets['template'])
                                else:
                                    model_inputs.append(gr.State(''))
                                model_outputs.append(hint)
                                widgets['model'].change(_on_model, model_inputs, model_outputs)
                            elif kind == 'dataset':
                                primary = [
                                    key for key in ('dataset', 'split_dataset_ratio', 'max_length', 'padding_free')
                                    if key in present
                                ]
                                with gr.Row(equal_height=True):
                                    for key in primary:
                                        field = stage_fields[key]
                                        widgets[key] = bind_widget(
                                            _build_widget(field, _raw(None, stage, key, preset), catalog,
                                                          scale=20 if key == 'dataset' else 10), stage, key, field)
                                rest = [key for key in present if key not in widgets]
                                if rest:
                                    with gr.Row():
                                        for key in rest:
                                            field = stage_fields[key]
                                            widgets[key] = bind_widget(
                                                _build_widget(field, _raw(None, stage, key, preset), catalog, scale=10),
                                                stage, key, field)
                            else:
                                row_keys = []
                                for key in present:
                                    row_keys.append(key)
                                    if len(row_keys) == 4:
                                        with gr.Row():
                                            for item in row_keys:
                                                field = dict(stage_fields[item])
                                                field['_stage'] = stage
                                                widgets[item] = bind_widget(
                                                    _build_widget(field, _raw(None, stage, item, preset), catalog),
                                                    stage, item, field)
                                        row_keys = []
                                if row_keys:
                                    with gr.Row():
                                        for item in row_keys:
                                            field = dict(stage_fields[item])
                                            field['_stage'] = stage
                                            widgets[item] = bind_widget(
                                                _build_widget(field, _raw(None, stage, item, preset), catalog),
                                                stage, item, field)
                    if stage == 'data':
                        upload = gr.File(label='上传数据集', file_types=['.jsonl', '.json', '.txt'], type='filepath')
                    if stage in COMMAND_STAGES or stage == 'data':
                        with gr.Accordion('更多参数', open=False):
                            extra = gr.Textbox(
                                label='更多参数',
                                info='以json格式或--xxx xxx命令行格式填入，也支持 YAML 键值',
                                lines=4,
                            )
                        widgets['extra_yaml'] = binder.add(
                            extra, lambda view, _runs, stage=stage: _raw(view, stage, 'extra_yaml', preset) or '')
                    ordered = [key for key in list(widgets) if key in widgets]
                    stage_inputs[stage] = ordered
                    with gr.Row():
                        if stage == 'data':
                            check_btn = gr.Button('校验格式')
                        elif stage in COMMAND_STAGES:
                            check_btn = gr.Button('仅生成运行命令')
                        else:
                            check_btn = None
                        run_btn = gr.Button(RUN_LABEL[stage], variant='primary')
                    with gr.Accordion('运行时', open=True):
                        command = binder.add(
                            gr.Textbox(label='运行命令', info='执行的实际命令', lines=8),
                            lambda view, _runs, stage=stage: _command(view, stage),
                            slot=f'{stage}.command')
                        logs = binder.add(
                            gr.Textbox(label='日志输出', lines=10),
                            lambda view, _runs, stage=stage: _logs(view, stage),
                            slot=f'{stage}.logs')
                    pending.append(('stage', stage, list(ordered), dict(widgets), upload if stage == 'data' else None,
                                    run_btn, check_btn, command, logs))

            with gr.TabItem('推理服务'):
                gr.Markdown('部署最近一次晋级的检查点。Mock 模式提供 OpenAI 兼容的 /v1/models 与 /v1/chat/completions，不加载权重。')
                serve_fields = fields['serve']
                widgets = {}
                for title, opened, keys, kind in SECTIONS['serve']:
                    present = [key for key in keys if key in serve_fields]
                    with gr.Accordion(title, open=opened):
                        if kind == 'model':
                            with gr.Row(equal_height=True):
                                for key in ('model', 'model_type', 'template'):
                                    field = serve_fields[key]
                                    widgets[key] = bind_widget(
                                        _build_widget(field, _raw(None, 'serve', key, preset), catalog, scale=20),
                                        'serve', key, field)
                            hint = binder.add(gr.Markdown(''),
                                              lambda view, _runs: _hint(_raw(view, 'serve', 'model', preset)))
                            widgets['model'].change(
                                _on_model,
                                [widgets['model'], widgets['model_type'], widgets['template']],
                                [widgets['model_type'], widgets['template'], hint],
                            )
                            rest = [key for key in present if key not in widgets]
                            for offset in range(0, len(rest), 4):
                                with gr.Row():
                                    for key in rest[offset:offset + 4]:
                                        field = serve_fields[key]
                                        widgets[key] = bind_widget(
                                            _build_widget(field, _raw(None, 'serve', key, preset), catalog),
                                            'serve', key, field)
                        else:
                            with gr.Row():
                                for key in present:
                                    field = serve_fields[key]
                                    widgets[key] = bind_widget(
                                        _build_widget(field, _raw(None, 'serve', key, preset), catalog),
                                        'serve', key, field)
                with gr.Accordion('更多参数', open=False):
                    extra = gr.Textbox(label='更多参数', info='以json格式或--xxx xxx命令行格式填入，也支持 YAML 键值', lines=4)
                widgets['extra_yaml'] = binder.add(
                    extra, lambda view, _runs: (_raw(view, 'serve', 'extra_yaml', preset) or ''))
                ordered = list(widgets)
                with gr.Row():
                    preview_btn = gr.Button('仅生成运行命令')
                    start_btn = gr.Button('部署模型', variant='primary')
                    stop_btn = gr.Button('停止部署')
                with gr.Accordion('运行时', open=True):
                    command = binder.add(
                        gr.Textbox(label='运行命令', info='执行的实际命令', lines=8),
                        lambda view, _runs: _command(view, 'serve'),
                        slot='serve.command')
                    logs = binder.add(
                        gr.Textbox(label='日志输出', lines=10),
                        lambda view, _runs: _logs(view, 'serve'),
                        slot='serve.logs')
                with gr.Row():
                    prompt = gr.Textbox(label='请输入：', lines=2, scale=20)
                    send_btn = gr.Button('🚀 发送', variant='primary', scale=2)
                reply = gr.Textbox(label='模型回复', lines=6)
                pending.append(('serve', list(ordered), dict(widgets), start_btn, stop_btn, preview_btn, prompt,
                                send_btn, reply, command, logs))

            with gr.TabItem('审计'):
                gr.Markdown('每一次运行都留在 SQLite 里：参数、结果、命令、开始和结束时间。重跑某一阶段后，后面的步骤会标成已作废。')
                table = binder.add(
                    gr.Dataframe(
                        headers=['轮次', '阶段', '状态', '开始', '结束'],
                        value=[],
                        interactive=False,
                        wrap=True,
                    ),
                    lambda view, _runs: [[
                        step.get('round_index'),
                        step.get('stage'),
                        step.get('status'),
                        step.get('started_at') or '',
                        step.get('ended_at') or '',
                    ] for step in (view or {}).get('steps') or []],
                )
                binder.add(
                    gr.Textbox(label='日志输出', lines=12),
                    lambda view, _runs: '\n\n'.join(
                        f"#{step.get('id')} {step.get('stage')} {step.get('status')}\n{step.get('command') or ''}\n{step.get('logs') or ''}"
                        for step in (view or {}).get('steps') or []),
                )
                _ = table

        for item in pending:
            if item[0] == 'stage':
                _, stage, ordered, widgets, stage_upload, run_btn, check_btn, command, logs = item
                _wire_stage(stage, ordered, widgets, stage_upload, run_dd, run_btn, check_btn, command, logs, binder,
                            refresh_values, error_values)
            else:
                _, ordered, widgets, start_btn, stop_btn, preview_btn, prompt, send_btn, reply, command, logs = item
                _wire_serve(ordered, widgets, run_dd, start_btn, stop_btn, preview_btn, prompt, send_btn, reply,
                            command, logs, binder, refresh_values, error_values)

        def refresh(run_id):
            view, runs = _load(run_id)
            values = binder.values(view, runs)
            if 'board' in binder.slots:
                pass
            # run dropdown is not part of the binder; return it in front.
            selected = view['run']['id'] if view else None
            return [gr.update(choices=_run_choices(runs), value=selected), *values]

        outputs = [run_dd, *binder.outputs]

        def create(exp_name, exp_scenario, model_id, model_type, template, system):
            try:
                view = get_pipeline().create_run(exp_name, exp_scenario, model_id, {
                    'model_type': model_type,
                    'template': template,
                    'system': system,
                })
            except PipelineError as exc:
                current, runs = _load(None)
                values = binder.values(current, runs)
                if 'board' in binder.slots:
                    values[binder.slots['board']] = str(exc)
                selected = current['run']['id'] if current else None
                return [gr.update(choices=_run_choices(runs), value=selected), *values]
            values = binder.values(view, get_pipeline().list_runs())
            return [gr.update(choices=_run_choices(get_pipeline().list_runs()), value=view['run']['id']), *values]

        refresh_btn.click(refresh, [run_dd], outputs)
        run_dd.change(refresh, [run_dd], outputs)
        create_btn.click(create, [name, scenario, create_model, create_type, create_template, create_system], outputs)
        if app is not None:
            app.load(lambda: refresh(None), outputs=outputs)


def _wire_stage(stage, ordered, widgets, upload, run_dd, run_btn, check_btn, command, logs, binder, refresh_values,
                error_values):
    inputs = [run_dd, *[widgets[key] for key in ordered]]
    if upload is not None:
        inputs.append(upload)

    def _collect(run_id, *values):
        file_value = None
        if upload is not None:
            *field_values, file_value = values
        else:
            field_values = values
        params = _params(ordered, field_values)
        if upload is not None and file_value:
            path = Path(file_value)
            saved = get_pipeline().save_upload(path.name, path.read_bytes())
            params['source'] = 'upload'
            params['dataset'] = saved['path']
            if not saved['validation']['ok']:
                raise PipelineError('上传的文件没有通过格式校验')
        return run_id, params

    def run_clicked(run_id, *values):
        try:
            run_id, params = _collect(run_id, *values)
            if not run_id:
                raise PipelineError('请先新建实验')
            get_pipeline().run_stage(run_id, stage, params)
            return refresh_values(run_id)
        except PipelineError as exc:
            return error_values(run_id, stage, str(exc))

    def preview_clicked(run_id, *values):
        try:
            run_id, params = _collect(run_id, *values)
            if not run_id:
                raise PipelineError('请先新建实验')
            if stage == 'data':
                report = get_pipeline().validate_request(params)
                text = f"格式{'通过' if report.get('ok') else '未通过'} · {report.get('num_rows', 0)} 行 · {report.get('format') or ''}"
                if report.get('errors'):
                    text += '\n' + '\n'.join(item.get('error', '') for item in report['errors'])
                return '', text
            result = get_pipeline().preview(run_id, stage, params)
            return result.get('command') or '', '已生成命令，尚未运行。'
        except PipelineError as exc:
            return '', str(exc)

    run_btn.click(run_clicked, inputs, [run_dd, *binder.outputs])
    if check_btn is not None:
        check_btn.click(preview_clicked, inputs, [command, logs])


def _wire_serve(ordered, widgets, run_dd, start_btn, stop_btn, preview_btn, prompt, send_btn, reply, command, logs,
                binder, refresh_values, error_values):
    inputs = [run_dd, *[widgets[key] for key in ordered]]

    def start(run_id, *values):
        try:
            if not run_id:
                raise PipelineError('请先新建实验')
            get_pipeline().start_serve(run_id, _params(ordered, values))
            return refresh_values(run_id)
        except PipelineError as exc:
            return error_values(run_id, 'serve', str(exc))

    def stop(run_id):
        try:
            if not run_id:
                raise PipelineError('请先新建实验')
            get_pipeline().stop_serve(run_id)
            return refresh_values(run_id)
        except PipelineError as exc:
            return error_values(run_id, 'serve', str(exc))

    def preview(run_id, *values):
        try:
            if not run_id:
                raise PipelineError('请先新建实验')
            result = get_pipeline().preview(run_id, 'serve', _params(ordered, values))
            return result.get('command') or '', '已生成命令，尚未运行。'
        except PipelineError as exc:
            return '', str(exc)

    def send(run_id, text, temperature):
        try:
            if not run_id:
                raise PipelineError('请先新建实验')
            if not str(text or '').strip():
                raise PipelineError('请输入内容')
            payload = get_pipeline().chat(run_id, [{'role': 'user', 'content': text}], temperature)
            return (((payload.get('choices') or [{}])[0].get('message') or {}).get('content')) or ''
        except PipelineError as exc:
            return str(exc)

    start_btn.click(start, inputs, [run_dd, *binder.outputs])
    stop_btn.click(stop, [run_dd], [run_dd, *binder.outputs])
    preview_btn.click(preview, inputs, [command, logs])
    send_btn.click(send, [run_dd, prompt, widgets['temperature']], [reply])


def launch_standalone(host=None, port=None):
    """RSI tab plus the Web-UI tab names when the full swift import is unavailable."""
    if host is None or port is None:
        host, port = resolve_server()
    if host in {'0.0.0.0', '::', '[::]'}:
        print('SWIFT Web-UI has no authentication. Prefer 127.0.0.1 unless you put a proxy in front.')
    with gr.Blocks(title='SWIFT WebUI', theme=gr.themes.Base()) as app:
        gr.HTML('<h1><center>🚀SWIFT: 轻量级大模型训练推理框架</center></h1>')
        gr.HTML(
            '<h3><center>默认打开 RSI 闭环。其余标签与 swift web-ui 相同；'
            '当前进程没有加载到 swift 时，那些标签只保留入口说明。</center></h3>'
        )
        with gr.Tabs():
            build_rsi_tab(app)
            for label in OTHER_TABS:
                with gr.TabItem(label):
                    gr.Markdown(
                        '这一页是 ms-swift Web-UI 的原有表单。当前环境没有加载到 swift'
                        '（通常是因为没有安装 torch）。装好依赖后，`python -m rsi` 和 `swift web-ui`'
                        ' 会在同一个 7860 端口打开这些页面的原始 Gradio 表单。'
                    )
    print(f'SWIFT WebUI + RSI: http://{host}:{port}')
    app.queue().launch(server_name=host, server_port=int(port), inbrowser=False, height=800, share=False)
