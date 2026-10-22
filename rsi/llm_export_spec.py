# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI mirror of the ms-swift Web-UI 「LLM导出」 tab.

Labels, help, and grouping are read from ``swift/ui/llm_export``. Gradio is not
imported. ``--model``, ``--model_type``, and ``--template`` stay on the
experiment. The LoRA / checkpoint path is a separate editable ``--adapters``
field, filled by the pipeline from this round's weights.
"""
from functools import lru_cache
from pathlib import Path

from rsi.webui_stage_spec import _clone, _fill_gpu, _layout, _unwrap_root, _walk

UI_FILE = Path(__file__).resolve().parents[1] / 'swift' / 'ui' / 'llm_export' / 'llm_export.py'
LOCKED_FIELDS = ('model', 'model_type', 'template')

# ExportArguments.quant_n_samples. The textbox has no value, so the Web-UI
# shows the dataclass default.
QUANT_N_SAMPLES = 256

# Methods that need a calibration ``--dataset`` (see ExportArguments / QuantEngine).
CALIB_QUANT_METHODS = frozenset({'awq', 'gptq', 'gptq_v2'})

# Form / draft / sticky keys for export-time quant. Empty method = merge-only.
EXPORT_QUANT_KEYS = frozenset(('quant_method', 'quant_bits', 'quant_n_samples'))

QUANT_METHOD_INFO = (
    '不选量化 = 只合并 LoRA。选 AWQ 需要校准数据集，且环境已安装 autoawq（pip install autoawq）。'
)
QUANT_BITS_INFO = '不量化时请留空；AWQ/GPTQ 通常填 4。'
DATASET_INFO = (
    'AWQ/GPTQ 量化的校准数据，可多选或填本地路径。仅合并 LoRA 时可不填。'
)


def export_quant_available(method: str) -> str | None:
    """Return a Chinese error if the chosen export quant package is missing."""
    name = str(method or '').strip().lower()
    if name == 'awq':
        import importlib.util
        if importlib.util.find_spec('awq') is None:
            return (
                '当前环境未安装 autoawq（import awq 失败）。'
                '请先执行 pip install autoawq，或将「量化方法」留空仅合并 LoRA。'
            )
        return None
    if name in {'gptq', 'gptq_v2'}:
        import importlib.util
        if importlib.util.find_spec('optimum') is None:
            return (
                '当前环境未安装 optimum（GPTQ 导出需要）。'
                '请先安装 optimum，或将「量化方法」留空仅合并 LoRA。'
            )
        return None
    return None


def _widget(**kwargs):
    item = {
        'id': '',
        'widget': 'Textbox',
        'label': '',
        'info': '',
        'default': None,
        'minimum': None,
        'maximum': None,
        'step': None,
        'lines': None,
        'multiselect': False,
        'allow_custom': False,
        'locked': False,
        'choices': [],
        'choices_from': '',
        'primary': False,
        'visible': True,
    }
    item.update(kwargs)
    return item


def _adapters_row():
    return {
        'kind': 'row',
        'widgets': [
            _widget(
                id='adapters',
                label='adapter id或路径(--adapters)',
                info='只有一个lora模块时填adapter路径或`name=/path`；多个lora模块时填键值对：`name1=/path1 name2=/path2`',
                default='',
                lines=2,
            )
        ],
    }


def _runtime_block():
    return {
        'kind': 'accordion',
        'id': 'runtime_tab',
        'label': '运行时',
        'open': False,
        'blocks': [
            {
                'kind': 'row',
                'widgets': [
                    _widget(
                        id='running_tasks',
                        widget='Dropdown',
                        label='运行中导出任务',
                        info='所有的swift export命令启动的任务',
                    ),
                    _widget(id='refresh_tasks', widget='Button', label='找回导出任务', primary=True),
                    _widget(id='kill_task', widget='Button', label='停止导出'),
                ],
            },
            {
                'kind': 'row',
                'widgets': [
                    _widget(id='log', label='日志输出', info='日志会自动刷新。', lines=6),
                ],
            },
        ],
    }


def _each(blocks, fn):
    for block in blocks or []:
        kind = block.get('kind')
        if kind == 'row':
            for item in block.get('widgets') or []:
                fn(item)
        elif kind in {'accordion', 'tab'}:
            _each(block.get('blocks') or [], fn)


def _blank_choice(item):
    choices = ['' if choice is None else str(choice) for choice in (item.get('choices') or [])]
    if '' not in choices:
        choices = ['', *choices]
    item['choices'] = choices
    item['choices_from'] = ''
    if item.get('default') in (None,):
        item['default'] = ''


def _split_gpu(blocks):
    """The GPU dropdown sits under the 更多参数 row in the Web-UI, not inside it."""
    found = []
    for block in blocks:
        if block.get('kind') != 'row':
            found.append(block)
            continue
        widgets = list(block.get('widgets') or [])
        gpu = [item for item in widgets if item['id'] == 'gpu_id']
        rest = [item for item in widgets if item['id'] != 'gpu_id']
        if gpu and rest:
            copied = dict(block)
            copied['widgets'] = rest
            found.append(copied)
            found.append({'kind': 'row', 'widgets': gpu})
            continue
        found.append(block)
    return found


def _sections(blocks):
    sections = []
    rows = []

    def flush():
        nonlocal rows
        if rows:
            sections.append({'kind': 'rows', 'rows': rows})
            rows = []

    for block in blocks:
        if block.get('kind') == 'row':
            rows.append(block.get('widgets') or [])
            continue
        if block.get('kind') == 'accordion':
            flush()
            inner = [child.get('widgets') or [] for child in block.get('blocks') or [] if child.get('kind') == 'row']
            sections.append({
                'kind': 'accordion',
                'id': block.get('id') or '',
                'label': block.get('label') or '运行时',
                'open': block.get('open') is True,
                'rows': inner,
            })
            continue
        flush()
    flush()
    return sections


def _shape(device_count: int, devices=None):
    blocks = [_clone(block) for block in _unwrap_root(_walk(UI_FILE, 'LLMExport', 'ExportArguments'))]
    _fill_gpu(blocks, device_count, devices=devices)
    _each(blocks, lambda item: item.__setitem__('locked', item['id'] in LOCKED_FIELDS))

    def tune(item):
        if item['id'] == 'quant_method':
            _blank_choice(item)
            item['info'] = QUANT_METHOD_INFO
        if item['id'] == 'quant_bits':
            _blank_choice(item)
            item['info'] = QUANT_BITS_INFO
        if item['id'] == 'quant_n_samples' and item.get('default') is None:
            item['default'] = QUANT_N_SAMPLES
        if item['id'] == 'output_dir':
            item['locked'] = False
        if item['id'] == 'dataset':
            item['multiselect'] = True
            item['allow_custom'] = True
            item['info'] = DATASET_INFO

    _each(blocks, tune)
    blocks = _split_gpu(blocks)
    shaped = []
    for block in blocks:
        widgets = list(block.get('widgets') or [])
        if block.get('kind') == 'row' and any(item['id'] == 'model' for item in widgets):
            shaped.append(block)
            shaped.append(_adapters_row())
            continue
        if block.get('kind') == 'row' and any(item['id'] == 'more_params' for item in widgets):
            shaped.append(_runtime_block())
            widgets.extend([
                _widget(id='export_preview', widget='Button', label='预览命令'),
                _widget(id='export', widget='Button', label='开始导出', primary=True),
            ])
            copied = dict(block)
            copied['widgets'] = widgets
            shaped.append(copied)
            continue
        shaped.append(block)
    return shaped


@lru_cache(maxsize=16)
def export_layout(device_count: int = 0, devices: tuple = ()) -> dict:
    """Sections in Web-UI order. ``devices`` / ``device_count`` select the GPU list."""
    blocks = _shape(int(device_count), devices=devices or None)
    layout = _layout(blocks)
    layout['locked'] = list(LOCKED_FIELDS)
    layout['sections'] = _sections(blocks)
    layout['fields'] = [
        field for field in layout['fields']
        if field['key'] not in {'export', 'export_preview', 'refresh_tasks', 'kill_task', 'running_tasks', 'log'}
    ]
    return layout
