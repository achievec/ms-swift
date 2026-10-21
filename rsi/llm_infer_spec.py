# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI mirror of the ms-swift Web-UI 「LLM推理」 tab.

Labels, help text, widget types, and defaults are read from ``swift/ui/llm_infer``
and the ``InferArguments`` dataclass tree. Gradio is not imported: the RSI page
draws this spec itself, and deploy/stop/logs stay on the RSI service.
"""
import ast
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UI_DIR = REPO / 'swift' / 'ui' / 'llm_infer'

# Same sentence as swift/template/templates/qwen.py QwenTemplateMeta.default_system.
QWEN_DEFAULT_SYSTEM = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

# Experiment-bound. The page shows them and does not let the operator edit them.
LOCKED_FIELDS = ('model', 'model_type', 'template', 'adapters')

CHOICES_FROM = {
    'model': 'models',
    'model_type': 'model_types',
    'template': 'templates',
}


def template_default_system(template: str) -> str:
    """System prompt the Web-UI fills when the template is Qwen2.5."""
    if (template or '').strip() == 'qwen2_5':
        return QWEN_DEFAULT_SYSTEM
    return ''


def cuda_visible_devices(gpu_id) -> str | None:
    """Map the Web-UI GPU multiselect onto ``CUDA_VISIBLE_DEVICES``.

    ``None`` means the caller did not choose. ``''`` means CPU only.
    A mix of ``cpu`` and GPU ids is rejected, matching ``LLMInfer.deploy``.
    """
    if gpu_id is None or gpu_id == '':
        return None
    if isinstance(gpu_id, (list, tuple)):
        devices = [str(item).strip() for item in gpu_id if str(item).strip()]
    else:
        devices = [part for part in str(gpu_id).replace(',', ' ').split() if part]
    if not devices:
        return None
    if 'cpu' in devices:
        if len(devices) != 1:
            raise ValueError('选择了 CPU 时不能再选 GPU')
        return ''
    return ','.join(devices)


def chat_request_body(served_model_name: str, messages, *, temperature=None, top_k=None, top_p=None,
                      repetition_penalty=None, max_new_tokens=None, model=None, system=None, stream=False) -> dict:
    """OpenAI chat body. ``model`` is the Web-UI LoRA module (empty means the served name)."""
    prepared = [dict(item) for item in (messages or []) if isinstance(item, dict)]
    text = str(system or '').strip()
    if text:
        if not prepared or prepared[0].get('role') != 'system':
            prepared.insert(0, {'role': 'system', 'content': text})
        else:
            prepared[0]['content'] = text
    body = {
        'model': str(model or '').strip() or served_model_name,
        'messages': prepared,
        'stream': bool(stream),
        'stop': ['Observation:'],
    }
    if temperature not in (None, ''):
        body['temperature'] = temperature
    if top_k not in (None, ''):
        body['top_k'] = top_k
    if top_p not in (None, ''):
        body['top_p'] = top_p
    if repetition_penalty not in (None, ''):
        body['repetition_penalty'] = repetition_penalty
    if max_new_tokens not in (None, ''):
        body['max_tokens'] = max_new_tokens
    return body


def _const(node):
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _gr_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'gr':
        return func.attr
    return ''


def _keyword(call: ast.Call, name: str):
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def _call_of(stmt):
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return stmt.value
    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
        return stmt.value
    return None


def _locale_maps() -> dict:
    found = {}
    for path in sorted(UI_DIR.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if not any(isinstance(target, ast.Name) and target.id == 'locale_dict' for target in stmt.targets):
                    continue
                found[node.name] = ast.literal_eval(stmt.value)
    return found


def _class_index() -> dict:
    found = {}
    for path in REPO.joinpath('swift').rglob('*.py'):
        if path.name.startswith('test_'):
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name not in found:
                found[node.name] = node
    return found


def _annotation_names(node) -> set:
    names = set()
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names.add(node.target.id)
    return names


def _dataclass_fields(name: str, classes: dict, seen: set | None = None) -> set:
    if seen is None:
        seen = set()
    if name in seen or name not in classes:
        return set()
    seen.add(name)
    node = classes[name]
    names = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names |= _dataclass_fields(base.id, classes, seen)
    for stmt in node.body:
        names |= _annotation_names(stmt)
    return names


def _literal_choices(class_name: str, field: str, classes: dict) -> list:
    node = classes.get(class_name)
    if node is None:
        return []
    for stmt in node.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        if stmt.target.id != field or not isinstance(stmt.annotation, ast.Subscript):
            continue
        slice_node = stmt.annotation.slice
        elts = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
        return [item.value for item in elts if isinstance(item, ast.Constant)]
    for base in node.bases:
        if isinstance(base, ast.Name):
            found = _literal_choices(base.id, field, classes)
            if found:
                return found
    return []


def _function(class_name: str) -> ast.FunctionDef:
    path = {
        'Model': UI_DIR / 'model.py',
        'Generate': UI_DIR / 'generate.py',
        'Runtime': UI_DIR / 'runtime.py',
        'LLMInfer': UI_DIR / 'llm_infer.py',
    }[class_name]
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == 'do_build_ui':
                return stmt
    raise RuntimeError(f'{class_name}.do_build_ui 不在 {path}')


def _widget(call: ast.Call, stack: list, locales: dict, arguments: set) -> dict | None:
    kind = _gr_name(call)
    if kind in {'', 'State', 'Blocks', 'Row', 'Accordion', 'Tabs', 'TabItem'}:
        return None
    elem = _const(_keyword(call, 'elem_id')) or ''
    tab_label = ''
    for frame in reversed(stack):
        if frame['kind'] == 'TabItem' and frame.get('label'):
            tab_label = frame['label']
            break
    locale = locales.get(elem) or {}
    label = locale.get('label') or ''
    if label and elem in arguments:
        label = f'{label}(--{elem})'
    if kind == 'Button':
        label = locale.get('value') or label
    if kind == 'Chatbot':
        label = 'Chatbot'
    if not elem:
        elem = (tab_label or kind).lower()
        label = label or tab_label or kind
    item = {
        'id': elem,
        'widget': kind,
        'label': label,
        'info': locale.get('info') or '',
        'default': _const(_keyword(call, 'value')),
        'minimum': _const(_keyword(call, 'minimum')),
        'maximum': _const(_keyword(call, 'maximum')),
        'step': _const(_keyword(call, 'step')),
        'lines': _const(_keyword(call, 'lines')),
        'multiselect': bool(_const(_keyword(call, 'multiselect'))),
        'primary': _const(_keyword(call, 'variant')) == 'primary',
        'visible': False if _const(_keyword(call, 'visible')) is False else True,
        'locked': elem in LOCKED_FIELDS,
        'choices': [],
        'choices_from': CHOICES_FROM.get(elem, ''),
    }
    if kind == 'Checkbox' and item['default'] is None:
        item['default'] = False
    return item


def _walk(class_name: str, locales: dict, arguments: set, stack: list | None = None, rows: list | None = None):
    if stack is None:
        stack = []
    if rows is None:
        rows = []
    fn = _function(class_name)

    def current_row():
        for frame in reversed(stack):
            if frame['kind'] == 'Row':
                return frame['widgets']
        return None

    def consume(body):
        for stmt in body:
            if isinstance(stmt, ast.With) and stmt.items:
                call = stmt.items[0].context_expr
                if not isinstance(call, ast.Call):
                    consume(stmt.body)
                    continue
                kind = _gr_name(call) or 'with'
                frame = {
                    'kind': kind,
                    'elem': _const(_keyword(call, 'elem_id')) or '',
                    'label': _const(_keyword(call, 'label')) or '',
                    'open': _const(_keyword(call, 'open')),
                    'widgets': [],
                }
                stack.append(frame)
                if kind == 'Row':
                    rows.append(frame)
                consume(stmt.body)
                stack.pop()
                continue
            call = _call_of(stmt)
            if call is None:
                continue
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == 'build_ui' and isinstance(func.value, ast.Name):
                _walk(func.value.id, locales, arguments, stack, rows)
                continue
            item = _widget(call, stack, locales, arguments)
            if item is None:
                continue
            bucket = current_row()
            if bucket is None:
                frame = {'kind': 'Row', 'elem': '', 'label': '', 'open': None, 'widgets': [item]}
                rows.append(frame)
            else:
                bucket.append(item)

    consume(fn.body)
    return rows


def _flatten_locales(maps: dict) -> dict:
    merged = {}
    for name in ('Generate', 'Model', 'Runtime', 'LLMInfer'):
        for key, value in (maps.get(name) or {}).items():
            merged[key] = {part: (text.get('zh') or '') for part, text in value.items() if isinstance(text, dict)}
    return merged


def _sections(rows: list, locales: dict) -> list:
    runtime_label = (locales.get('runtime_tab') or {}).get('label') or '运行时'
    form_rows = []
    runtime_rows = []
    device_rows = []
    chatbot = None
    prompt = None
    media = []
    actions = []
    for row in rows:
        ids = [item['id'] for item in row['widgets']]
        if 'prompt' in ids:
            prompt = next(item for item in row['widgets'] if item['id'] == 'prompt')
            media.extend(item for item in row['widgets'] if item['id'] != 'prompt')
            continue
        if set(ids) <= {'image', 'video', 'audio'} and ids:
            media.extend(row['widgets'])
            continue
        if set(ids) <= {'clear_history', 'submit'} and ids:
            actions.extend(row['widgets'])
            continue
        if ids == ['chatbot'] or (len(row['widgets']) == 1 and row['widgets'][0]['widget'] == 'Chatbot'):
            chatbot = row['widgets'][0]
            continue
        if 'running_tasks' in ids or 'log' in ids or 'refresh_tasks' in ids:
            runtime_rows.append(row['widgets'])
            continue
        if 'gpu_id' in ids:
            device_rows.append(row['widgets'])
            continue
        form_rows.append(row['widgets'])
    return [
        {'kind': 'rows', 'rows': form_rows},
        {'kind': 'accordion', 'id': 'runtime_tab', 'label': runtime_label, 'open': False, 'rows': runtime_rows},
        {'kind': 'rows', 'rows': device_rows},
        {'kind': 'chatbot', 'item': chatbot},
        {'kind': 'prompt', 'prompt': prompt, 'media': media},
        {'kind': 'actions', 'items': actions},
    ]


@lru_cache(maxsize=1)
def _parsed() -> dict:
    maps = _locale_maps()
    locales = _flatten_locales(maps)
    classes = _class_index()
    arguments = _dataclass_fields('InferArguments', classes)
    rows = _walk('LLMInfer', locales, arguments)
    infer_choices = _literal_choices('InferArguments', 'infer_backend', classes)
    for row in rows:
        for item in row['widgets']:
            if item['id'] == 'infer_backend':
                item['choices'] = list(infer_choices)
    return {'rows': rows, 'locales': locales, 'arguments': arguments}


# RSI keeps 找回部署 and the running-deploy list, and stops the process with 停止推理.
# The Web-UI show/stop pair is not drawn: the deploy log streams while a service is running.
_RUNTIME_DROP = frozenset({'show_log', 'stop_show_log'})
_LOG_INFO = '日志会自动刷新。'


def _tune_runtime(item: dict) -> bool:
    if item['id'] in _RUNTIME_DROP:
        return False
    if item['id'] == 'kill_task':
        item['label'] = '停止推理'
    elif item['id'] == 'log':
        item['info'] = _LOG_INFO
        item['visible'] = True
    return True


def llm_infer_layout(device_count: int = 0) -> dict:
    """Sections in Web-UI order. ``device_count`` is how many CUDA devices the box has."""
    parsed = _parsed()
    rows = []
    for row in parsed['rows']:
        widgets = []
        for item in row['widgets']:
            widget = dict(item)
            if widget['id'] == 'gpu_id':
                count = max(0, int(device_count))
                widget['choices'] = [str(index) for index in range(count)] + ['cpu']
                widget['default'] = '0' if count else 'cpu'
                widget['multiselect'] = True
            widgets.append(widget)
        copied = dict(row)
        copied['widgets'] = widgets
        rows.append(copied)
    sections = _sections(rows, parsed['locales'])
    by_id = {}
    for row in rows:
        for item in row['widgets']:
            by_id[item['id']] = item
    alerts = {}
    for key in ('generate_alert', 'load_alert', 'loaded_alert', 'port_alert'):
        alerts[key] = (parsed['locales'].get(key) or {}).get('value') or ''
    if '展示部署状态' in alerts.get('load_alert', ''):
        alerts['load_alert'] = '部署已开始，日志会自动显示。'
    for item in list(by_id.values()):
        if not _tune_runtime(item):
            by_id.pop(item['id'], None)
    for section in sections:
        if section.get('rows') is not None:
            section['rows'] = [
                [item for item in row if item['id'] not in _RUNTIME_DROP] for row in section['rows']
            ]
            section['rows'] = [row for row in section['rows'] if row]
        if section.get('items') is not None:
            section['items'] = [item for item in section['items'] if item['id'] not in _RUNTIME_DROP]
    return {
        'sections': sections,
        'locked': list(LOCKED_FIELDS),
        'alerts': alerts,
        'by_id': by_id,
    }


def layout_ids(layout: dict) -> list:
    found = []
    for section in layout['sections']:
        if section['kind'] == 'rows':
            for row in section['rows']:
                found.extend(item['id'] for item in row)
        elif section['kind'] == 'accordion':
            for row in section['rows']:
                found.extend(item['id'] for item in row)
        elif section['kind'] == 'chatbot' and section.get('item'):
            found.append(section['item']['id'])
        elif section['kind'] == 'prompt':
            if section.get('prompt'):
                found.append(section['prompt']['id'])
            found.extend(item['id'] for item in section.get('media') or [])
        elif section['kind'] == 'actions':
            found.extend(item['id'] for item in section.get('items') or [])
    return found
