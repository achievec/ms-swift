# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI mirror of the Web-UI SFT, GRPO, and eval forms.

Labels, help, widget types, and grouping are read from ``swift/ui``. Gradio is
not imported. RSI still owns run / preview / logs, and locks ``--model``,
``--model_type``, and ``--template`` after 新建实验.
"""
import ast
from functools import lru_cache
from pathlib import Path

from rsi.llm_infer_spec import _class_index, _dataclass_fields, _literal_choices

REPO = Path(__file__).resolve().parents[1]
UI_ROOT = REPO / 'swift' / 'ui'

# Chosen only in 新建实验. Shown read-only at the top of every later stage.
LOCKED_FIELDS = ('model', 'model_type', 'template')
# RSI writes these under the experiment directory.
PINNED_FIELDS = ('output_dir', 'eval_output_dir')

# Web-UI task panels. RSI already has stage logs and 运行本阶段.
SKIP_BUILD = frozenset({'Runtime', 'GRPORuntime', 'EvalRuntime', 'LLMRollout'})
SKIP_IDS = frozenset({
    'train_record',
    'clear_cache',
    'reset',
    'submit',
    'dry_run',
    'kill_task',
    'show_log',
    'stop_show_log',
    'refresh_tasks',
    'running_tasks',
    'log',
    'logging_dir',
    'running_cmd',
})

CHOICES_FROM = {
    'model': 'models',
    'ref_model': 'models',
    'reward_model': 'models',
    'teacher_model': 'models',
    'model_type': 'model_types',
    'ref_model_type': 'model_types',
    'reward_model_type': 'model_types',
    'teacher_model_type': 'model_types',
    'template': 'templates',
    'dataset': 'datasets',
    'eval_dataset': 'eval_datasets_by_backend',
}

_MISSING = object()
EVAL_BACKENDS = ['Native', 'OpenCompass', 'VLMEvalKit']


def _tuner_choices():
    from rsi.schemas import TUNER_CHOICES
    return list(TUNER_CHOICES)


def _call_name(node) -> str:
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f'{parent}.{node.attr}' if parent else node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ''


def _eval(node, group):
    """Best-effort constant fold for widget kwargs. ``_MISSING`` if unknown."""
    if node is None:
        return _MISSING
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval(node.operand, group)
        if isinstance(value, (int, float)):
            return -value
        return _MISSING
    if isinstance(node, ast.List):
        values = [_eval(item, group) for item in node.elts]
        if any(item is _MISSING for item in values):
            return _MISSING
        return values
    if isinstance(node, ast.Tuple):
        return _eval(ast.List(elts=node.elts, ctx=ast.Load()), group)
    if isinstance(node, ast.IfExp):
        test = _eval(node.test, group)
        if test is _MISSING:
            return _MISSING
        return _eval(node.body if test else node.orelse, group)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        left = node.left
        if isinstance(left, ast.Attribute) and left.attr == 'group' and node.comparators:
            other = _eval(node.comparators[0], group)
            if other is _MISSING:
                return _MISSING
            return group == other
        return _MISSING
    if isinstance(node, ast.Name):
        if node.id == 'default_backend':
            return 'Native'
        if node.id in {'default_device', 'device_count'}:
            return _MISSING
        return _MISSING
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name in {'list', 'sorted'} and node.args:
            return _eval(node.args[0], group)
        if name in {'get_supported_tuners', 'list.get_supported_tuners'}:
            return _tuner_choices()
        if name.endswith('keys') and 'eval_dataset_dict' in name:
            return list(EVAL_BACKENDS)
        if name.endswith('get') and 'eval_dataset_dict' in name:
            return _MISSING
        return _MISSING
    return _MISSING


def _keyword(call: ast.Call, name: str):
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def _gr_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'gr':
        return func.attr
    return ''


def _call_of(stmt):
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return stmt.value
    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
        return stmt.value
    return None


def _parse(path: Path):
    return ast.parse(path.read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def _parsed_files() -> dict:
    found = {}
    for path in UI_ROOT.rglob('*.py'):
        try:
            found[path.resolve()] = _parse(path)
        except (OSError, SyntaxError, UnicodeError):
            continue
    return found


def _follow_import(path: Path, name: str, seen: set):
    path = path.resolve()
    if path in seen or path not in _parsed_files():
        return None
    seen.add(path)
    tree = _parsed_files()[path]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return path, node
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        alias = next((item for item in node.names if (item.asname or item.name) == name), None)
        if alias is None:
            continue
        target = _module_file(path, node.level or 0, node.module or '')
        if target is None:
            return None
        return _follow_import(target, alias.name, seen)
    return None


def _module_file(importer: Path, level: int, module: str):
    if level:
        parent = importer.parent
        for _ in range(level - 1):
            parent = parent.parent
        parts = [part for part in module.split('.') if part] if module else []
        candidate = parent.joinpath(*parts) if parts else parent
    else:
        return None
    if candidate.is_dir():
        init = candidate / '__init__.py'
        return init if init.is_file() else None
    py = candidate if str(candidate).endswith('.py') else candidate.with_suffix('.py')
    return py if py.is_file() else None


def _resolve_name(file_path: Path, name: str):
    return _follow_import(file_path, name, set())


def _assign(class_node, name: str):
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
            return stmt.value
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == name:
            return stmt.value
    return None


def _class_attr(file_path: Path, class_node: ast.ClassDef, name: str, seen: set):
    key = (str(file_path), class_node.name, name)
    if key in seen:
        return None
    seen.add(key)
    value = _assign(class_node, name)
    if value is not None:
        return value
    for base in class_node.bases:
        if not isinstance(base, ast.Name):
            continue
        found = _resolve_name(file_path, base.id)
        if not found:
            continue
        parent_file, parent = found
        value = _class_attr(parent_file, parent, name, seen)
        if value is not None:
            return value
    return None


def _group_of(file_path: Path, class_node: ast.ClassDef) -> str:
    value = _class_attr(file_path, class_node, 'group', set())
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return 'llm_train'


def _locale_of(file_path: Path, class_node: ast.ClassDef) -> dict:
    merged = {}

    def absorb(path, node, seen):
        key = (str(path), node.name)
        if key in seen:
            return
        seen.add(key)
        for base in node.bases:
            if isinstance(base, ast.Name):
                found = _resolve_name(path, base.id)
                if found:
                    absorb(found[0], found[1], seen)
        value = _assign(node, 'locale_dict')
        if isinstance(value, ast.Dict):
            try:
                data = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                data = {}
            if isinstance(data, dict):
                merged.update(data)

    absorb(file_path, class_node, set())
    flat = {}
    for key, value in merged.items():
        if not isinstance(value, dict):
            continue
        flat[key] = {part: (text.get('zh') or '') for part, text in value.items() if isinstance(text, dict)}
    return flat


def _method(file_path: Path, class_node: ast.ClassDef, name: str):
    for stmt in class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
            return file_path, class_node, stmt
    for base in class_node.bases:
        if not isinstance(base, ast.Name):
            continue
        found = _resolve_name(file_path, base.id)
        if not found:
            continue
        method = _method(found[0], found[1], name)
        if method:
            return method
    return None


def _choices_meta(node, group):
    """Return (choices, choices_from) for a Dropdown choices expr."""
    if node is None:
        return [], ''
    value = _eval(node, group)
    if isinstance(value, list):
        choices = []
        for item in value:
            if item is None:
                choices.append('')
            elif isinstance(item, bool):
                choices.append('true' if item else 'false')
            else:
                choices.append(str(item) if not isinstance(item, str) else item)
        return choices, ''
    name = _call_name(node) if isinstance(node, ast.Call) else ''
    if 'get_model_list' in name:
        return [], 'models'
    if 'get_model_name_list' in name:
        return [], 'model_types'
    if 'TEMPLATE_MAPPING' in name:
        return [], 'templates'
    if 'get_dataset_list' in name:
        return [], 'datasets'
    if 'get_supported_tuners' in name:
        return _tuner_choices(), ''
    if 'eval_dataset_dict' in name and name.endswith('keys'):
        return list(EVAL_BACKENDS), ''
    if 'eval_dataset_dict' in name:
        return [], 'eval_datasets_by_backend'
    return [], ''


def _literal_default_choices(elem: str, arg_cls: str, classes: dict) -> list:
    found = _literal_choices(arg_cls, elem, classes)
    if not found:
        return []
    choices = []
    for item in found:
        if item is None:
            choices.append('')
        else:
            choices.append(str(item))
    if '' in choices:
        choices = [''] + [item for item in choices if item != '']
    return choices


def _widget(call: ast.Call, locales: dict, arguments: set, group: str, arg_cls: str, classes: dict) -> dict | None:
    kind = _gr_name(call)
    if kind in {'', 'State', 'Blocks', 'Row', 'Accordion', 'Tabs', 'TabItem', 'Plot', 'Button'}:
        return None
    elem = ''
    raw_id = _keyword(call, 'elem_id')
    folded = _eval(raw_id, group) if raw_id is not None else _MISSING
    if isinstance(folded, str):
        elem = folded
    if not elem or elem in SKIP_IDS:
        return None
    locale = locales.get(elem) or {}
    label = locale.get('label') or ''
    if label and elem in arguments:
        label = f'{label}(--{elem})'
    if not label:
        label = elem
    default = _eval(_keyword(call, 'value'), group)
    if default is _MISSING:
        default = None
    if isinstance(default, bool):
        pass
    elif default is None and kind == 'Checkbox':
        default = False
    minimum = _eval(_keyword(call, 'minimum'), group)
    maximum = _eval(_keyword(call, 'maximum'), group)
    step = _eval(_keyword(call, 'step'), group)
    lines = _eval(_keyword(call, 'lines'), group)
    multiselect = _eval(_keyword(call, 'multiselect'), group) is True
    allow_custom = _eval(_keyword(call, 'allow_custom_value'), group) is True
    is_list = _eval(_keyword(call, 'is_list'), group) is True
    choices, choices_from = _choices_meta(_keyword(call, 'choices'), group)
    if not choices_from:
        choices_from = CHOICES_FROM.get(elem, '')
    if kind == 'Dropdown' and not choices and not choices_from:
        choices = _literal_default_choices(elem, arg_cls, classes)
    if kind == 'Dropdown' and elem == 'gpu_id':
        multiselect = True
        choices_from = ''
    item = {
        'id': elem,
        'widget': kind,
        'label': label,
        'info': locale.get('info') or '',
        'default': default if default is None or isinstance(default, (str, int, float, bool)) else None,
        'minimum': minimum if isinstance(minimum, (int, float)) else None,
        'maximum': maximum if isinstance(maximum, (int, float)) else None,
        'step': step if isinstance(step, (int, float)) else None,
        'lines': lines if isinstance(lines, int) else None,
        'multiselect': multiselect or is_list and kind == 'Dropdown',
        'allow_custom': allow_custom or choices_from == 'datasets',
        'is_list': is_list,
        'locked': elem in LOCKED_FIELDS or elem in PINNED_FIELDS,
        'choices': choices,
        'choices_from': '' if choices else choices_from,
    }
    return item


def _empty_box():
    return {'blocks': []}


def _drop_empty(blocks: list) -> list:
    kept = []
    for block in blocks:
        kind = block.get('kind')
        if kind == 'row':
            widgets = [item for item in block.get('widgets') or [] if item]
            if widgets:
                kept.append({'kind': 'row', 'widgets': widgets})
            continue
        if kind == 'accordion':
            inner = _drop_empty(block.get('blocks') or [])
            if inner:
                copied = dict(block)
                copied['blocks'] = inner
                kept.append(copied)
            continue
        if kind == 'tabs':
            tabs = []
            for tab in block.get('tabs') or []:
                inner = _drop_empty(tab.get('blocks') or [])
                if inner:
                    tabs.append({'id': tab.get('id') or '', 'label': tab.get('label') or '', 'blocks': inner})
            if tabs:
                kept.append({'kind': 'tabs', 'tabs': tabs})
            continue
        if kind == 'tab':
            inner = _drop_empty(block.get('blocks') or [])
            if inner:
                copied = dict(block)
                copied['blocks'] = inner
                kept.append(copied)
    return kept


def _consume(body, box, ctx):
    """Fill ``box['blocks']``. ``ctx`` carries group, locales, file, and an optional tabs collector."""
    for stmt in body:
        if isinstance(stmt, ast.With) and stmt.items:
            call = stmt.items[0].context_expr
            if not isinstance(call, ast.Call):
                _consume(stmt.body, box, ctx)
                continue
            kind = _gr_name(call) or 'with'
            if kind in {'Blocks', 'with'}:
                _consume(stmt.body, box, ctx)
                continue
            if kind == 'Row':
                row = {'kind': 'row', 'widgets': []}
                _consume_row(stmt.body, row, ctx)
                if row['widgets'] or row.get('_extra'):
                    box['blocks'].append(row)
                continue
            if kind == 'Accordion':
                elem = _eval(_keyword(call, 'elem_id'), ctx['group'])
                elem = elem if isinstance(elem, str) else ''
                label = (ctx['locales'].get(elem) or {}).get('label') or elem or '设置'
                opened = _eval(_keyword(call, 'open'), ctx['group'])
                child = _empty_box()
                _consume(stmt.body, child, ctx)
                box['blocks'].append({
                    'kind': 'accordion',
                    'id': elem,
                    'label': label,
                    'open': opened is not False,
                    'blocks': child['blocks'],
                })
                continue
            if kind == 'Tabs':
                tabs = {'kind': 'tabs', 'tabs': []}
                nested = dict(ctx)
                nested['tabs'] = tabs
                _consume(stmt.body, _empty_box(), nested)
                box['blocks'].append(tabs)
                continue
            if kind == 'TabItem':
                elem = _eval(_keyword(call, 'elem_id'), ctx['group'])
                elem = elem if isinstance(elem, str) else ''
                label = (ctx['locales'].get(elem) or {}).get('label') or elem or '参数'
                child = _empty_box()
                _consume(stmt.body, child, ctx)
                tab = {'kind': 'tab', 'id': elem, 'label': label, 'blocks': child['blocks']}
                if ctx.get('tabs') is not None:
                    ctx['tabs']['tabs'].append(tab)
                else:
                    box['blocks'].append(tab)
                continue
            _consume(stmt.body, box, ctx)
            continue
        call = _call_of(stmt)
        if call is None:
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == 'build_ui' and isinstance(func.value, ast.Name):
            _inline_build(func.value.id, box, ctx)
            continue
        item = _widget(call, ctx['locales'], ctx['arguments'], ctx['group'], ctx['arg_cls'], ctx['classes'])
        if item is None:
            continue
        if not box['blocks'] or box['blocks'][-1].get('kind') != 'row':
            box['blocks'].append({'kind': 'row', 'widgets': []})
        box['blocks'][-1]['widgets'].append(item)


def _consume_row(body, row, ctx):
    holder = _empty_box()
    _consume(body, holder, ctx)
    for block in holder['blocks']:
        if block.get('kind') == 'row':
            row['widgets'].extend(block.get('widgets') or [])
        else:
            # A nested accordion/tabs cannot live inside a row; the caller stores rows only.
            # Attach it by stashing on the row and letting the parent lift it. See _consume Row branch.
            row.setdefault('_extra', []).append(block)


def _inline_build(class_name: str, box, ctx):
    if class_name in SKIP_BUILD:
        return
    found = _resolve_name(ctx['file'], class_name)
    if not found:
        return
    path, node = found
    group = _group_of(path, node)
    method = _method(path, node, 'do_build_ui')
    if not method:
        return
    method_file, _method_class, fn = method
    locales = _locale_of(path, node)
    locales.update(_locale_of(method_file, _method_class))
    # The caller's locale fills ids defined on the root tab (训练参数, GPU, 更多参数).
    merged_locales = dict(ctx.get('root_locales') or {})
    merged_locales.update(locales)
    child_ctx = {
        'file': method_file,
        'group': group,
        'locales': merged_locales,
        'root_locales': ctx.get('root_locales') or {},
        'arguments': ctx['arguments'],
        'arg_cls': ctx['arg_cls'],
        'classes': ctx['classes'],
        'tabs': ctx.get('tabs'),
    }
    child = _empty_box()
    _consume(fn.body, child, child_ctx)
    blocks = _lift_row_extras(child['blocks'])
    tabs = ctx.get('tabs')
    if tabs is not None:
        for block in blocks:
            if block.get('kind') == 'tab':
                tabs['tabs'].append(block)
            elif block.get('kind') == 'tabs':
                tabs['tabs'].extend(block.get('tabs') or [])
            else:
                tabs['tabs'].append({'kind': 'tab', 'id': '', 'label': block.get('label') or class_name, 'blocks': [block]})
        return
    box['blocks'].extend(blocks)


def _lift_row_extras(blocks: list) -> list:
    """Move accordions that were nested inside a Gradio row out to siblings, preserving order."""
    lifted = []
    for block in blocks:
        if block.get('kind') == 'row':
            extras = block.pop('_extra', None)
            if block.get('widgets'):
                lifted.append(block)
            if extras:
                lifted.extend(_lift_row_extras(extras))
            continue
        if block.get('kind') in {'accordion', 'tab'}:
            copied = dict(block)
            copied['blocks'] = _lift_row_extras(block.get('blocks') or [])
            lifted.append(copied)
            continue
        if block.get('kind') == 'tabs':
            copied = dict(block)
            copied['tabs'] = []
            for tab in block.get('tabs') or []:
                tab = dict(tab)
                tab['blocks'] = _lift_row_extras(tab.get('blocks') or [])
                copied['tabs'].append(tab)
            lifted.append(copied)
            continue
        lifted.append(block)
    return lifted


def _walk(root_file: Path, root_class: str, arg_cls: str) -> list:
    found = _resolve_name(root_file, root_class)
    if not found:
        raise RuntimeError(f'{root_class} 不在 {root_file}')
    path, node = found
    method = _method(path, node, 'do_build_ui')
    if not method:
        raise RuntimeError(f'{root_class}.do_build_ui 缺失')
    method_file, method_class, fn = method
    classes = _class_index()
    arguments = set()
    # Argument names live on the dataclass, not the UI class.
    arguments = _dataclass_fields(arg_cls, classes)
    locales = _locale_of(method_file, method_class)
    ctx = {
        'file': method_file,
        'group': _group_of(path, node),
        'locales': locales,
        'root_locales': locales,
        'arguments': arguments,
        'arg_cls': arg_cls,
        'classes': classes,
    }
    box = _empty_box()
    _consume(fn.body, box, ctx)
    return _drop_empty(_lift_row_extras(box['blocks']))


def _index_widgets(blocks, found):
    for block in blocks:
        kind = block.get('kind')
        if kind == 'row':
            for item in block.get('widgets') or []:
                found[item['id']] = item
        elif kind in {'accordion', 'tab'}:
            _index_widgets(block.get('blocks') or [], found)
        elif kind == 'tabs':
            for tab in block.get('tabs') or []:
                _index_widgets(tab.get('blocks') or [], found)


def _flatten(blocks, fields, seen):
    for block in blocks:
        kind = block.get('kind')
        if kind == 'row':
            group = ''
            for item in block.get('widgets') or []:
                if item['id'] in seen:
                    continue
                seen.add(item['id'])
                fields.append(_field_from_widget(item, group))
        elif kind in {'accordion', 'tab'}:
            before = len(fields)
            _flatten(block.get('blocks') or [], fields, seen)
            for item in fields[before:]:
                if not item.get('group'):
                    item['group'] = block.get('label') or '设置'
        elif kind == 'tabs':
            for tab in block.get('tabs') or []:
                before = len(fields)
                _flatten(tab.get('blocks') or [], fields, seen)
                for item in fields[before:]:
                    if not item.get('group'):
                        item['group'] = tab.get('label') or '参数'


def _field_type(item: dict) -> str:
    if item.get('multiselect'):
        return 'multicombo'
    if item['widget'] == 'Checkbox':
        return 'checkbox'
    if item['widget'] == 'Slider':
        return 'slider'
    if item['widget'] == 'Dropdown' and item.get('choices'):
        return 'select'
    if item['widget'] == 'Dropdown':
        return 'combo'
    if item['widget'] == 'Textbox' and (item.get('lines') or 0) >= 4:
        return 'textarea'
    return 'text'


def _field_from_widget(item: dict, group: str) -> dict:
    field = {
        'key': item['id'],
        'label': item['label'],
        'arg': f"--{item['id']}",
        'type': _field_type(item),
        'group': group or '基础',
        'advanced': False,
    }
    if item.get('info'):
        field['info'] = item['info']
    if item.get('choices'):
        field['choices'] = list(item['choices'])
    if item.get('choices_from'):
        field['choices_from'] = item['choices_from']
        field['allow_custom'] = True
    elif item.get('allow_custom'):
        field['allow_custom'] = True
    if item.get('minimum') is not None:
        field['minimum'] = item['minimum']
    if item.get('maximum') is not None:
        field['maximum'] = item['maximum']
    if item.get('step') is not None:
        field['step'] = item['step']
    if item.get('default') is not None:
        field['default'] = item['default']
    if item.get('locked'):
        field['locked'] = True
    return field


def _fill_gpu(blocks, device_count: int):
    count = max(0, int(device_count))
    choices = [str(index) for index in range(count)] + ['cpu']
    if '0' not in choices:
        choices = ['0', *choices]
    default = '0'
    for block in blocks:
        kind = block.get('kind')
        if kind == 'row':
            for item in block.get('widgets') or []:
                if item['id'] == 'gpu_id':
                    item['choices'] = choices
                    item['default'] = default
                    item['multiselect'] = True
                    item['choices_from'] = ''
        elif kind in {'accordion', 'tab'}:
            _fill_gpu(block.get('blocks') or [], device_count)
        elif kind == 'tabs':
            for tab in block.get('tabs') or []:
                _fill_gpu(tab.get('blocks') or [], device_count)


def _layout(blocks) -> dict:
    by_id = {}
    _index_widgets(blocks, by_id)
    fields = []
    _flatten(blocks, fields, set())
    return {'blocks': blocks, 'by_id': by_id, 'fields': fields, 'locked': list(LOCKED_FIELDS)}


@lru_cache(maxsize=4)
def _built(root: str) -> tuple:
    if root == 'sft':
        blocks = _walk(UI_ROOT / 'llm_train' / 'llm_train.py', 'LLMTrain', 'RLHFArguments')
    elif root == 'rl':
        blocks = _walk(UI_ROOT / 'llm_grpo' / 'llm_grpo.py', 'LLMGRPO', 'RLHFArguments')
    elif root == 'eval':
        blocks = _walk(UI_ROOT / 'llm_eval' / 'llm_eval.py', 'LLMEval', 'EvalArguments')
    else:
        raise KeyError(root)
    return tuple(blocks)


def _unwrap_root(blocks: list) -> list:
    """The Web-UI root is the tab itself. RSI already titles the stage, so drop that wrapper."""
    if len(blocks) == 1 and blocks[0].get('kind') == 'tab':
        return blocks[0].get('blocks') or []
    return blocks


def _adapters_widget() -> dict:
    """Eval has no adapters box in the Web-UI. RSI shows the checkpoint LoRA here."""
    return {
        'id': 'adapters',
        'widget': 'Textbox',
        'label': 'adapter id或路径(--adapters)',
        'info': '本轮检查点的 LoRA 路径，可改。',
        'default': '',
        'minimum': None,
        'maximum': None,
        'step': None,
        'lines': 1,
        'multiselect': False,
        'allow_custom': False,
        'is_list': True,
        'locked': False,
        'choices': [],
        'choices_from': '',
    }


def _insert_after_model(blocks: list, widget: dict) -> bool:
    for index, block in enumerate(blocks):
        kind = block.get('kind')
        if kind == 'row' and any(item.get('id') == 'model' for item in block.get('widgets') or []):
            blocks.insert(index + 1, {'kind': 'row', 'widgets': [widget]})
            return True
        if kind in {'accordion', 'tab'} and _insert_after_model(block.get('blocks') or [], widget):
            return True
        if kind == 'tabs':
            for tab in block.get('tabs') or []:
                if _insert_after_model(tab.get('blocks') or [], widget):
                    return True
    return False


# Shown only when at least two GPUs are selected. One card does not use them.
_MULTI_GPU_WIDGETS = frozenset({'deepspeed', 'sequence_parallel_size', 'use_ddp', 'ddp_num'})

# Tuner and task tabs that do not apply to the current selection stay off the form.
# Keys match swift.ui.llm_train Tuner.tabs_to_filter / Task.tabs_to_filter.
_TAB_SHOW = {
    'lora_tab': ('tuner_type', ('lora', 'longlora', 'adalora')),
    'llamapro_tab': ('tuner_type', ('llamapro', )),
    'lisa_tab': ('tuner_type', ('full', )),
    'adalora_tab': ('tuner_type', ('adalora', )),
    'lora_ga_tab': ('init_weights', ('lora-ga', )),
    'reft_tab': ('tuner_type', ('reft', )),
    'vera_tab': ('tuner_type', ('vera', )),
    'boft_tab': ('tuner_type', ('boft', )),
    'fourierft_tab': ('tuner_type', ('fourierft', )),
    'embed_tab': ('task_type', ('embedding', )),
    'seq_cls_tab': ('task_type', ('seq_cls', )),
}


def _annotate_stage(blocks: list) -> None:
    for block in blocks:
        kind = block.get('kind')
        if kind == 'row':
            for item in block.get('widgets') or []:
                if item.get('id') in _MULTI_GPU_WIDGETS:
                    item['min_gpus'] = 2
                # Gradio leaves this dropdown unset. An HTML select would otherwise
                # submit the first entry, which is zero0, and swift then requires deepspeed.
                if item.get('id') == 'deepspeed':
                    choices = [str(choice) for choice in item.get('choices') or []]
                    if '' not in choices:
                        item['choices'] = ['', *choices]
            continue
        if kind in {'accordion', 'tab'}:
            _annotate_stage(block.get('blocks') or [])
            continue
        if kind == 'tabs':
            for tab in block.get('tabs') or []:
                spec = _TAB_SHOW.get(tab.get('id') or '')
                if spec:
                    tab['show_key'] = spec[0]
                    tab['show_values'] = list(spec[1])
                _annotate_stage(tab.get('blocks') or [])


def stage_layout(stage: str, device_count: int = 0) -> dict:
    """``sft`` mirrors LLM预训练/微调, ``rl`` mirrors LLM GRPO, ``eval`` mirrors LLM评测."""
    key = {'sft': 'sft', 'sft_eval': 'eval', 'rl': 'rl', 'rl_eval': 'eval'}[stage]
    blocks = [_clone(block) for block in _unwrap_root(list(_built(key)))]
    if key == 'eval':
        _insert_after_model(blocks, _adapters_widget())
    _fill_gpu(blocks, device_count)
    if key in {'sft', 'rl'}:
        _annotate_stage(blocks)
    return _layout(blocks)


def _clone(value):
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    return value
