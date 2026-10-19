# Copyright (c) ModelScope Contributors. All rights reserved.
"""Validate files against the ms-swift dataset shapes.

Accepted families match AutoPreprocessor (see
docs/source_en/Customization/Custom-dataset.md):

- messages (standard)
- sharegpt ``conversation`` with human/assistant
- alpaca ``instruction`` + ``output``
- query/response, with optional ``solution`` / ``answer`` for GRPO
"""
import json
import re
from pathlib import Path

MAX_BYTES = 20 * 1024 * 1024
ROLES = {'system', 'user', 'assistant', 'tool', 'tool_call'}
PUBLIC_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*(?:#[0-9]+)?$')


def _preview_row(row: dict) -> dict:
    raw = json.dumps(row, ensure_ascii=False)
    if len(raw) > 400:
        raw = raw[:400] + '…'
    return json.loads(raw) if len(json.dumps(row, ensure_ascii=False)) <= 400 else {'preview': raw}


def parse_rows(text: str, filename: str = 'data.jsonl') -> list:
    stripped = text.strip()
    if not stripped:
        raise ValueError('文件为空')
    name = filename.lower()
    if name.endswith('.json') and not name.endswith('.jsonl'):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f'JSON 解析失败：{exc}') from exc
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            if not data:
                raise ValueError('JSON 数组为空')
            return data
        raise ValueError('JSON 须为对象或数组')
    if stripped.startswith('['):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            if not data:
                raise ValueError('JSON 数组为空')
            return data
    rows = []
    for index, line in enumerate(stripped.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'第 {index} 行不是合法 JSON：{exc}') from exc
        if not isinstance(obj, dict):
            raise ValueError(f'第 {index} 行必须是 JSON 对象')
        rows.append(obj)
    if not rows:
        raise ValueError('没有数据行')
    return rows


def classify_row(row: dict):
    """Return (format_name, error). format_name is empty when error is set."""
    if not isinstance(row, dict):
        return '', '每一行必须是 JSON 对象'
    if 'messages' in row:
        messages = row['messages']
        if not isinstance(messages, list) or not messages:
            return '', 'messages 必须是非空数组'
        has_assistant = False
        for message in messages:
            if not isinstance(message, dict):
                return '', 'messages 中的元素必须是对象'
            role = message.get('role')
            if role not in ROLES:
                return '', f'不支持的 role: {role}'
            if 'content' not in message and 'tool_calls' not in message:
                return '', 'message 缺少 content'
            content = message.get('content')
            if role == 'assistant' and content:
                has_assistant = True
        if has_assistant:
            return 'messages', None
        if row.get('solution') or row.get('answer'):
            return 'messages+solution', None
        return 'messages-prompt', None
    conversation = row.get('conversation')
    if isinstance(conversation, list) and conversation:
        for turn in conversation:
            if not isinstance(turn, dict) or 'human' not in turn or 'assistant' not in turn:
                return '', 'sharegpt 的 conversation 需要 human 与 assistant'
        return 'sharegpt', None
    if row.get('instruction') and (row.get('output') is not None or row.get('response') is not None):
        return 'alpaca', None
    if row.get('query') and (row.get('response') is not None or row.get('solution') is not None
                             or row.get('answer') is not None):
        return 'query', None
    return '', '无法识别数据格式（需要 messages / conversation / instruction+output / query+response）'


def validate_rows(rows: list) -> dict:
    formats = {}
    errors = []
    warnings = []
    prompt_only = 0
    for index, row in enumerate(rows, 1):
        kind, error = classify_row(row)
        if error:
            if len(errors) < 20:
                errors.append({'row': index, 'error': error})
            continue
        formats[kind] = formats.get(kind, 0) + 1
        if kind == 'messages-prompt':
            prompt_only += 1
    if prompt_only and prompt_only == len(rows) - len(errors):
        warnings.append('全部样本都没有 assistant / solution。')
    preview = []
    for row in rows[:2]:
        try:
            preview.append(_preview_row(row))
        except (TypeError, ValueError):
            preview.append({'preview': str(row)[:200]})
    return {
        'ok': bool(rows) and not errors,
        'num_rows': len(rows),
        'formats': formats,
        'errors': errors,
        'error_count': len(errors) if len(errors) < 20 else len(errors),
        'warnings': warnings,
        'preview': preview,
    }


def validate_text(text: str, filename: str = 'data.jsonl') -> dict:
    if len(text.encode('utf-8')) > MAX_BYTES:
        return {
            'ok': False,
            'num_rows': 0,
            'formats': {},
            'errors': [{
                'row': 0,
                'error': '文件超过 20MB'
            }],
            'error_count': 1,
            'warnings': [],
            'preview': [],
        }
    try:
        rows = parse_rows(text, filename)
    except ValueError as exc:
        return {
            'ok': False,
            'num_rows': 0,
            'formats': {},
            'errors': [{
                'row': 0,
                'error': str(exc)
            }],
            'error_count': 1,
            'warnings': [],
            'preview': [],
        }
    report = validate_rows(rows)
    # error_count should reflect total, not the capped list
    if report['errors'] and report['num_rows']:
        total = 0
        for index, row in enumerate(rows, 1):
            _, error = classify_row(row)
            if error:
                total += 1
        report['error_count'] = total
    return report


def validate_file(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {
            'ok': False,
            'num_rows': 0,
            'formats': {},
            'errors': [{
                'row': 0,
                'error': f'文件不存在：{path}'
            }],
            'error_count': 1,
            'warnings': [],
            'preview': [],
            'dataset': str(path),
        }
    if path.stat().st_size > MAX_BYTES:
        return {
            'ok': False,
            'num_rows': 0,
            'formats': {},
            'errors': [{
                'row': 0,
                'error': '文件超过 20MB'
            }],
            'error_count': 1,
            'warnings': [],
            'preview': [],
            'dataset': str(path),
        }
    text = path.read_text(encoding='utf-8')
    report = validate_text(text, path.name)
    report['dataset'] = str(path.resolve())
    return report


def validate_public_id(dataset_id: str) -> dict:
    dataset_id = (dataset_id or '').strip()
    if not PUBLIC_ID_RE.match(dataset_id):
        return {
            'ok': False,
            'num_rows': None,
            'formats': {},
            'errors': [{
                'row': 0,
                'error': '公开数据集 id 需形如 org/name 或 org/name#500'
            }],
            'error_count': 1,
            'warnings': [],
            'preview': [],
            'dataset': dataset_id,
        }
    return {
        'ok': True,
        'num_rows': None,
        'formats': {'dataset': 1},
        'errors': [],
        'error_count': 0,
        'warnings': ['公开数据集 id 已记录，训练时再下载。'],
        'preview': [],
        'dataset': dataset_id,
    }
