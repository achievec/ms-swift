# Copyright (c) ModelScope Contributors. All rights reserved.
"""Side tools for data prep, experiment reports, and checkpoint export.

These helpers never start or reorder an RSI stage. Training still goes through
pipeline.run_stage.
"""
import json
import random
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from rsi.backends import swift_launch
from rsi.commands import format_shell
from rsi.dataset_validate import MAX_BYTES, parse_rows, validate_rows
from rsi.pipeline import PipelineError, _adapter_base_model, _is_adapter_checkpoint
from rsi.serving import loopback

_EXPORTS = {}
_EXPORT_LOCK = threading.Lock()


def _assistant_text(row: dict) -> str:
    for message in reversed(row.get('messages') or []):
        if isinstance(message, dict) and message.get('role') == 'assistant' and message.get('content'):
            return str(message['content'])
    for key in ('output', 'response'):
        if row.get(key):
            return str(row[key])
    for turn in reversed(row.get('conversation') or []):
        if isinstance(turn, dict) and turn.get('assistant'):
            return str(turn['assistant'])
    return ''


def _solution_keys(row: dict) -> list:
    raw = row.get('solution')
    if raw is None or raw == '':
        raw = row.get('answer')
    if raw is None or raw == '':
        return []
    return [part.strip() for part in str(raw).split('|') if part.strip()]


def score_row(row: dict) -> dict:
    keys = _solution_keys(row)
    answer = _assistant_text(row)
    if not keys:
        return {'keys': [], 'hit': 0, 'score': None, 'missing': []}
    missing = [key for key in keys if key not in answer]
    hit = len(keys) - len(missing)
    return {'keys': keys, 'hit': hit, 'score': hit / len(keys), 'missing': missing}


def load_rows(text: str, filename: str = 'data.jsonl') -> tuple:
    if len(text.encode('utf-8')) > MAX_BYTES:
        raise PipelineError('文件超过 20MB')
    try:
        rows = parse_rows(text, filename)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    report = validate_rows(rows)
    if not report['ok']:
        raise PipelineError('数据格式不符合要求')
    return rows, report


def _user_text(row: dict) -> str:
    for message in row.get('messages') or []:
        if isinstance(message, dict) and message.get('role') == 'user' and message.get('content'):
            return str(message['content'])
    for key in ('query', 'instruction'):
        if row.get(key):
            return str(row[key])
    return ''


def _clip(text: str, limit: int = 160) -> str:
    compact = ' '.join(str(text or '').split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + '…'


def _row_card(index: int, row: dict) -> dict:
    item = score_row(row)
    return {
        'row': index,
        'user': _clip(_user_text(row)),
        'answer': _clip(_assistant_text(row)),
        'keys': item['keys'],
        'score': item['score'],
        'missing': item['missing'],
    }


def inspect_text(text: str, filename: str = 'data.jsonl', limit: int = 8) -> dict:
    rows, report = load_rows(text, filename)
    cards = [_row_card(index, row) for index, row in enumerate(rows, 1)]
    covered = [item for item in cards if item['score'] is not None]
    full = sum(1 for item in covered if item['score'] == 1)
    return {
        'validation': report,
        'coverage': {
            'num_rows': len(rows),
            'with_solution': len(covered),
            'full_hit': full,
            'partial': len(covered) - full,
        },
        'rows': cards,
        'preview': cards[:limit],
    }


def split_rows(rows: list, eval_ratio: float = 0.2, seed: int = 0) -> tuple:
    if len(rows) < 2:
        raise PipelineError('至少需要 2 条才能拆成训练集和留出集')
    try:
        ratio = float(eval_ratio)
    except (TypeError, ValueError) as exc:
        raise PipelineError('留出比例必须是数字') from exc
    if ratio <= 0 or ratio >= 1:
        raise PipelineError('留出比例需要在 0 和 1 之间')
    eval_count = int(round(len(rows) * ratio))
    eval_count = min(max(eval_count, 1), len(rows) - 1)
    order = list(range(len(rows)))
    random.Random(int(seed)).shuffle(order)
    eval_index = set(order[:eval_count])
    train = [row for index, row in enumerate(rows) if index not in eval_index]
    heldout = [row for index, row in enumerate(rows) if index in eval_index]
    return train, heldout


def _dump(rows: list) -> str:
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)


def write_split(home: Path, text: str, filename: str = 'data.jsonl', eval_ratio: float = 0.2, seed: int = 0) -> dict:
    rows, _report = load_rows(text, filename)
    train, heldout = split_rows(rows, eval_ratio, seed)
    folder = Path(home) / 'workbench' / uuid.uuid4().hex[:8]
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'train.jsonl').write_text(_dump(train), encoding='utf-8')
    (folder / 'eval.jsonl').write_text(_dump(heldout), encoding='utf-8')
    return {
        'train_rows': len(train),
        'eval_rows': len(heldout),
        'train_path': str((folder / 'train.jsonl').resolve()),
        'eval_path': str((folder / 'eval.jsonl').resolve()),
        'seed': int(seed),
        'how_to': '不会改当前实验。下一轮 SFT 把数据来源选成「服务器本地路径」，粘贴训练集。评测数据集选「服务器上的文件」，粘贴留出集。',
    }


def _without_last_assistant(row: dict) -> dict:
    copied = json.loads(json.dumps(row, ensure_ascii=False))
    messages = copied.get('messages')
    if isinstance(messages, list):
        while messages and isinstance(messages[-1], dict) and messages[-1].get('role') == 'assistant':
            messages.pop()
    return copied


def pack_next_round(home: Path, text: str, filename: str = 'data.jsonl', min_score: float = 1.0) -> dict:
    rows, _report = load_rows(text, filename)
    try:
        threshold = float(min_score)
    except (TypeError, ValueError) as exc:
        raise PipelineError('分数线必须是数字') from exc
    kept = []
    retry = []
    for row in rows:
        item = score_row(row)
        if item['score'] is not None and item['score'] >= threshold:
            kept.append(row)
        else:
            retry.append(_without_last_assistant(row))
    folder = Path(home) / 'workbench' / uuid.uuid4().hex[:8]
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'kept.jsonl').write_text(_dump(kept), encoding='utf-8')
    (folder / 'retry_prompts.jsonl').write_text(_dump(retry), encoding='utf-8')
    return {
        'kept_rows': len(kept),
        'retry_rows': len(retry),
        'kept_path': str((folder / 'kept.jsonl').resolve()),
        'retry_path': str((folder / 'retry_prompts.jsonl').resolve()),
        'min_score': threshold,
        'how_to': '可继续用的文件能当下一轮训练集。待补全文件只有题目。先在训练页部署推理，再点「用当前推理补全」。',
    }


def latest_dataset(view: dict) -> dict:
    """Local dataset recorded by the latest succeeded SFT step. Read-only."""
    current_round = (view.get('run') or {}).get('round_index')
    current = None
    latest = None
    for step in view.get('steps') or []:
        if step.get('stage') != 'sft' or step.get('status') != 'succeeded':
            continue
        params = step.get('params') or {}
        result = step.get('result') or {}
        raw = result.get('dataset') or params.get('dataset') or ''
        if not raw:
            continue
        item = {'round_index': step.get('round_index'), 'path': str(raw)}
        latest = item
        if current_round is not None and step.get('round_index') == current_round:
            current = item
    found = current or latest
    if not found:
        raise PipelineError('这一轮还没有成功的 SFT，没有可打开的数据集')
    path = Path(found['path'])
    if not path.is_file():
        raise PipelineError(f'数据不是本地文件，无法在这里打开：{found["path"]}')
    if path.stat().st_size > MAX_BYTES:
        raise PipelineError('数据文件超过 20MB')
    text = path.read_text(encoding='utf-8')
    report = inspect_text(text, path.name)
    checkpoints = []
    for row in view.get('checkpoints') or []:
        if not row.get('active', 1):
            continue
        checkpoints.append({
            'round_index': row.get('round_index'),
            'kind': row.get('kind'),
            'path': row.get('path') or '',
        })
    return {
        'round_index': found['round_index'],
        'path': str(path.resolve()),
        'filename': path.name,
        'text': text,
        'inspect': report,
        'checkpoints': checkpoints,
        'how_to': '上面可以逐条看问题和回答。要准备下一轮文件，用本页的拆分或筛选。',
        'notice': _eval_mismatch(path, view),
    }


def _eval_mismatch(dataset: Path, view: dict) -> str:
    forms = view.get('forms') or {}
    values = ((forms.get('sft_eval') or {}).get('values') or {})
    configured = str(values.get('custom_eval_config') or '')
    if not configured or dataset.name == 'itops_mix.jsonl':
        return ''
    if 'itops_eval' in configured and dataset.name not in configured:
        return '评测还在用 IT 运维冒烟的留出集，和这份训练数据不是一套。下一轮评测请改成同一套题的留出集。'
    return ''


def read_owned_file(home: Path, raw: str) -> tuple:
    """Read a jsonl this workbench wrote. Other paths are refused."""
    root = (Path(home) / 'workbench').resolve()
    path = Path(raw or '').expanduser()
    try:
        path = path.resolve()
    except OSError as exc:
        raise PipelineError('文件不存在') from exc
    if path != root and root not in path.parents:
        raise PipelineError('只能读取数据与运维页生成的文件')
    if not path.is_file():
        raise PipelineError('文件不存在')
    if path.stat().st_size > MAX_BYTES:
        raise PipelineError('文件超过 20MB')
    return path.read_text(encoding='utf-8'), path.name


def render_report(view: dict) -> str:
    run = view.get('run') or {}
    lines = [
        f"# {run.get('name') or '实验'}",
        '',
        f"- 场景：{run.get('scenario') or ''}",
        f"- 基座：{run.get('base_model') or ''}",
        f"- 状态：{run.get('status') or ''}",
        f"- 当前轮：{run.get('round_index') or ''}",
        '',
        '## 各轮',
        '',
    ]
    for rnd in view.get('rounds') or []:
        score = rnd.get('promoted_score')
        score_text = '未晋级' if score is None else str(score)
        lines.append(
            f"- 第 {rnd.get('round_index')} 轮 {rnd.get('status')}，基座 {rnd.get('base_model')}，"
            f"晋级 {rnd.get('promoted_kind') or '无'}（{score_text}）"
        )
    checkpoints = [row for row in (view.get('checkpoints') or []) if row.get('active', 1) and row.get('path')]
    if checkpoints:
        lines.extend(['', '## 检查点', ''])
        for row in checkpoints:
            lines.append(f"- 第 {row.get('round_index')} 轮 {row.get('kind')}：{row.get('path')}")
    lines.extend(['', '## 步骤', ''])
    for step in view.get('steps') or []:
        metrics = ((step.get('result') or {}).get('metrics') or {})
        bits = []
        for key in ('accuracy', 'train_loss', 'reward'):
            if key in metrics:
                bits.append(f'{key}={metrics[key]}')
        extra = '，'.join(bits)
        lines.append(f"- 第 {step.get('round_index')} 轮 {step.get('stage')} {step.get('status')}"
                     + (f"：{extra}" if extra else ''))
    try:
        loaded = latest_dataset(view)
    except PipelineError:
        loaded = None
    if loaded and loaded.get('notice'):
        lines.extend(['', '## 数据', '', f"- {loaded['notice']}"])
    service = view.get('service') or {}
    if service.get('status') == 'running':
        host = loopback(service.get('host') or '127.0.0.1')
        lines.extend(['', '## 推理', '', f"- http://{host}:{service.get('port')}/v1"])
    lines.append('')
    return '\n'.join(lines)


def probe_service(service: dict) -> dict:
    if not service or service.get('status') != 'running':
        return {'ready': False, 'detail': '推理服务未启动'}
    host = loopback(service.get('host') or '127.0.0.1')
    url = f"http://{host}:{service.get('port')}/v1/models"
    request = urllib.request.Request(url)
    if service.get('api_key'):
        request.add_header('Authorization', f"Bearer {service['api_key']}")
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return {'ready': response.status == 200, 'url': url, 'status': response.status}
    except urllib.error.HTTPError as exc:
        return {'ready': False, 'url': url, 'status': exc.code, 'detail': exc.reason}
    except urllib.error.URLError as exc:
        return {'ready': False, 'url': url, 'detail': str(exc.reason)}


FILL_LIMIT = 20


def completion_text(payload: dict) -> str:
    choices = (payload or {}).get('choices') or []
    if not choices or not isinstance(choices[0], dict):
        raise PipelineError('推理没有返回回答')
    message = choices[0].get('message') or {}
    content = message.get('content') or ''
    if not str(content).strip():
        raise PipelineError('推理返回了空回答')
    return str(content)


def _prompt_messages(row: dict) -> list:
    prompt = _without_last_assistant(row)
    messages = [item for item in (prompt.get('messages') or []) if isinstance(item, dict)]
    if messages:
        return messages
    query = row.get('query') or row.get('instruction') or ''
    if not str(query).strip():
        raise PipelineError('有一条数据没有可提问的内容')
    return [{'role': 'user', 'content': str(query)}]


def fill_rows(rows: list, answer_fn, limit: int = FILL_LIMIT) -> tuple:
    """Fill rows that have no assistant answer. Already answered rows stay as they are."""
    filled = []
    answered = 0
    pending = 0
    for row in rows:
        if _assistant_text(row):
            filled.append(row)
            continue
        if answered >= limit:
            filled.append(row)
            pending += 1
            continue
        messages = _prompt_messages(row)
        content = answer_fn(messages)
        copied = _without_last_assistant(row)
        copied['messages'] = list(messages) + [{'role': 'assistant', 'content': content}]
        filled.append(copied)
        answered += 1
    if answered == 0 and pending == 0:
        raise PipelineError('这些题已经有回答，没有需要补全的')
    return filled, answered, pending


def write_completed(home: Path, rows: list, answer_fn, limit: int = FILL_LIMIT) -> dict:
    filled, answered, pending = fill_rows(rows, answer_fn, limit)
    ready = [row for row in filled if _assistant_text(row)]
    waiting = [row for row in filled if not _assistant_text(row)]
    folder = Path(home) / 'workbench' / uuid.uuid4().hex[:8]
    folder.mkdir(parents=True, exist_ok=False)
    next_path = folder / 'next.jsonl'
    next_path.write_text(_dump(ready), encoding='utf-8')
    waiting_path = ''
    if waiting:
        dest = folder / 'still_open.jsonl'
        dest.write_text(_dump(waiting), encoding='utf-8')
        waiting_path = str(dest.resolve())
    return {
        'answered': answered,
        'ready_rows': len(ready),
        'pending_rows': pending,
        'next_path': str(next_path.resolve()),
        'waiting_path': waiting_path,
        'how_to': '不会替换当前实验的数据集。下一轮 SFT 把数据来源选成「服务器本地路径」，粘贴 next.jsonl。',
    }


def export_argv(checkpoint: str, output_dir: str, model: str = '') -> list:
    if not checkpoint:
        raise PipelineError('没有可导出的检查点')
    path = Path(checkpoint)
    if not path.exists():
        raise PipelineError(f'检查点不存在：{checkpoint}')
    if path.is_dir() and _is_adapter_checkpoint(path):
        base = _adapter_base_model(path, model)
        if not base:
            raise PipelineError('这是 LoRA 检查点，还需要基座模型才能合并')
        return [
            'swift', 'export', '--model', base, '--adapters', str(path), '--merge_lora', 'true', '--output_dir',
            output_dir
        ]
    return ['swift', 'export', '--model', str(path), '--output_dir', output_dir]


def promoted_checkpoint(view: dict) -> str:
    for rnd in reversed(view.get('rounds') or []):
        if rnd.get('promoted_checkpoint'):
            return rnd['promoted_checkpoint']
    return ''


def export_checkpoint(view: dict) -> str:
    """Promoted checkpoint when there is one, otherwise this round's RL or SFT checkpoint."""
    promoted = promoted_checkpoint(view)
    if promoted:
        return promoted
    current = (view.get('run') or {}).get('round_index')
    chosen = ''
    rank = -1
    for row in view.get('checkpoints') or []:
        if row.get('active', 1) in (0, False) or not row.get('path'):
            continue
        if current is not None and row.get('round_index') not in (None, current):
            continue
        score = {'sft': 1, 'rl': 2}.get(row.get('kind'), 0)
        if score > rank:
            rank = score
            chosen = row['path']
    return chosen


def export_status(run_id: str) -> dict:
    with _EXPORT_LOCK:
        current = _EXPORTS.get(run_id)
    return current or {'status': 'idle', 'run_id': run_id}


def start_export(home: Path, run_id: str, checkpoint: str, model: str = '') -> dict:
    output_dir = Path(home) / 'exports' / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logical = export_argv(checkpoint, str(output_dir.resolve()), model)
    command = format_shell(logical)
    argv, env = swift_launch(logical)
    record = {
        'status': 'running',
        'run_id': run_id,
        'checkpoint': checkpoint,
        'output_dir': str(output_dir.resolve()),
        'command': command,
        'log': '',
    }
    with _EXPORT_LOCK:
        previous = _EXPORTS.get(run_id)
        if previous and previous.get('status') == 'running':
            raise PipelineError('这个实验已经有一个导出在跑')
        _EXPORTS[run_id] = record

    def _run():
        import subprocess
        log_path = output_dir / 'export.log'
        with log_path.open('w', encoding='utf-8') as handle:
            handle.write(command + '\n')
            handle.flush()
            completed = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False)
        text = log_path.read_text(encoding='utf-8', errors='replace')
        with _EXPORT_LOCK:
            current = _EXPORTS.get(run_id) or record
            current['log'] = text[-4000:]
            current['returncode'] = completed.returncode
            current['status'] = 'succeeded' if completed.returncode == 0 else 'failed'
            _EXPORTS[run_id] = current

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {'ok': True, 'command': command, 'output_dir': str(output_dir.resolve()), 'status': 'running'}
