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
from rsi.pipeline import PipelineError
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


def inspect_text(text: str, filename: str = 'data.jsonl', limit: int = 8) -> dict:
    rows, report = load_rows(text, filename)
    scored = [score_row(row) for row in rows]
    covered = [item for item in scored if item['score'] is not None]
    full = sum(1 for item in covered if item['score'] == 1)
    preview = []
    for index, row in enumerate(rows[:limit], 1):
        item = score_row(row)
        preview.append({
            'row': index,
            'answer': _assistant_text(row)[:180],
            'score': item['score'],
            'missing': item['missing'],
        })
    return {
        'validation': report,
        'coverage': {
            'num_rows': len(rows),
            'with_solution': len(covered),
            'full_hit': full,
            'partial': len(covered) - full,
        },
        'preview': preview,
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
    }


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


def export_argv(checkpoint: str, output_dir: str) -> list:
    if not checkpoint:
        raise PipelineError('没有可导出的检查点')
    path = Path(checkpoint)
    if not path.exists():
        raise PipelineError(f'检查点不存在：{checkpoint}')
    return ['swift', 'export', '--adapters', str(path), '--merge_lora', 'true', '--output_dir', output_dir]


def promoted_checkpoint(view: dict) -> str:
    for rnd in reversed(view.get('rounds') or []):
        if rnd.get('promoted_checkpoint'):
            return rnd['promoted_checkpoint']
    return ''


def export_status(run_id: str) -> dict:
    with _EXPORT_LOCK:
        current = _EXPORTS.get(run_id)
    return current or {'status': 'idle', 'run_id': run_id}


def start_export(home: Path, run_id: str, checkpoint: str) -> dict:
    output_dir = Path(home) / 'exports' / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logical = export_argv(checkpoint, str(output_dir.resolve()))
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
