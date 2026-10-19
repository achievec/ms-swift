# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rsi.logtail import ConsoleTee, collapse_cr, normalize_log, strip_ansi
from rsi.store import Store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    home = tmp_path / 'rsi-home'
    app = create_app(home)
    with TestClient(app) as test_client:
        yield test_client, home


def test_normalize_collapses_cr_and_strips_ansi():
    assert collapse_cr('prog 1%\rprog 2%\rprog 3%\n') == 'prog 3%\n'
    assert collapse_cr('a\r\nb\r\n') == 'a\nb\n'
    assert collapse_cr('prog 1%\r') == 'prog 1%'
    assert strip_ansi('\x1b[32mred\x1b[0m') == 'red'
    assert strip_ansi('up\x1b[A') == 'up'
    text = normalize_log('prog 1%\rprog 2%\r\x1b[32mprog 3%\x1b[0m\x1b[A\n[A\nkeep [INFO]\n')
    assert 'prog 1%' not in text
    assert 'prog 2%' not in text
    assert 'prog 3%' in text
    assert '\x1b' not in text
    assert '[A' not in text
    assert '[32m' not in text
    assert '[INFO]' in text


def test_console_tee_throttles_progress_and_strips_ansi():
    lines = []
    clock = {'t': 0.0}
    tee = ConsoleTee('[sft abcdef01]', interval=1.0, now=lambda: clock['t'], write=lines.append)
    tee.feed(b'hello\n')
    tee.feed(b'\rbar 0\rbar 1\rbar 2')
    clock['t'] = 0.2
    tee.feed(b'\rbar 3')
    tee.feed(b'\n\x1b[32mdone\x1b[0m\n')
    tee.finish()
    assert lines[0] == '[sft abcdef01] hello'
    assert '[sft abcdef01] bar 0' in lines
    assert '[sft abcdef01] bar 1' not in lines
    assert '[sft abcdef01] bar 2' not in lines
    assert '[sft abcdef01] bar 3' in lines
    assert '[sft abcdef01] done' in lines
    assert all('\x1b' not in line for line in lines)


def test_tee_writes_raw_log_and_prefixed_console(tmp_path, capsys):
    script = tmp_path / 'tick.py'
    script.write_text(textwrap.dedent("""\
        import sys
        sys.stdout.write('hello\\n')
        sys.stdout.write('\\rbar 0\\rbar 1\\n')
        sys.stdout.write('\\x1b[32mdone\\x1b[0m\\n')
        sys.stdout.flush()
    """), encoding='utf-8')
    log_path = tmp_path / 'swift.log'
    from rsi.logtail import run_and_tee
    with log_path.open('wb') as handle:
        code = run_and_tee([sys.executable, str(script)], None, handle, '[sft abcdef01]', interval=30)
    assert code == 0
    raw = log_path.read_bytes()
    assert b'\rbar 0' in raw
    assert b'\x1b[32m' in raw
    out = capsys.readouterr().out
    assert '[sft abcdef01] hello' in out
    assert '[sft abcdef01] done' in out
    assert '\x1b' not in out


def test_log_tail_offset_cr_and_ansi(client):
    api, home = client
    created = api.post('/api/runs', json={
        'name': 'log-tail',
        'scenario': '日志跟随',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    assert created.status_code == 200, created.text
    run_id = created.json()['run']['id']
    stage_dir = home / 'runs' / run_id / 'round_1' / 'sft'
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / 'swift.log'
    store = Store(home / 'rsi.sqlite')
    with store.transaction() as conn:
        store.insert_step(conn, run_id, 1, 'sft', {'output_dir': str(stage_dir)}, 'swift')

    log_path.write_bytes(b'hello\n')
    first = api.get(f'/api/runs/{run_id}/stages/sft/log', params={'offset': 0, 'round': 1})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body['text'] == 'hello\n'
    assert body['offset'] == 0
    assert body['next_offset'] == 6
    assert body['size'] == 6
    assert body['status'] == 'running'
    assert body['done'] is False

    with log_path.open('ab') as handle:
        handle.write(b'world\n')
    second = api.get(f'/api/runs/{run_id}/stages/sft/log', params={'offset': body['next_offset']})
    assert second.status_code == 200
    assert second.json()['text'] == 'world\n'
    assert 'hello' not in second.json()['text']
    assert second.json()['next_offset'] == 12

    raw = b'prog 1%\rprog 2%\r\x1b[32mprog 3%\x1b[0m\x1b[A\n[A\nkeep [INFO]\n'
    log_path.write_bytes(raw)
    shown = api.get(f'/api/runs/{run_id}/stages/sft/log', params={'offset': 0}).json()['text']
    assert 'prog 1%' not in shown
    assert 'prog 2%' not in shown
    assert 'prog 3%' in shown
    assert '\x1b' not in shown
    assert '[A' not in shown
    assert '[INFO]' in shown

    blob = b'HEAD-MARK\n' + (b'x' * 80000) + b'\nTAIL-MARK\n'
    log_path.write_bytes(blob)
    tailed = api.get(f'/api/runs/{run_id}/stages/sft/log').json()
    assert tailed['offset'] > 0
    assert tailed['offset'] == len(blob) - 65536
    assert tailed['next_offset'] == len(blob)
    assert 'TAIL-MARK' in tailed['text']
    assert 'HEAD-MARK' not in tailed['text']

    missing = api.get('/api/runs/missing/stages/sft/log')
    assert missing.status_code == 404
    assert api.get(f'/api/runs/{run_id}/stages/not-a-stage/log').status_code == 400
    assert api.get(f'/api/runs/{run_id}/stages/sft/log', params={'round': 9}).status_code == 404


def test_log_tail_rejects_paths_outside_the_run(client, tmp_path):
    api, home = client
    run_id = api.post('/api/runs', json={
        'name': 'log-guard',
        'scenario': '日志路径',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    }).json()['run']['id']
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'swift.log').write_text('SECRET\n', encoding='utf-8')
    store = Store(home / 'rsi.sqlite')
    with store.transaction() as conn:
        store.insert_step(conn, run_id, 1, 'sft', {'output_dir': str(outside)}, 'swift')
    body = api.get(f'/api/runs/{run_id}/stages/sft/log').json()
    assert 'SECRET' not in body['text']
    assert body['status'] == 'running'


def test_eval_log_lives_under_eval_output(client):
    api, home = client
    run_id = api.post('/api/runs', json={
        'name': 'eval-log',
        'scenario': '评测日志',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    }).json()['run']['id']
    eval_dir = home / 'runs' / run_id / 'round_1' / 'sft_eval' / 'eval_output'
    eval_dir.mkdir(parents=True)
    (eval_dir / 'swift.log').write_text('eval-line\n', encoding='utf-8')
    store = Store(home / 'rsi.sqlite')
    with store.transaction() as conn:
        store.insert_step(conn, run_id, 1, 'sft_eval', {'eval_output_dir': str(eval_dir)}, 'swift')
    body = api.get(f'/api/runs/{run_id}/stages/sft_eval/log', params={'offset': 0}).json()
    assert body['text'] == 'eval-line\n'
    assert body['done'] is False


def test_running_styles_and_live_log_hooks():
    root = Path(__file__).resolve().parents[1]
    css = (root / 'static' / 'app.css').read_text(encoding='utf-8')
    script = (root / 'static' / 'app.js').read_text(encoding='utf-8')
    assert 'prefers-reduced-motion' in css
    assert 'rsi-breathe' in css
    assert 'rsi-flow' in css
    assert 'rsi-dot' in css
    assert 'data-live-log' in script
    assert 'data-started' in script
    assert '运行中 ·' in script
