# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rsi.logtail import (
    LIVE_TAIL_BYTES,
    MAX_READ,
    TAIL_BYTES,
    ConsoleTee,
    align_start_to_newline,
    collapse_cr,
    normalize_log,
    read_log_window,
    strip_ansi,
    truncate_log_text,
)
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


def test_truncate_log_text_keeps_tail_only():
    assert truncate_log_text('') == ''
    assert truncate_log_text('short') == 'short'
    big = ('a' * 1000) + 'TAIL'
    clipped = truncate_log_text(big, max_bytes=8)
    assert clipped.endswith('TAIL')
    assert len(clipped.encode('utf-8')) <= 8
    huge = 'x' * (TAIL_BYTES + 50_000) + '\nEND\n'
    assert truncate_log_text(huge).endswith('END\n')
    assert len(truncate_log_text(huge).encode('utf-8')) <= TAIL_BYTES


def test_align_start_to_newline_skips_partial_first_line():
    # ``raw`` is the window already seeked to ``start`` (not the whole file).
    full = b'BROKEN-START\nkeep-me\n'
    start, blob = align_start_to_newline(full[3:], 3)
    assert start == len(b'BROKEN-START\n')
    assert blob == b'keep-me\n'
    assert align_start_to_newline(b'only-partial', 2) == (2, b'only-partial')
    assert align_start_to_newline(b'full\n', 0) == (0, b'full\n')


def test_truncate_log_text_aligns_to_newline():
    # Mid-line cut would otherwise paint "31<02:18..." like a broken tqdm tail.
    prefix = 'IGNORE-HEAD\n'
    mid = '93%|████| 931/1000 [35:31<02:18, 2.27s/it]'
    body = prefix + mid + '\n{"loss": "0.045", "global_step/max_steps": "939/1000"}\n'
    raw = body.encode('utf-8')
    # Force the window to start inside the tqdm line.
    cut_inside = len(prefix.encode('utf-8')) + 10
    max_bytes = len(raw) - cut_inside
    clipped = truncate_log_text(body, max_bytes=max_bytes)
    assert not clipped.startswith('31<')
    assert clipped.startswith('{') or clipped.startswith('\n') or 'global_step' in clipped
    assert 'IGNORE-HEAD' not in clipped
    assert '939/1000' in clipped


def test_read_log_window_tail_aligns_and_marks_truncated(tmp_path):
    path = tmp_path / 'swift.log'
    head = b'HEAD-MARK\n'
    mid = b'partial-tqdm-line-XXXXXXXX'
    # Make the file larger than LIVE_TAIL so a default live tail must truncate.
    pad = b'y' * (LIVE_TAIL_BYTES + 10_000)
    tail = b'\nTAIL-MARK\n'
    path.write_bytes(head + mid + pad + tail)
    window = read_log_window(path, None, None, tail=LIVE_TAIL_BYTES)
    assert window['truncated'] is True
    assert window['offset'] > 0
    assert window['text'].startswith('TAIL-MARK') or 'TAIL-MARK' in window['text']
    assert 'HEAD-MARK' not in window['text']
    assert not window['text'].startswith('partial-tqdm')
    # Load from the start in pages.
    first = read_log_window(path, 0, MAX_READ)
    assert first['offset'] == 0
    assert first['truncated'] is False
    assert 'HEAD-MARK' in first['text']
    assert first['next_offset'] > 0


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

    # Larger than LIVE_TAIL so the default panel paint is a truncated, newline-aligned tail.
    blob = b'HEAD-MARK\n' + (b'x' * (LIVE_TAIL_BYTES + 50_000)) + b'\nTAIL-MARK\n'
    log_path.write_bytes(blob)
    tailed = api.get(f'/api/runs/{run_id}/stages/sft/log').json()
    assert tailed['truncated'] is True
    assert tailed['offset'] > 0
    # Raw cut may land mid-`x`; alignment skips to the newline before TAIL-MARK.
    assert tailed['offset'] >= len(blob) - LIVE_TAIL_BYTES
    assert tailed['next_offset'] == len(blob)
    assert tailed['text'].lstrip().startswith('TAIL-MARK')
    assert 'HEAD-MARK' not in tailed['text']

    # Page from the start to reach the head without dumping the whole file at once.
    early = api.get(f'/api/runs/{run_id}/stages/sft/log', params={'offset': 0, 'limit': 64}).json()
    assert early['offset'] == 0
    assert early['truncated'] is False
    assert early['text'].startswith('HEAD-MARK')
    assert early['next_offset'] == 64

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


def test_mock_writes_swift_log_under_the_experiment(client, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    api, home = client
    created = api.post('/api/runs', json={
        'name': '实验日志',
        'scenario': '每次实验的训练日志落在自己的目录',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    assert created.status_code == 200, created.text
    run_id = created.json()['run']['id']
    dotted = api.post(f'/api/runs/{run_id}/stages/sft/run', json={
        'params': {'source': 'bundled', 'bundled': 'itops_mix', 'output_dir': '.'},
    })
    assert dotted.status_code == 200, dotted.text
    assert api.post(f'/api/runs/{run_id}/stages/sft_eval/run', json={'params': {'eval_output_dir': '.'}}).status_code == 200
    assert api.post(f'/api/runs/{run_id}/stages/rl/run', json={'params': {}}).status_code == 200
    assert api.post(f'/api/runs/{run_id}/stages/rl_eval/run', json={'params': {}}).status_code == 200

    expected = {
        'sft': home / 'runs' / run_id / 'round_1' / 'sft' / 'swift.log',
        'sft_eval': home / 'runs' / run_id / 'round_1' / 'sft_eval' / 'eval_output' / 'swift.log',
        'rl': home / 'runs' / run_id / 'round_1' / 'rl' / 'swift.log',
        'rl_eval': home / 'runs' / run_id / 'round_1' / 'rl_eval' / 'eval_output' / 'swift.log',
    }
    # Dual eval also writes a base-model log under eval_output/base/.
    base_logs = {
        'sft_eval': home / 'runs' / run_id / 'round_1' / 'sft_eval' / 'eval_output' / 'base' / 'swift.log',
        'rl_eval': home / 'runs' / run_id / 'round_1' / 'rl_eval' / 'eval_output' / 'base' / 'swift.log',
    }
    out = capsys.readouterr().out
    assert not (tmp_path / 'swift.log').exists()
    for stage, path in expected.items():
        assert path.is_file(), stage
        text = path.read_text(encoding='utf-8')
        assert 'swift ' in text
        assert text not in out
        assert f'实验日志：{path}' in out
        shown = api.get(f'/api/runs/{run_id}/stages/{stage}/log', params={'offset': 0}).json()
        assert shown['size'] == path.stat().st_size
        assert shown['text'].strip()
        assert 'swift' in shown['text']
    for stage, path in base_logs.items():
        assert path.is_file(), stage
        assert 'swift ' in path.read_text(encoding='utf-8')
        assert f'实验日志：{path}' in out
    assert out.count('实验日志：') == 6


def test_training_stream_stays_out_of_the_process_log(tmp_path, monkeypatch, capsys):
    script = tmp_path / 'fake-swift.py'
    script.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        print('TRAIN_STREAM_LINE')
        print('prog 1\\rprog 2')
        if '--output_dir' in args:
            out = Path(args[args.index('--output_dir') + 1])
            ckpt = out / 'checkpoint-3'
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / 'args.json').write_text('{}')
        raise SystemExit(0)
    """), encoding='utf-8')
    script.chmod(0o755)
    monkeypatch.setenv('RSI_SWIFT_BIN', str(script))
    monkeypatch.chdir(tmp_path)
    from rsi.backends import BackendError, MockBackend, MsSwiftBackend

    work = tmp_path / 'runs' / 'abcd1234' / 'round_1' / 'sft'
    backend = MsSwiftBackend()
    outcome = backend.execute('sft', {'output_dir': '.', 'model': 'Qwen/Qwen2.5-7B-Instruct'}, {
        'workdir': str(work),
        'run_id': 'abcd1234',
        'round_index': 1,
    })
    log_path = work / 'swift.log'
    raw = log_path.read_text(encoding='utf-8')
    assert 'TRAIN_STREAM_LINE' in raw
    assert 'TRAIN_STREAM_LINE' in outcome['logs']
    assert (work / 'checkpoint-3').is_dir()
    assert not (tmp_path / 'swift.log').exists()
    assert not (tmp_path / 'checkpoint-3').exists()
    out = capsys.readouterr().out
    assert out.strip() == f'[sft abcd1234] 实验日志：{log_path}'
    assert 'TRAIN_STREAM_LINE' not in out

    with pytest.raises(BackendError, match='当前工作目录'):
        MockBackend().execute('sft', {'output_dir': '.'}, {'workdir': '.', 'round_index': 1, 'run_id': 'x'})
    with pytest.raises(BackendError, match='当前工作目录'):
        MsSwiftBackend().execute('sft', {'output_dir': '.'}, {'workdir': '', 'round_index': 1})
    assert not (tmp_path / 'swift.log').exists()


def test_run_and_tee_can_skip_the_console(tmp_path, capsys):
    script = tmp_path / 'quiet.py'
    script.write_text('print("only-in-file")\n', encoding='utf-8')
    log_path = tmp_path / 'swift.log'
    from rsi.logtail import run_and_tee
    with log_path.open('wb') as handle:
        code = run_and_tee([sys.executable, str(script)], None, handle, '[sft x]', echo=False)
    assert code == 0
    assert b'only-in-file' in log_path.read_bytes()
    assert capsys.readouterr().out == ''


def test_running_styles_and_live_log_hooks():
    root = Path(__file__).resolve().parents[1]
    css = (root / 'static' / 'app.css').read_text(encoding='utf-8')
    script = (root / 'static' / 'app.js').read_text(encoding='utf-8')
    tools = (root / 'static' / 'log-tools.js').read_text(encoding='utf-8')
    assert 'prefers-reduced-motion' in css
    assert 'rsi-breathe' in css
    assert 'rsi-flow' in css
    assert 'rsi-dot' in css
    assert 'data-live-log' in script
    assert 'data-started' in script
    assert '运行中 ·' in script
    assert 'logCursors' in script
    assert 'loadEarlierLogs' in script
    assert 'data-log-earlier' in tools
    assert '加载更早' in tools
