# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rsi.flow import PASS_ACCURACY, decision_actions, eval_verdict


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    app = create_app(tmp_path / 'rsi-home')
    with TestClient(app) as test_client:
        yield test_client


def _run(api, run_id, stage, params=None):
    return api.post(f'/api/runs/{run_id}/stages/{stage}/run', json={'params': params or {}})


def _create(api, name='分支'):
    created = api.post('/api/runs', json={
        'name': name,
        'scenario': '评测之后再决定要不要做 RL',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    assert created.status_code == 200, created.text
    return created.json()['run']['id']


def _through_sft_eval(api, run_id, params=None):
    assert _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200
    evaluated = _run(api, run_id, 'sft_eval', params or {})
    assert evaluated.status_code == 200, evaluated.text
    return evaluated.json()['view']


def _ids(view):
    return [item['id'] for item in view['flow']['actions']]


def test_verdict_uses_accuracy_unless_overridden():
    assert PASS_ACCURACY == 0.5
    assert eval_verdict(None) is None
    assert eval_verdict({'status': 'failed', 'params_json': '{}', 'result_json': '{}'}) is None
    passed = {'status': 'succeeded', 'params_json': '{}', 'result_json': '{"metrics":{"accuracy":0.51}}'}
    failed = {'status': 'succeeded', 'params_json': '{}', 'result_json': '{"metrics":{"accuracy":0.49}}'}
    forced = {'status': 'succeeded', 'params_json': '{"passed": false}', 'result_json': '{"metrics":{"accuracy":0.9}}'}
    assert eval_verdict(passed) is True
    assert eval_verdict(failed) is False
    assert eval_verdict(forced) is False
    actions, prompt = decision_actions(
        {'status': 'active', 'round_index': 1},
        {'status': 'active', 'round_index': 1},
        {'sft_eval': forced},
    )
    assert [item['id'] for item in actions] == ['next']
    assert '不通过' in prompt
    assert 'rl' not in [item['id'] for item in actions]
    assert 'finish' not in [item['id'] for item in actions]
    assert 'stop' not in [item['id'] for item in actions]


def test_passed_sft_eval_offers_rl_next_or_finish_and_skip_keeps_checkpoint(client):
    api = client
    run_id = _create(api)
    view = _through_sft_eval(api, run_id)
    assert _ids(view) == ['rl', 'next', 'finish']
    assert view['forms']['rl']['status'] == 'ready'
    assert view['forms']['promote']['status'] == 'ready'
    assert view['forms']['rl_eval']['status'] == 'locked'
    node = {item['id']: item for item in view['flow']['nodes']}
    assert node['sft_eval']['verdict'] == '通过'
    assert 'stop' not in node
    assert node['sft_fail']['status'] == 'locked'
    assert node['sft_next']['status'] == 'ready'
    assert node['sft_finish']['status'] == 'ready'
    assert node['rl']['status'] == 'ready'
    edges = view['flow']['edges']
    assert any(edge['from'] == 'sft_eval' and edge['to'] == 'rl' and edge['live'] and edge['label'] == 'RL'
               for edge in edges)
    assert any(edge['from'] == 'sft_eval' and edge['to'] == 'sft_finish' and edge['label'] == '完成' and edge['live']
               for edge in edges)
    assert any(edge['from'] == 'sft_eval' and edge['to'] == 'sft_next' and edge['label'] == '下一轮数据' and edge['live']
               for edge in edges)
    assert any(edge['from'] == 'sft_eval' and edge['to'] == 'sft_fail' and edge['label'] == '不通过' and not edge['live']
               for edge in edges)

    decided = api.post(f'/api/runs/{run_id}/decide', json={'action': 'next'})
    assert decided.status_code == 200, decided.text
    assert decided.json()['stage'] == 'promote'
    sft_path = next(row['path'] for row in decided.json()['view']['candidates'] if row['kind'] == 'sft')
    assert sft_path
    promoted = _run(api, run_id, 'promote', {'choice': 'sft', 'reason': '跳过 RL，带上 SFT 检查点'})
    assert promoted.status_code == 200, promoted.text
    nxt = _run(api, run_id, 'next_round', {'reuse_dataset': True})
    assert nxt.status_code == 200, nxt.text
    body = nxt.json()['view']
    assert body['run']['round_index'] == 2
    assert body['round']['base_model'] == sft_path
    assert not any(step['stage'] == 'rl' and step['status'] == 'succeeded' for step in body['steps'])
    assert any(step['stage'] == 'sft' and step['result']['checkpoint'] == sft_path for step in body['steps'])


def test_failed_sft_eval_only_opens_next_data(client):
    api = client
    run_id = _create(api, '评测未通过')
    view = _through_sft_eval(api, run_id, {'passed': False})
    assert _ids(view) == ['next']
    assert view['forms']['rl']['status'] == 'locked'
    assert view['run']['status'] == 'active'
    node = {item['id']: item for item in view['flow']['nodes']}
    assert node['sft_eval']['verdict'] == '不通过'
    assert 'stop' not in node
    assert node['sft_fail']['status'] == 'failed'
    assert node['sft_next']['status'] == 'ready'
    assert node['sft_finish']['status'] == 'locked'
    assert node['rl']['status'] == 'locked'
    assert any(edge['from'] == 'sft_eval' and edge['to'] == 'rl' and edge['label'] == 'RL' and not edge['live']
               for edge in view['flow']['edges'])
    assert any(edge['from'] == 'sft_eval' and edge['to'] == 'sft_fail' and edge['label'] == '不通过' and edge['live']
               for edge in view['flow']['edges'])
    assert any(edge['from'] == 'sft_fail' and edge['to'] == 'sft_next' and edge['live'] for edge in view['flow']['edges'])
    assert any(edge['from'] == 'sft_eval' and edge['to'] == 'sft_finish' and not edge['live'] for edge in view['flow']['edges'])
    blocked = _run(api, run_id, 'rl', {})
    assert blocked.status_code == 400
    assert blocked.json()['detail'] == 'SFT 评测未通过，不能进入 RL'
    finish = api.post(f'/api/runs/{run_id}/decide', json={'action': 'finish'})
    assert finish.status_code == 400
    stopped = api.post(f'/api/runs/{run_id}/decide', json={'action': 'stop'})
    assert stopped.status_code == 400
    still = api.get(f'/api/runs/{run_id}').json()
    assert still['run']['status'] == 'active'
    assert still['flow']['actions']
    assert any(item['kind'] == 'sft' and item['active'] for item in still['checkpoints'])

    other = _create(api, '未通过后下一轮')
    _through_sft_eval(api, other, {'passed': False})
    assert api.post(f'/api/runs/{other}/decide', json={'action': 'next'}).status_code == 200
    base = api.get(f'/api/runs/{other}').json()['candidates']
    base_path = next(row['path'] for row in base if row['kind'] == 'base')
    sft_path = next(row['path'] for row in base if row['kind'] == 'sft')
    assert _run(api, other, 'promote', {'choice': 'base'}).status_code == 200
    nxt = _run(api, other, 'next_round', {})
    assert nxt.status_code == 200, nxt.text
    assert nxt.json()['view']['round']['base_model'] == base_path
    assert any(item['kind'] == 'sft' and item['path'] == sft_path for item in nxt.json()['view']['checkpoints'])


def test_finish_ends_the_run_and_rl_eval_splits_pass_and_fail(client):
    api = client
    run_id = _create(api, '直接完成')
    _through_sft_eval(api, run_id)
    finished = api.post(f'/api/runs/{run_id}/decide', json={'action': 'finish'})
    assert finished.status_code == 200, finished.text
    view = finished.json()['view']
    assert view['run']['status'] == 'completed'
    assert finished.json()['stage'] == 'finish'
    assert view['mutable'] is False
    # 「完成」must not permanently lock the next-round exit.
    assert _ids(view) == ['next']
    node = {item['id']: item for item in view['flow']['nodes']}
    assert node['sft_finish']['status'] == 'succeeded'
    assert node['sft_next']['status'] == 'ready'
    assert node['sft_next']['clickable'] is True
    assert node['rl_finish']['status'] == 'locked'
    assert node['sft_fail']['status'] == 'locked'
    assert 'stop' not in node
    assert any(item['kind'] == 'sft' for item in view['checkpoints'])
    assert _run(api, run_id, 'rl', {}).status_code == 400
    assert api.post(f'/api/runs/{run_id}/decide', json={'action': 'finish'}).status_code == 400
    assert api.post(f'/api/runs/{run_id}/decide', json={'action': 'rl'}).status_code == 400
    reopened = api.post(f'/api/runs/{run_id}/decide', json={'action': 'next'})
    assert reopened.status_code == 200, reopened.text
    assert reopened.json()['stage'] == 'promote'
    opened = reopened.json()['view']
    assert opened['run']['status'] == 'active'
    assert opened['round']['status'] == 'active'
    assert opened['mutable'] is True
    sft_path = next(row['path'] for row in opened['candidates'] if row['kind'] == 'sft')
    assert _run(api, run_id, 'promote', {'choice': 'sft', 'reason': '完成后继续'}).status_code == 200
    nxt = _run(api, run_id, 'next_round', {'reuse_dataset': True})
    assert nxt.status_code == 200, nxt.text
    assert nxt.json()['view']['run']['round_index'] == 2
    assert nxt.json()['view']['round']['base_model'] == sft_path

    continued = _create(api, 'RL 评测未通过')
    _through_sft_eval(api, continued)
    assert _run(api, continued, 'rl', {}).status_code == 200
    evaluated = _run(api, continued, 'rl_eval', {'passed': False})
    assert evaluated.status_code == 200, evaluated.text
    failed_view = evaluated.json()['view']
    assert _ids(failed_view) == ['next']
    assert failed_view['flow']['nodes']
    failed_nodes = {item['id']: item for item in failed_view['flow']['nodes']}
    assert failed_nodes['rl_eval']['verdict'] == '不通过'
    assert failed_nodes['rl_fail']['status'] == 'failed'
    assert failed_nodes['rl_next']['status'] == 'ready'
    assert failed_nodes['rl_finish']['status'] == 'locked'
    assert 'stop' not in failed_nodes
    assert any(edge['from'] == 'rl_eval' and edge['to'] == 'rl_fail' and edge['label'] == '不通过' and edge['live']
               for edge in failed_view['flow']['edges'])
    assert any(edge['from'] == 'rl_eval' and edge['to'] == 'rl_next' and edge['label'] == '下一轮数据' and edge['live']
               for edge in failed_view['flow']['edges'])
    assert any(edge['from'] == 'rl_eval' and edge['to'] == 'rl_finish' and edge['label'] == '完成' and not edge['live']
               for edge in failed_view['flow']['edges'])
    rl_path = next(item['path'] for item in failed_view['candidates'] if item['kind'] == 'rl')
    assert api.post(f'/api/runs/{continued}/decide', json={'action': 'finish'}).status_code == 400
    assert api.post(f'/api/runs/{continued}/decide', json={'action': 'rl'}).status_code == 400
    assert api.post(f'/api/runs/{continued}/decide', json={'action': 'next'}).status_code == 200
    assert _run(api, continued, 'promote', {'choice': 'rl'}).status_code == 200
    nxt = _run(api, continued, 'next_round', {})
    assert nxt.status_code == 200, nxt.text
    assert nxt.json()['view']['round']['base_model'] == rl_path

    passed_rl = _create(api, 'RL 评测通过')
    _through_sft_eval(api, passed_rl)
    assert _run(api, passed_rl, 'rl', {}).status_code == 200
    rl_done = _run(api, passed_rl, 'rl_eval', {})
    assert rl_done.status_code == 200, rl_done.text
    passed_view = rl_done.json()['view']
    assert _ids(passed_view) == ['next', 'finish']
    passed_nodes = {item['id']: item for item in passed_view['flow']['nodes']}
    assert passed_nodes['rl_next']['status'] == 'ready'
    assert passed_nodes['rl_finish']['status'] == 'ready'
    assert passed_nodes['rl_fail']['status'] == 'locked'
    assert any(edge['from'] == 'rl_eval' and edge['to'] == 'rl_finish' and edge['live'] for edge in passed_view['flow']['edges'])
    assert any(edge['from'] == 'rl_eval' and edge['to'] == 'rl_fail' and not edge['live'] for edge in passed_view['flow']['edges'])

    # Screenshot case: RL → RL 评测通过 → 完成 must leave 下一轮数据 unlocked.
    done_rl = api.post(f'/api/runs/{passed_rl}/decide', json={'action': 'finish'})
    assert done_rl.status_code == 200, done_rl.text
    after = done_rl.json()['view']
    assert after['run']['status'] == 'completed'
    assert _ids(after) == ['next']
    after_nodes = {item['id']: item for item in after['flow']['nodes']}
    assert after_nodes['rl_finish']['status'] == 'succeeded'
    assert after_nodes['rl_next']['status'] == 'ready'
    assert after_nodes['rl_next']['clickable'] is True
    assert after_nodes['rl_fail']['status'] == 'locked'
    assert any(edge['from'] == 'rl_eval' and edge['to'] == 'rl_next' and edge['live'] for edge in after['flow']['edges'])
    assert any(edge['from'] == 'rl_eval' and edge['to'] == 'rl_fail' and not edge['live'] for edge in after['flow']['edges'])
    resume = api.post(f'/api/runs/{passed_rl}/decide', json={'action': 'next'})
    assert resume.status_code == 200, resume.text
    assert resume.json()['view']['run']['status'] == 'active'
    assert resume.json()['stage'] == 'promote'


def test_guide_draws_every_branch_before_evaluation(client):
    api = client
    run_id = _create(api, '未评测')
    view = api.get(f'/api/runs/{run_id}').json()
    nodes = {item['id']: item for item in view['flow']['nodes']}
    for node_id in (
        'sft', 'sft_eval', 'sft_fail', 'sft_next', 'sft_finish',
        'rl', 'rl_eval', 'rl_fail', 'rl_next', 'rl_finish', 'promote',
    ):
        assert node_id in nodes, node_id
    assert 'stop' not in nodes
    assert nodes['sft_fail']['label'] == '不通过'
    assert nodes['sft_next']['label'] == '下一轮数据'
    assert nodes['sft_finish']['label'] == '完成'
    assert nodes['rl_fail']['label'] == '不通过'
    assert nodes['rl_next']['label'] == '下一轮数据'
    assert nodes['rl_finish']['label'] == '完成'
    assert nodes['promote']['label'] == '晋级'
    assert nodes['promote']['h'] == nodes['sft']['h'] == nodes['sft_eval']['h']
    edges = view['flow']['edges']

    def linked(src, dst, label):
        return any(edge['from'] == src and edge['to'] == dst and edge['label'] == label for edge in edges)

    assert linked('sft_eval', 'sft_fail', '不通过')
    assert linked('sft_eval', 'rl', 'RL')
    assert linked('sft_eval', 'sft_finish', '完成')
    assert linked('sft_eval', 'sft_next', '下一轮数据')
    assert linked('rl_eval', 'rl_fail', '不通过')
    assert linked('rl_eval', 'rl_finish', '完成')
    assert linked('rl_eval', 'rl_next', '下一轮数据')
    assert linked('promote', 'sft', '下一轮')
    assert 'data' not in nodes
    assert all(not edge['live'] for edge in edges if edge['from'] in {'sft_eval', 'rl_eval'})
    boxes = view['flow']['nodes']
    assert all(item['h'] >= 52 for item in boxes)
    for index, left in enumerate(boxes):
        for right in boxes[index + 1:]:
            apart = (left['x'] + left['w'] <= right['x'] or right['x'] + right['w'] <= left['x']
                     or left['y'] + left['h'] <= right['y'] or right['y'] + right['h'] <= left['y'])
            assert apart, (left['id'], right['id'])


def test_page_hides_hardware_and_stop_copy():
    root = Path(__file__).resolve().parents[1]
    html = root.joinpath('static', 'index.html').read_text(encoding='utf-8')
    script = root.joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert 'A100' not in html
    assert '60GB' not in html
    assert 'hardware-pill' not in html
    assert '18 核' not in html
    assert 'A100' not in script
    assert 'hardware-pill' not in script
    assert '终止' not in html
    assert '终止' not in script
    assert '大模型训练平台' in html
    assert '递归自改进' not in html
    assert '新建 RSI' not in html
    assert '>RSI<' not in html
    assert "action: 'next'" in script
    assert '仍可点「下一轮数据」' in script
    assert "status === 'completed'" in script


def _js_function(source, name):
    marker = f'function {name}('
    start = source.index(marker)
    brace = source.index('{', start)
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f'{name} is missing its closing brace')


def test_header_base_caption_collapses_when_unchanged():
    page = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert 'baseCaption(view.run.base_model, view.round.base_model)' in page
    assert '最初基座 <span class="mono">${escapeHtml(view.run.base_model)}' not in page
    script = '\n'.join(_js_function(page, name) for name in ('escapeHtml', 'normalizeBase', 'sameBase', 'baseCaption'))
    script += r"""
const cases = [
  ['same-id', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct'],
  ['slash', 'Qwen/Qwen2.5-7B-Instruct/', 'Qwen//Qwen2.5-7B-Instruct'],
  ['windows', 'C:\\models\\Qwen', 'C:/models/Qwen'],
  ['space', '  Qwen/Qwen2.5-7B-Instruct  ', 'Qwen/Qwen2.5-7B-Instruct'],
  ['empty', '', ''],
  ['blank', '   ', null],
  ['escape', 'A<B>&', 'A<B>&'],
  ['promoted', 'Qwen/Qwen2.5-7B-Instruct', '/data/out/checkpoint-100'],
  ['suffix', 'Qwen/Qwen2.5-7B-Instruct', '/ckpt/Qwen/Qwen2.5-7B-Instruct'],
];
for (const [name, initial, current] of cases) {
  console.log(JSON.stringify({ name, html: baseCaption(initial, current) }));
}
"""
    completed = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    rows = {item['name']: item['html'] for item in map(json.loads, completed.stdout.splitlines())}
    single = '基座 <span class="mono">Qwen/Qwen2.5-7B-Instruct</span>'
    assert rows['same-id'] == single
    assert rows['slash'] == '基座 <span class="mono">Qwen//Qwen2.5-7B-Instruct</span>'
    assert rows['windows'] == '基座 <span class="mono">C:/models/Qwen</span>'
    assert rows['space'] == single
    assert rows['empty'] == '基座 <span class="mono"></span>'
    assert rows['blank'] == '基座 <span class="mono"></span>'
    assert rows['escape'] == '基座 <span class="mono">A&lt;B&gt;&amp;</span>'
    for name in ('same-id', 'slash', 'windows', 'space', 'empty', 'blank', 'escape'):
        assert '最初基座' not in rows[name]
        assert '本轮基座' not in rows[name]
    promoted = rows['promoted']
    assert promoted.startswith('最初基座 <span class="mono">Qwen/Qwen2.5-7B-Instruct</span><br/>')
    assert '本轮基座 <span class="mono">/data/out/checkpoint-100</span>' in promoted
    suffix = rows['suffix']
    assert '最初基座' in suffix and '本轮基座' in suffix
    assert '/ckpt/Qwen/Qwen2.5-7B-Instruct' in suffix


def test_displayed_times_use_shanghai():
    source = Path(__file__).resolve().parents[1].joinpath('static', 'time.js').read_text(encoding='utf-8')
    page = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert 'formatShanghai(' in page
    assert "timeZone: 'Asia/Shanghai'" in source
    script = source + """
    console.log(formatShanghai('2026-10-19T03:17:04Z'));
    console.log(formatShanghai('2026-10-19T16:00:00Z'));
    console.log(formatShanghai(''));
    """
    completed = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    lines = completed.stdout.splitlines()
    assert lines[0] == '2026-10-19 11:17:04'
    assert lines[1] == '2026-10-20 00:00:00'
    assert lines[2] == ''


def test_duration_column_uses_chinese_units():
    source = Path(__file__).resolve().parents[1].joinpath('static', 'time.js').read_text(encoding='utf-8')
    page = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert '<th>耗时</th>' in page
    assert 'formatDurationZh(' in page
    assert 'data-duration-start' in page
    script = source + """
    const start = '2026-10-19T03:00:00Z';
    console.log(formatDurationZh(start, '2026-10-19T06:05:04Z'));
    console.log(formatDurationZh(start, '2026-10-19T03:05:04Z'));
    console.log(formatDurationZh(start, '2026-10-19T03:00:12Z'));
    console.log(formatDurationZh(start, '2026-10-19T06:00:04Z'));
    console.log(formatDurationZh(start, '2026-10-19T04:05:00Z'));
    console.log(formatDurationZh(start, start));
    console.log(formatDurationZh('', '2026-10-19T03:00:12Z'));
    console.log(formatDurationZh(null, null));
    console.log(formatDurationZh(start, '', Date.parse('2026-10-19T03:00:12Z')));
    """
    completed = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
    lines = completed.stdout.splitlines()
    assert lines == [
        '3 小时 5 分 4 秒',
        '5 分 4 秒',
        '12 秒',
        '3 小时 4 秒',
        '1 小时 5 分',
        '0 秒',
        '—',
        '—',
        '12 秒',
    ]
