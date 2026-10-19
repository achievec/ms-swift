# Copyright (c) ModelScope Contributors. All rights reserved.
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
    assert _run(api, run_id, 'data', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200
    assert _run(api, run_id, 'sft', {}).status_code == 200
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
    labels = {edge['label'] for edge in view['flow']['edges'] if edge['kind'] == 'down'}
    assert labels == {'下一轮数据', '完成'}
    assert all(edge['from'] == 'sft_eval' for edge in view['flow']['edges'] if edge['kind'] == 'down')
    assert any(edge['from'] == 'sft_eval' and edge['to'] == 'rl' and edge['live'] and edge['label'] == 'RL'
               for edge in view['flow']['edges'])

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
    assert 'finish' not in node
    assert not any(edge['to'] == 'rl' for edge in view['flow']['edges'])
    assert any(edge['from'] == 'sft_eval' and edge['to'] == 'next_data' and edge['label'] == '不通过'
               for edge in view['flow']['edges'])
    blocked = _run(api, run_id, 'rl', {})
    assert blocked.status_code == 400
    assert blocked.json()['detail'] == 'SFT 评测未通过，不能进入在线 RL'
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
    assert view['flow']['actions'] == []
    node = {item['id']: item for item in view['flow']['nodes']}
    assert node['finish']['status'] == 'succeeded'
    assert 'stop' not in node
    assert 'next_data' not in node
    assert any(item['kind'] == 'sft' for item in view['checkpoints'])
    assert _run(api, run_id, 'rl', {}).status_code == 400

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
    assert 'finish' not in failed_nodes
    assert 'stop' not in failed_nodes
    assert any(edge['from'] == 'rl_eval' and edge['to'] == 'next_data' and edge['label'] == '不通过'
               for edge in failed_view['flow']['edges'])
    assert not any(edge['kind'] == 'down' and edge['from'] == 'sft_eval' for edge in failed_view['flow']['edges'])
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
    assert _ids(rl_done.json()['view']) == ['next', 'finish']
    passed_edges = [edge for edge in rl_done.json()['view']['flow']['edges'] if edge['kind'] == 'down']
    assert {edge['label'] for edge in passed_edges} == {'下一轮数据', '完成'}
    assert {edge['from'] for edge in passed_edges} == {'rl_eval'}


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
