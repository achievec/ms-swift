# Copyright (c) ModelScope Contributors. All rights reserved.
"""Eval soft-success when swift exits non-zero after writing a report."""
import json
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rsi.eval_report import eval_artifacts_usable

BASE = 'Qwen/Qwen2.5-7B-Instruct'


def _write_report(eval_dir: Path, bleu_1=0.21, rouge_l_f=0.44) -> Path:
    reports = eval_dir / 'native' / '20261021_173847' / 'reports'
    dataset = reports / 'model'
    dataset.mkdir(parents=True)
    payload = {
        'dataset_name': 'general_qa',
        'model_name': 'model',
        'metrics': [
            {'name': 'bleu-1', 'score': bleu_1, 'categories': []},
            {'name': 'Rouge-L-F', 'score': rouge_l_f, 'categories': []},
        ],
    }
    (dataset / 'general_qa.json').write_text(json.dumps(payload), encoding='utf-8')
    html = reports / 'report.html'
    html.write_text('<html><body>eval report</body></html>', encoding='utf-8')
    return html


def test_eval_artifacts_usable(tmp_path):
    assert eval_artifacts_usable(tmp_path) == {}
    html = _write_report(tmp_path)
    found = eval_artifacts_usable(tmp_path)
    assert found['report_html'] == str(html.resolve())
    assert found['scores']['bleu-1'] == 0.21


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    app = create_app(tmp_path / 'rsi-home')
    with TestClient(app) as test_client:
        yield test_client


def _create(api):
    return api.post('/api/runs', json={
        'name': '软成功评测',
        'scenario': 'exit 1 但有报告',
        'base_model': BASE,
        'model_type': 'qwen2',
        'template': 'qwen2_5',
    }).json()['run']['id']


def _run(api, run_id, stage, params=None):
    return api.post(f'/api/runs/{run_id}/stages/{stage}/run', json={'params': params or {}})


def test_msswift_eval_exit_1_with_report_is_soft_success(tmp_path, monkeypatch):
    script = tmp_path / 'fake-swift'
    script.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import sys
        print('Finished evaluation')
        raise SystemExit(1)
    """), encoding='utf-8')
    script.chmod(0o755)
    monkeypatch.setenv('RSI_SWIFT_BIN', str(script))

    eval_dir = tmp_path / 'runs' / 'r1' / 'round_1' / 'rl_eval' / 'eval_output'
    eval_dir.mkdir(parents=True)
    html = _write_report(eval_dir)

    from rsi.backends import BackendError, MsSwiftBackend

    backend = MsSwiftBackend()
    outcome = backend.execute('rl_eval', {
        'model': BASE,
        'eval_dataset': 'general_qa',
        'eval_output_dir': str(eval_dir),
        'adapters': str(tmp_path / 'adapter'),
    }, {
        'workdir': str(eval_dir),
        'run_id': 'r1',
        'round_index': 1,
    })
    assert outcome['ok'] is True
    assert outcome['soft_exit_code'] == 1
    assert outcome['metrics']['report_html'] == str(html.resolve())
    assert outcome['metrics']['scores']['bleu-1'] == 0.21
    assert '退出码 1' in outcome['logs']
    assert '按成功处理' in outcome['logs']

    empty = tmp_path / 'empty-eval'
    empty.mkdir()
    with pytest.raises(BackendError, match='退出码 1'):
        backend.execute('rl_eval', {
            'model': BASE,
            'eval_dataset': 'general_qa',
            'eval_output_dir': str(empty),
        }, {'workdir': str(empty), 'run_id': 'r1', 'round_index': 1})


def test_eval_base_continues_after_adapter_soft_exit(client):
    """Pipeline keeps adapter metrics and still schedules 基座 when soft-exit succeeds."""
    api = client
    run_id = _create(api)
    assert _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200

    pipeline = api.app.state.pipeline
    real_execute = pipeline.backend.execute
    calls = []

    def execute(stage, params, context):
        calls.append({
            'adapters': str(params.get('adapters') or ''),
            'eval_output_dir': str(params.get('eval_output_dir') or ''),
        })
        # First (adapter) call: mimic MsSwiftBackend soft-exit — succeed with
        # soft_exit_code so pipeline continues to base.
        produced = real_execute(stage, params, context)
        if len(calls) == 1:
            produced = dict(produced)
            produced['soft_exit_code'] = 1
            produced['logs'] = (produced.get('logs') or '') + '\n[rsi] soft exit 1\n'
        return produced

    pipeline.backend.execute = execute
    evaluated = _run(api, run_id, 'sft_eval', {})
    assert evaluated.status_code == 200, evaluated.text
    body = evaluated.json()
    assert body['status'] == 'succeeded'
    assert len(calls) == 2
    assert calls[0]['adapters']
    assert not calls[1]['adapters']
    assert str(calls[1]['eval_output_dir']).endswith('/eval_output/base')
    result = body['result']
    compare = result['compare']
    assert [row['role'] for row in compare] == ['base', 'sft']
    base_report = api.get(f'/api/runs/{run_id}/stages/sft_eval/report?kind=base')
    assert base_report.status_code == 200, base_report.text
    sft_report = api.get(f'/api/runs/{run_id}/stages/sft_eval/report?kind=sft')
    assert sft_report.status_code == 200, sft_report.text


def test_base_hard_fail_keeps_adapter_success(client):
    api = client
    run_id = _create(api)
    assert _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'}).status_code == 200

    from rsi.backends import BackendError

    pipeline = api.app.state.pipeline
    real_execute = pipeline.backend.execute
    calls = []

    def execute(stage, params, context):
        calls.append(str(params.get('adapters') or ''))
        if len(calls) == 1:
            return real_execute(stage, params, context)
        raise BackendError('swift 退出码 1', logs='base boom')

    pipeline.backend.execute = execute
    evaluated = _run(api, run_id, 'sft_eval', {})
    assert evaluated.status_code == 200, evaluated.text
    body = evaluated.json()
    assert body['status'] == 'succeeded'
    result = body['result']
    assert result.get('base_error') == 'swift 退出码 1'
    assert [row['role'] for row in result['compare']] == ['sft']
    assert result['metrics'].get('accuracy') is not None
    view = api.get(f'/api/runs/{run_id}').json()
    form = view['forms']['sft_eval']
    assert form['result']['base_error'] == 'swift 退出码 1'
    assert form['result']['metrics']['accuracy'] is not None
    assert '基座评测失败' in (form.get('logs') or '')


def test_ui_mentions_base_error_note():
    script = Path(__file__).resolve().parents[1] / 'static' / 'app.js'
    text = script.read_text(encoding='utf-8')
    assert 'base_error' in text
    assert '基座评测失败' in text
