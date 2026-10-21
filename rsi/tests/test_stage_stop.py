# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SWIFT_STAGES = ('sft', 'sft_eval', 'rl', 'rl_eval')
SFT_PARAMS = {'source': 'bundled', 'bundled': 'itops_mix'}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    app = create_app(tmp_path / 'rsi-home')
    with TestClient(app) as test_client:
        yield test_client


def _create(api) -> str:
    created = api.post('/api/runs', json={
        'name': '停止阶段',
        'scenario': '训练中途停止',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    assert created.status_code == 200, created.text
    return created.json()['run']['id']


def _run(api, run_id, stage, params=None):
    response = api.post(f'/api/runs/{run_id}/stages/{stage}/run', json={'params': params or {}})
    assert response.status_code == 200, response.text
    return response.json()


def _prep(api, run_id, stage):
    order = {'sft': [], 'sft_eval': ['sft'], 'rl': ['sft', 'sft_eval'], 'rl_eval': ['sft', 'sft_eval', 'rl']}
    for earlier in order[stage]:
        params = SFT_PARAMS if earlier == 'sft' else {}
        body = _run(api, run_id, earlier, params)
        assert body['status'] == 'succeeded', body


def _install_blocking(pipeline, script: Path):
    from rsi.backends import BackendError
    from rsi.logtail import run_and_tee

    def execute(stage, params, context):
        workdir = Path(context['workdir'])
        workdir.mkdir(parents=True, exist_ok=True)
        log_path = workdir / 'swift.log'
        with log_path.open('ab') as handle:
            code = run_and_tee(
                [sys.executable, str(script)],
                None,
                handle,
                f'[{stage}]',
                echo=False,
                on_start=context.get('on_proc'),
                new_session=True,
            )
        if code != 0:
            raise BackendError(f'swift 退出码 {code}', logs=f'exit {code}')
        return {'ok': True, 'logs': 'done', 'command': f'swift {stage}', 'metrics': {}}

    pipeline.backend.name = 'swift'
    pipeline.backend.execute = execute


def _wait_pid(path: Path, timeout: float = 5.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.is_file() and path.read_text(encoding='utf-8').strip():
            return int(path.read_text(encoding='utf-8').strip())
        time.sleep(0.02)
    raise AssertionError(f'pid file was not written: {path}')


def _assert_dead(pid: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f'process {pid} is still running')


def test_stop_rejects_idle_and_instant_stages(client):
    api = client
    run_id = _create(api)
    idle = api.post(f'/api/runs/{run_id}/stages/sft/stop', json={})
    assert idle.status_code == 400
    assert idle.json()['detail'] == '没有正在运行的阶段'
    for stage in ('sft_eval', 'rl', 'rl_eval', 'promote', 'next_round'):
        denied = api.post(f'/api/runs/{run_id}/stages/{stage}/stop', json={})
        assert denied.status_code == 400, denied.text
        if stage in SWIFT_STAGES:
            assert denied.json()['detail'] == '没有正在运行的阶段'
        else:
            assert denied.json()['detail'] == '这一阶段不能停止'


@pytest.mark.parametrize('stage', SWIFT_STAGES)
def test_stop_running_swift_stage_kills_process_group(client, tmp_path, stage):
    api = client
    run_id = _create(api)
    _prep(api, run_id, stage)
    pid_path = tmp_path / f'{stage}-child.pid'
    script = tmp_path / f'{stage}-sleep.py'
    script.write_text(
        textwrap.dedent(f"""\
            import subprocess, sys, time
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            with open({str(pid_path)!r}, "w", encoding="utf-8") as handle:
                handle.write(str(child.pid))
            time.sleep(60)
        """),
        encoding='utf-8')
    pipeline = api.app.state.pipeline
    _install_blocking(pipeline, script)
    params = SFT_PARAMS if stage == 'sft' else {}
    started = _run(api, run_id, stage, params)
    assert started['status'] == 'running'
    child_pid = _wait_pid(pid_path)
    assert pipeline._stages.running(run_id)

    stopped = api.post(f'/api/runs/{run_id}/stages/{stage}/stop', json={})
    assert stopped.status_code == 200, stopped.text
    form = stopped.json()['view']['forms'][stage]
    assert form['status'] == 'failed'
    assert form['result']['stopped'] is True
    assert '本阶段已停止' in (form['logs'] or '')
    _assert_dead(child_pid)
    deadline = time.time() + 5
    while pipeline._stages.running(run_id) and time.time() < deadline:
        time.sleep(0.05)
    assert pipeline._stages.running(run_id) is False

    again = api.post(f'/api/runs/{run_id}/stages/{stage}/stop', json={})
    assert again.status_code == 400
    assert again.json()['detail'] == '没有正在运行的阶段'


def test_stage_form_stop_button_only_when_running():
    script = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    body = script[script.index('function renderStage('):script.index('function flowEdgeGeometry(')]
    assert "const stop = form.status === 'running'" in body
    assert "data-action=\"stage-stop\">停止</button>" in body
    assert ': \'\';' in body
    assert '${stop}' in body
    assert '终止' not in body
    assert "stages/${stage}/stop" in script
    assert "dataset.action === 'stage-stop'" in script
