# Copyright (c) ModelScope Contributors. All rights reserved.
import re

import pytest
from fastapi.testclient import TestClient


def test_qwen25_matches_webui_dropdown():
    from rsi.webui_catalog import load_catalog, match_model
    catalog = load_catalog()
    for model_id in (
            'Qwen/Qwen2.5-7B-Instruct',
            'Qwen/Qwen2.5-14B-Instruct',
            'Qwen/Qwen2.5-32B-Instruct',
            'Qwen/Qwen2.5-72B-Instruct',
    ):
        assert model_id in catalog['models']
    meta = match_model('Qwen/Qwen2.5-7B-Instruct')
    assert meta['model_type'] == 'qwen2'
    assert meta['template'] == 'qwen2_5'
    assert 'qwen2' in catalog['model_types']
    assert 'qwen2_5' in catalog['templates']
    assert 'AI-ModelScope/alpaca-gpt4-data-zh' in catalog['datasets']


def test_schema_labels_follow_webui():
    from rsi.schemas import public_schema
    from rsi.webui_labels import control_label
    fields = {item['key']: item for item in public_schema()['fields']['sft']}
    assert fields['model']['label'] == '模型id或路径'
    assert control_label(fields['model']['label'], fields['model']['arg']) == '模型id或路径(--model)'
    assert fields['model_type']['label'] == '模型类型'
    assert fields['model_type']['help'] == 'SWIFT已支持的模型类型'
    assert fields['template']['label'] == '模型Prompt模板类型'
    assert fields['learning_rate']['label'] == '学习率'
    assert fields['per_device_train_batch_size']['label'] == '训练batch size'
    assert 'zero2_offload' in fields['deepspeed']['choices']
    assert 'dora' not in fields['tuner_type']['choices']
    assert 'lora' in fields['tuner_type']['choices']


def test_more_params_accept_cli_and_json(tmp_path, monkeypatch):
    monkeypatch.delenv('RSI_BACKEND', raising=False)
    from rsi.app import create_app
    app = create_app(tmp_path / 'home')
    with TestClient(app) as api:
        created = api.post('/api/runs', json={
            'name': 'IT 运维助手',
            'scenario': '企业内部 IT 工单分诊与处理建议',
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
            'model_meta': {'model_type': 'qwen2', 'template': 'qwen2_5'},
        })
        assert created.status_code == 200, created.text
        run_id = created.json()['run']['id']
        preview = api.post(f'/api/runs/{run_id}/stages/sft/preview', json={'params': {}})
        assert preview.status_code == 200, preview.text
        command = preview.json()['command']
        assert re.search(r'--model_type qwen2(\s|\\)', command)
        assert 'qwen2_5' in command
        flagged = api.post(
            f'/api/runs/{run_id}/stages/sft/preview',
            json={'params': {'extra_yaml': '--learning_rate 2e-5 --gradient_checkpointing false'}},
        )
        assert flagged.status_code == 200, flagged.text
        flagged_command = flagged.json()['command']
        assert '2e-5' in flagged_command
        assert 'false' in flagged_command
        as_json = api.post(
            f'/api/runs/{run_id}/stages/sft/preview',
            json={'params': {'extra_yaml': '{"learning_rate": "3e-5"}'}},
        )
        assert as_json.status_code == 200, as_json.text
        assert '3e-5' in as_json.json()['command']


def test_gradio_tab_uses_webui_model_controls(tmp_path, monkeypatch):
    gradio = pytest.importorskip('gradio')
    monkeypatch.setenv('RSI_HOME', str(tmp_path / 'home'))
    import rsi.gradio_app as ui
    ui._pipeline = None
    with gradio.Blocks() as demo:
        ui.build_rsi_tab(demo)
    labels = []
    values = []

    def walk(block):
        for child in getattr(block, 'children', []) or []:
            label = getattr(child, 'label', None)
            if label:
                labels.append(label)
            value = getattr(child, 'value', None)
            if isinstance(value, str):
                values.append(value)
            walk(child)

    walk(demo)
    assert '模型id或路径(--model)' in labels
    assert '模型类型(--model_type)' in labels
    assert '模型Prompt模板类型(--template)' in labels
    assert 'System字段(--system)' in labels
    assert 'Qwen/Qwen2.5-7B-Instruct' in values
    assert 'qwen2' in values
    assert 'qwen2_5' in values


def test_default_port_is_webui_port(monkeypatch):
    monkeypatch.delenv('WEBUI_PORT', raising=False)
    monkeypatch.delenv('RSI_PORT', raising=False)
    monkeypatch.delenv('WEBUI_SERVER', raising=False)
    monkeypatch.delenv('RSI_HOST', raising=False)
    from rsi.gradio_app import resolve_server
    host, port = resolve_server()
    assert host == '127.0.0.1'
    assert port == 7860
    assert port != 8765
    monkeypatch.setenv('WEBUI_PORT', '7870')
    assert resolve_server()[1] == 7870
