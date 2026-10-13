# Copyright (c) ModelScope Contributors. All rights reserved.
import re

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


def test_schema_model_labels_follow_webui():
    from rsi.schemas import public_schema
    fields = {item['key']: item for item in public_schema()['fields']['sft']}
    assert fields['model']['label'] == '模型id或路径'
    assert fields['model']['arg'] == '--model'
    assert fields['model']['help'] == '实际的模型id'
    assert fields['model']['type'] == 'combo'
    assert fields['model_type']['label'] == '模型类型'
    assert fields['model_type']['help'] == 'SWIFT已支持的模型类型'
    assert fields['template']['label'] == '模型Prompt模板类型'
    assert fields['template']['help'] == '选择匹配模型的Prompt模板'
    assert fields['model']['group'] == '模型设置'
    eval_fields = {item['key']: item for item in public_schema()['fields']['sft_eval']}
    assert eval_fields['model_type']['label'] == '选择模型类型'
    assert 'checkpoint-xxx' in eval_fields['model']['help']


def test_create_fills_model_type_on_the_sft_command(tmp_path, monkeypatch):
    monkeypatch.delenv('RSI_BACKEND', raising=False)
    from rsi.app import create_app
    app = create_app(tmp_path / 'home')
    with TestClient(app) as api:
        created = api.post('/api/runs', json={
            'name': 'IT 运维助手',
            'scenario': '企业内部 IT 工单分诊与处理建议',
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
        })
        assert created.status_code == 200, created.text
        run_id = created.json()['run']['id']
        data = api.post(f'/api/runs/{run_id}/stages/data/run', json={'params': {'source': 'bundled', 'bundled': 'itops_mix'}})
        assert data.status_code == 200, data.text
        preview = api.post(f'/api/runs/{run_id}/stages/sft/preview', json={'params': {}})
        assert preview.status_code == 200, preview.text
        command = preview.json()['command']
        assert re.search(r'--model_type qwen2(\s|\\|$)', command)
        assert '--template qwen2_5' in command
        assert 'Qwen/Qwen2.5-7B-Instruct' in command


def test_default_port_is_webui_7860(monkeypatch):
    monkeypatch.delenv('WEBUI_PORT', raising=False)
    monkeypatch.delenv('RSI_PORT', raising=False)
    monkeypatch.delenv('WEBUI_SERVER', raising=False)
    monkeypatch.delenv('RSI_HOST', raising=False)
    from rsi.__main__ import resolve_server
    host, port = resolve_server()
    assert host == '127.0.0.1'
    assert port == 7860
    monkeypatch.setenv('WEBUI_PORT', '7861')
    assert resolve_server()[1] == 7861
