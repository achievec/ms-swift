# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from pathlib import Path

from fastapi.testclient import TestClient

from rsi.llm_infer_spec import (
    QWEN_DEFAULT_SYSTEM,
    chat_request_body,
    cuda_visible_devices,
    layout_ids,
    llm_infer_layout,
    template_default_system,
)
from rsi.serving import build_openai_app


ORDER = [
    'model', 'model_type', 'template', 'merge_lora', 'reset', 'infer_backend', 'max_new_tokens', 'temperature',
    'top_k', 'top_p', 'repetition_penalty', 'system', 'adapters', 'more_params', 'load_checkpoint', 'running_tasks',
    'refresh_tasks', 'kill_task', 'log', 'gpu_id', 'infer_model_type', 'port', 'chatbot',
    'prompt', 'image', 'video', 'audio', 'clear_history', 'submit',
]


def test_layout_matches_the_webui_llm_infer_tab():
    layout = llm_infer_layout(0)
    assert layout_ids(layout) == ORDER
    by_id = layout['by_id']
    assert by_id['model']['label'] == '模型id或路径(--model)'
    assert by_id['model']['info'] == '实际的模型id，如果是训练后的模型请填入checkpoint-xxx的目录'
    assert by_id['model_type']['label'] == '选择模型类型(--model_type)'
    assert by_id['model_type']['info'] == 'SWIFT已支持的模型类型'
    assert by_id['template']['label'] == '模型Prompt模板类型(--template)'
    assert by_id['template']['info'] == '选择匹配模型的Prompt模板'
    assert by_id['merge_lora']['label'] == '合并LoRA(--merge_lora)'
    assert by_id['merge_lora']['info'] == '仅在`tuner_type=lora`时可用'
    assert by_id['infer_backend']['label'] == '推理框架(--infer_backend)'
    assert by_id['infer_backend']['default'] == 'transformers'
    assert by_id['infer_backend']['choices'] == ['vllm', 'transformers', 'sglang', 'lmdeploy', 'pt']
    assert by_id['max_new_tokens']['label'] == '生成序列最大长度(--max_new_tokens)'
    assert by_id['max_new_tokens']['default'] == '2048'
    assert by_id['temperature']['label'] == '采样温度(--temperature)'
    assert by_id['temperature']['default'] == 0.3
    assert by_id['temperature']['minimum'] == 0.0
    assert by_id['temperature']['maximum'] == 10
    assert by_id['temperature']['step'] == 0.1
    assert by_id['top_k']['default'] == 20
    assert by_id['top_k']['step'] == 5
    assert by_id['top_p']['label'] == 'Top-p(--top_p)'
    assert by_id['top_p']['default'] == 0.7
    assert by_id['repetition_penalty']['label'] == '重复惩罚(--repetition_penalty)'
    assert by_id['repetition_penalty']['default'] == 1.05
    assert by_id['system']['label'] == 'System字段(--system)'
    assert by_id['system']['info'] == 'System字段支持在加载模型后修改'
    assert by_id['adapters']['label'] == 'adapter id或路径(--adapters)'
    assert 'name1=/path1' in by_id['adapters']['info']
    assert by_id['more_params']['label'] == '更多参数'
    assert '(--more_params)' not in by_id['more_params']['label']
    assert by_id['more_params']['info'] == '以json格式或--xxx xxx命令行格式填入'
    assert by_id['reset']['label'] == '恢复初始值'
    assert by_id['load_checkpoint']['label'] == '部署模型'
    assert by_id['gpu_id']['label'] == '选择可用GPU'
    assert '(--gpu_id)' not in by_id['gpu_id']['label']
    assert by_id['gpu_id']['info'] == '选择要用的 GPU（编号与型号）；CUDA 不可用时只能选 CPU'
    assert by_id['infer_model_type']['label'] == 'LoRA模块'
    assert 'default' in by_id['infer_model_type']['info']
    assert by_id['port']['label'] == '端口'
    assert by_id['port']['default'] == '8000'
    assert '(--port)' not in by_id['port']['label']
    assert by_id['prompt']['label'] == '请输入：'
    assert by_id['clear_history']['label'] == '清除对话信息'
    assert by_id['submit']['label'] == '🚀 发送'
    assert by_id['running_tasks']['label'] == '运行中部署'
    assert by_id['refresh_tasks']['label'] == '找回部署'
    assert 'show_log' not in by_id
    assert 'stop_show_log' not in by_id
    assert by_id['kill_task']['label'] == '停止推理'
    assert by_id['log']['label'] == '日志输出'
    assert by_id['log']['info'] == '日志会自动刷新。'
    assert '展示部署状态' not in by_id['log']['info']
    assert layout['alerts']['load_alert'] == '部署已开始，日志会自动显示。'
    assert by_id['chatbot']['label'] == 'Chatbot'
    assert layout['sections'][1]['label'] == '运行时'
    assert layout['sections'][1]['open'] is False
    assert layout['locked'] == ['model', 'model_type', 'template', 'adapters']
    for key in layout['locked']:
        assert by_id[key]['locked'] is True
    assert by_id['merge_lora']['locked'] is False
    assert layout['alerts']['generate_alert'] == '请先部署模型'
    assert layout['alerts']['loaded_alert'] == '模型加载完成'


def test_gpu_choices_follow_device_count():
    cpu = llm_infer_layout(0)['by_id']['gpu_id']
    assert cpu['choices'] == [{'value': 'cpu', 'label': 'CPU'}]
    assert cpu['default'] == 'cpu'
    assert cpu['multiselect'] is True
    gpu = llm_infer_layout(2)['by_id']['gpu_id']
    assert gpu['choices'] == [
        {'value': '0', 'label': 'GPU 0'},
        {'value': '1', 'label': 'GPU 1'},
        {'value': 'cpu', 'label': 'CPU'},
    ]
    assert gpu['default'] == '0'


def test_cuda_visible_devices_and_chat_body():
    assert cuda_visible_devices(None) is None
    assert cuda_visible_devices('') is None
    assert cuda_visible_devices('cpu') == ''
    assert cuda_visible_devices(['0', '1']) == '0,1'
    assert cuda_visible_devices('0 1') == '0,1'
    try:
        cuda_visible_devices(['0', 'cpu'])
    except ValueError as exc:
        assert 'CPU' in str(exc)
    else:
        raise AssertionError('cpu mixed with a GPU must fail')
    assert template_default_system('qwen2_5') == QWEN_DEFAULT_SYSTEM
    assert template_default_system('llama3') == ''
    body = chat_request_body(
        'served',
        [{'role': 'user', 'content': 'hi'}],
        temperature=0.3,
        top_k=20,
        top_p=0.7,
        repetition_penalty=1.05,
        max_new_tokens=2048,
        model='default',
        system=QWEN_DEFAULT_SYSTEM,
        stream=True,
    )
    assert body['model'] == 'default'
    assert body['stream'] is True
    assert body['max_tokens'] == 2048
    assert body['stop'] == ['Observation:']
    assert body['messages'][0] == {'role': 'system', 'content': QWEN_DEFAULT_SYSTEM}
    assert body['messages'][1]['content'] == 'hi'
    plain = chat_request_body('served', [{'role': 'user', 'content': 'hi'}])
    assert plain['model'] == 'served'
    assert plain['messages'][0]['role'] == 'user'


def test_mock_serve_streams_one_delta():
    app = build_openai_app('demo', '', 'scene', '/ckpt')
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json={
            'model': 'demo',
            'messages': [{'role': 'user', 'content': '你好'}],
            'stream': True,
        })
        assert response.status_code == 200
        assert 'text/event-stream' in response.headers['content-type']
        assert '你好' in response.text
        assert response.text.strip().endswith('data: [DONE]')
        payload = json.loads(response.text.split('data: ', 1)[1].split('\n', 1)[0])
        assert payload['choices'][0]['delta']['content']


def test_send_button_tracks_prompt_and_media():
    root = Path(__file__).resolve().parents[1] / 'static'
    page = (root / 'app.js').read_text(encoding='utf-8')
    css = (root / 'app.css').read_text(encoding='utf-8')
    assert 'function chatCanSend()' in page
    assert 'function syncSendButton()' in page
    assert "input.value.trim()" in page
    assert 'files.image || files.video || files.audio' in page
    assert "target.id === 'chat-input'" in page
    assert 'button.disabled = !ready' in page
    assert '.infer-row.actions button.primary { background: #2563eb;' in css
    assert 'inferLogVisible' not in page
    assert 'show_log' not in page
    assert 'stop_show_log' not in page
    assert "row[0].id === 'log' && !running" in page
    assert 'function inferDeployRunning()' in page
    assert 'function inferLabelParts(' in page
    assert 'class="lab"' in page
    assert 'class="flag"' in page
    assert 'textarea class="path"' in page
    assert '--caption:' in css
    assert 'min-height: 84px' not in css
    assert '.infer-row.infer-grid > .infer-sides' in css
