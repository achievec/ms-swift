# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rsi.llm_export_spec import export_layout

BASE = 'Qwen/Qwen2.5-7B-Instruct'
ARGS_BASE = '/data/models/Qwen2.5-7B-Instruct'

ORDER = [
    'model', 'model_type', 'template', 'adapters', 'merge_lora', 'device_map', 'quant_bits', 'quant_method',
    'quant_n_samples', 'max_length', 'output_dir', 'dataset', 'running_tasks', 'refresh_tasks', 'kill_task', 'log',
    'more_params', 'export_preview', 'export', 'gpu_id',
]


def _ids(layout):
    found = []
    for section in layout['sections']:
        rows = section.get('rows') or []
        for row in rows:
            found.extend(item['id'] for item in row)
    return found


def test_layout_matches_the_webui_llm_export_tab():
    layout = export_layout(0)
    assert _ids(layout) == ORDER
    by_id = layout['by_id']
    assert by_id['model']['label'] == '模型id或路径(--model)'
    assert by_id['model_type']['label'] == '选择模型类型(--model_type)'
    assert by_id['template']['label'] == '模型Prompt模板类型(--template)'
    assert by_id['adapters']['label'] == 'adapter id或路径(--adapters)'
    assert 'name1=/path1' in by_id['adapters']['info']
    assert by_id['merge_lora']['label'] == '合并LoRA(--merge_lora)'
    assert 'checkpoint同级目录' in by_id['merge_lora']['info']
    assert by_id['device_map']['label'] == '合并LoRA使用的device_map(--device_map)'
    assert by_id['quant_bits']['label'] == '量化比特数(--quant_bits)'
    assert by_id['quant_bits']['choices'][0] == ''
    assert '4' in by_id['quant_bits']['choices']
    assert 'float8' in by_id['quant_bits']['choices']
    assert by_id['quant_method']['choices'][0] == ''
    assert by_id['quant_method']['choices'][1:] == ['awq', 'gptq', 'bnb', 'fp8', 'gptq_v2']
    assert by_id['quant_n_samples']['label'] == '量化集采样数(--quant_n_samples)'
    assert by_id['quant_n_samples']['default'] == 256
    assert by_id['max_length']['label'] == '量化集的max-length(--max_length)'
    assert by_id['output_dir']['label'] == '输出路径(--output_dir)'
    assert by_id['output_dir']['locked'] is False
    assert by_id['dataset']['label'] == '校准数据集(--dataset)'
    assert by_id['dataset']['multiselect'] is True
    assert by_id['dataset']['allow_custom'] is True
    assert by_id['dataset']['choices_from'] == 'datasets'
    assert by_id['more_params']['label'] == '更多参数'
    assert '(--more_params)' not in by_id['more_params']['label']
    assert by_id['more_params']['lines'] == 4
    assert by_id['export']['label'] == '开始导出'
    assert by_id['export_preview']['label'] == '预览命令'
    assert by_id['gpu_id']['label'] == '选择可用GPU'
    assert by_id['gpu_id']['multiselect'] is True
    assert by_id['gpu_id']['info'] == '选择使用的GPU号，如CUDA不可用只能选择CPU'
    assert by_id['running_tasks']['label'] == '运行中导出任务'
    assert by_id['refresh_tasks']['label'] == '找回导出任务'
    assert by_id['kill_task']['label'] == '停止导出'
    assert by_id['log']['info'] == '日志会自动刷新。'
    assert 'show_log' not in by_id
    assert 'stop_show_log' not in by_id
    assert layout['locked'] == ['model', 'model_type', 'template']
    for key in layout['locked']:
        assert by_id[key]['locked'] is True
    assert by_id['adapters']['locked'] is False
    assert layout['sections'][1]['label'] == '运行时'
    assert layout['sections'][1]['kind'] == 'accordion'
    page = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
    assert '模型导出' in page
    assert 'llm_export' in page
    assert '填入本轮权重' in page
    assert '填入本轮晋级 checkpoint' in page


def test_gpu_choices_follow_device_count():
    cpu = export_layout(0)['by_id']['gpu_id']
    assert cpu['choices'] == ['0', 'cpu']
    assert cpu['default'] == '0'
    gpu = export_layout(2)['by_id']['gpu_id']
    assert gpu['choices'] == ['0', '1', 'cpu']


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    app = create_app(tmp_path / 'rsi-home')
    with TestClient(app) as test_client:
        yield test_client


def _create(api):
    created = api.post('/api/runs', json={
        'name': '导出',
        'scenario': '把本轮 LoRA 合并出来',
        'base_model': BASE,
        'model_type': 'qwen2',
        'template': 'qwen2_5',
    })
    assert created.status_code == 200, created.text
    return created.json()['run']['id']


def _run(api, run_id, stage, params=None):
    response = api.post(f'/api/runs/{run_id}/stages/{stage}/run', json={'params': params or {}})
    assert response.status_code == 200, response.text
    return response.json()


def test_export_fills_lora_and_ignores_model_edits(client):
    api = client
    run_id = _create(api)
    trained = _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'})
    checkpoint = Path(trained['result']['checkpoint'])
    (checkpoint / 'adapter_config.json').write_text('{"peft_type": "LORA"}', encoding='utf-8')
    args = json.loads((checkpoint / 'args.json').read_text(encoding='utf-8'))
    args['model'] = ARGS_BASE
    (checkpoint / 'args.json').write_text(json.dumps(args), encoding='utf-8')

    view = api.get(f'/api/runs/{run_id}').json()
    fill = view['export_fill']
    # Same as serve: LoRA export shows the resolved base from checkpoint args.json.
    assert fill['model'] == ARGS_BASE
    assert fill['model_type'] == 'qwen2'
    assert fill['template'] == 'qwen2_5'
    assert fill['adapters'] == str(checkpoint)
    assert fill['merge_lora'] is True
    assert fill['output_dir'].endswith(f'{checkpoint.name}-merged')
    assert view['export_form']['model'] == ARGS_BASE
    assert view['export_form']['adapters'] == str(checkpoint)
    assert view['export_job']['status'] == 'idle'

    preview = api.post(f'/api/runs/{run_id}/export/preview', json={'params': {
        'model': 'other/model',
        'model_type': 'llama',
        'template': 'llama3',
    }})
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body['params']['model'] == ARGS_BASE
    assert body['params']['model_type'] == 'qwen2'
    assert body['params']['template'] == 'qwen2_5'
    assert body['params']['adapters'] == str(checkpoint)
    command = body['command']
    assert 'swift export' in command
    assert f'--model {ARGS_BASE}' in command
    assert f'--adapters {checkpoint}' in command
    assert '--merge_lora true' in command
    assert '--model_type qwen2' in command
    assert '--template qwen2_5' in command
    assert 'other/model' not in command
    assert f'--model {BASE}' not in command
    assert '--more_params' not in command

    cleared = api.post(f'/api/runs/{run_id}/export/preview', json={'params': {
        'adapters': '',
        'merge_lora': False,
    }})
    assert cleared.status_code == 200, cleared.text
    assert '--adapters' not in cleared.json()['command']
    assert '--merge_lora' not in cleared.json()['command']
    # Adapters cleared; locked --model still comes from the LoRA args.json base.
    assert f'--model {ARGS_BASE}' in cleared.json()['command']

    started = api.post(f'/api/runs/{run_id}/export/run', json={'params': {}})
    assert started.status_code == 200, started.text
    done = started.json()
    assert done['status'] == 'succeeded'
    assert 'swift export' in done['log']
    assert Path(done['output_dir'], 'export_ok.txt').is_file()
    log = api.get(f'/api/runs/{run_id}/export/log')
    assert log.status_code == 200
    assert log.json()['running'] is False
    assert '模型导出已完成' in log.json()['log']
    again = api.get(f'/api/runs/{run_id}').json()
    assert again['export_job']['status'] == 'succeeded'
    assert again['export_form']['adapters'] == str(checkpoint)
    assert again['export_form']['model'] == ARGS_BASE
    stopped = api.post(f'/api/runs/{run_id}/export/stop', json={})
    assert stopped.status_code == 400


def test_rl_weights_win_and_a_full_checkpoint_becomes_model(client):
    api = client
    run_id = _create(api)
    _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'})
    _run(api, run_id, 'sft_eval', {})
    trained = _run(api, run_id, 'rl', {})
    checkpoint = Path(trained['result']['checkpoint'])
    (checkpoint / 'adapter_config.json').write_text('{}', encoding='utf-8')
    view = api.get(f'/api/runs/{run_id}').json()
    assert view['export_fill']['adapters'] == str(checkpoint)
    assert '/rl/' in view['export_fill']['adapters']

    (checkpoint / 'adapter_config.json').unlink()
    (checkpoint / 'config.json').write_text('{"model_type": "qwen2"}', encoding='utf-8')
    full = api.post(f'/api/runs/{run_id}/export/preview', json={'params': {}})
    assert full.status_code == 200, full.text
    command = full.json()['command']
    assert f'--model {checkpoint}' in command
    assert '--adapters' not in command
    assert full.json()['params']['model'] == BASE
    assert full.json()['params']['model_type'] == 'qwen2'


def test_export_prefills_hope_local_base_path(tmp_path, monkeypatch):
    """Export --model should show the hope-ready local dir, not the hub short id."""
    monkeypatch.setenv('RSI_MODEL_SOURCE', 'hope')
    monkeypatch.setenv('RSI_MODEL_ROOT', str(tmp_path / 'models'))
    monkeypatch.setenv('RSI_DATA_ROOT', str(tmp_path / 'datasets'))
    monkeypatch.setenv('RSI_FETCH_INLINE', '1')
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    model_dir = tmp_path / 'models' / 'huggingface.co' / 'Qwen' / 'Qwen2.5-7B-Instruct'
    model_dir.mkdir(parents=True)
    (model_dir / 'config.json').write_text('{}', encoding='utf-8')

    def refuse(*_args, **_kwargs):
        raise AssertionError('ready model must not call hope')

    from rsi.app import create_app
    app = create_app(tmp_path / 'rsi-home')
    app.state.pipeline.fetcher.runner = refuse
    with TestClient(app) as api:
        run_id = api.post('/api/runs', json={
            'name': 'export-hope',
            'scenario': '导出带上本地基座路径',
            'base_model': BASE,
            'model_type': 'qwen2',
            'template': 'qwen2_5',
        }).json()['run']['id']
        before = api.get(f'/api/runs/{run_id}').json()
        assert before['export_fill']['model'] == str(model_dir)
        assert before['export_form']['model'] == str(model_dir)

        trained = _run(api, run_id, 'sft', {'source': 'bundled', 'bundled': 'itops_mix'})
        checkpoint = Path(trained['result']['checkpoint'])
        (checkpoint / 'adapter_config.json').write_text('{"peft_type": "LORA"}', encoding='utf-8')
        after = api.get(f'/api/runs/{run_id}').json()
        assert after['export_fill']['model'] == str(model_dir)
        assert after['export_fill']['adapters'] == str(checkpoint)
        preview = api.post(f'/api/runs/{run_id}/export/preview', json={'params': {}})
        assert preview.status_code == 200, preview.text
        assert preview.json()['params']['model'] == str(model_dir)
        assert f'--model {model_dir}' in preview.json()['command']
        assert f'--adapters {checkpoint}' in preview.json()['command']
        assert f'--model {BASE}' not in preview.json()['command']
