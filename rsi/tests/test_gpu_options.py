# Copyright (c) ModelScope Contributors. All rights reserved.
from rsi.gpu_options import GPU_INFO, gpu_choices, gpu_default, parse_device_line
from rsi.llm_export_spec import export_layout
from rsi.llm_infer_spec import llm_infer_layout
from rsi.webui_stage_spec import stage_layout


def test_parse_nvidia_smi_line():
    parsed = parse_device_line('GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-abc)')
    assert parsed == {'index': '0', 'name': 'NVIDIA A100-SXM4-40GB'}
    assert parse_device_line('not a gpu') is None


def test_gpu_choices_with_device_names():
    devices = [
        'GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-a)',
        'GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-b)',
    ]
    choices = gpu_choices(devices)
    assert [item['value'] for item in choices] == ['0', '1', 'cpu']
    assert choices[0]['label'] == 'GPU 0 · NVIDIA A100-SXM4-40GB'
    assert choices[1]['label'] == 'GPU 1 · NVIDIA A100-SXM4-80GB'
    assert choices[2]['label'] == 'CPU'
    assert gpu_default(choices) == '0'


def test_gpu_choices_count_only_and_cpu_only():
    numbered = gpu_choices(device_count=2)
    assert numbered == [
        {'value': '0', 'label': 'GPU 0'},
        {'value': '1', 'label': 'GPU 1'},
        {'value': 'cpu', 'label': 'CPU'},
    ]
    forced = gpu_choices(device_count=0, ensure_gpu_zero=True)
    assert [item['value'] for item in forced] == ['0', 'cpu']
    cpu = gpu_choices(device_count=0, ensure_gpu_zero=False)
    assert cpu == [{'value': 'cpu', 'label': 'CPU'}]
    assert gpu_default(cpu) == 'cpu'


def test_layouts_surface_gpu_labels():
    devices = ['GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-x)']
    sft = stage_layout('sft', 1, devices=devices)['by_id']['gpu_id']
    assert sft['choices'][0] == {'value': '0', 'label': 'GPU 0 · NVIDIA GeForce RTX 4090'}
    assert sft['info'] == GPU_INFO
    infer = llm_infer_layout(1, devices=devices)['by_id']['gpu_id']
    assert infer['choices'][0]['label'].startswith('GPU 0 ·')
    assert infer['info'] == GPU_INFO
    exported = export_layout(1, tuple(devices))['by_id']['gpu_id']
    assert exported['choices'][0]['value'] == '0'
    assert '4090' in exported['choices'][0]['label']
