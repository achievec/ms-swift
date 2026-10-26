# Copyright (c) ModelScope Contributors. All rights reserved.
"""Build Native / OpenCompass / VLMEval task configs without loading a model."""
import pytest

pytest.importorskip('evalscope')

from swift.pipelines.eval.eval import SwiftEval, openai_compatible_eval_type


class _Args:
    model_suffix = 'qwen2.5'
    api_key = 'EMPTY'
    eval_limit = 4
    eval_num_proc = 2
    use_chat_template = True
    eval_dataset_args = {'general_qa': {'local_path': '/tmp/qa', 'subset_list': ['itops']}}
    eval_generation_config = {'temperature': 0, 'max_tokens': 16}
    extra_eval_args = {}
    eval_output_dir = '/tmp/rsi-native-eval'


def _evaler():
    evaler = SwiftEval.__new__(SwiftEval)
    evaler.args = _Args()
    return evaler


def _secret(value):
    if hasattr(value, 'get_secret_value'):
        return value.get_secret_value()
    return value


def test_native_task_cfg_uses_openai_api_and_keeps_task_fields():
    cfg = _evaler().get_native_task_cfg(['general_qa'], 'http://127.0.0.1:9/v1')
    assert cfg.eval_type == 'openai_api'
    assert cfg.model == 'qwen2.5'
    assert cfg.api_url == 'http://127.0.0.1:9/v1'
    assert _secret(cfg.api_key) == 'EMPTY'
    assert cfg.datasets == ['general_qa']
    assert cfg.limit == 4
    assert cfg.eval_batch_size == 2
    assert cfg.dataset_args['general_qa']['subset_list'] == ['itops']
    assert cfg.work_dir == '/tmp/rsi-native-eval/native'
    temperature = getattr(cfg.generation_config, 'temperature', None)
    if temperature is None and isinstance(cfg.generation_config, dict):
        temperature = cfg.generation_config.get('temperature')
    assert temperature == 0


def test_openai_compatible_eval_type_covers_service_and_openai_api_names():

    class OnlyService:
        SERVICE = 'openai_api'

    class OnlyOpenAI:
        OPENAI_API = 'openai_api'

    class Neither:
        pass

    assert openai_compatible_eval_type(OnlyService) == 'openai_api'
    assert openai_compatible_eval_type(OnlyOpenAI) == 'openai_api'
    assert openai_compatible_eval_type(Neither) == 'openai_api'
    assert openai_compatible_eval_type() == 'openai_api'


def test_opencompass_and_vlmeval_task_cfg_still_build():
    evaler = _evaler()
    opencompass = evaler.get_opencompass_task_cfg(['gsm8k'], 'http://127.0.0.1:9/v1')
    assert opencompass.eval_backend == 'OpenCompass'
    assert opencompass.work_dir == '/tmp/rsi-native-eval/opencompass'
    assert opencompass.eval_config['datasets'] == ['gsm8k']
    assert opencompass.eval_config['models'][0]['openai_api_base'].endswith('/chat/completions')
    assert opencompass.eval_config['limit'] == 4

    vlm = evaler.get_vlmeval_task_cfg(['MMBench'], 'http://127.0.0.1:9/v1')
    assert vlm.eval_backend == 'VLMEvalKit'
    assert vlm.work_dir == '/tmp/rsi-native-eval/vlmeval'
    assert vlm.eval_config['data'] == ['MMBench']
    assert vlm.eval_config['model'][0]['api_base'].endswith('/chat/completions')
    assert vlm.eval_config['nproc'] == 2
