# Copyright (c) ModelScope Contributors. All rights reserved.
from fastapi.testclient import TestClient


def test_schema_uses_webui_choice_lists(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    monkeypatch.setenv('RSI_HOME', str(tmp_path / 'home'))
    from rsi.app import create_app
    from rsi.webui_catalog import match_model

    matched = match_model('Qwen/Qwen2.5-7B-Instruct')
    assert matched == {'model_type': 'qwen2', 'template': 'qwen2_5'}

    with TestClient(create_app(tmp_path / 'home')) as api:
        schema = api.get('/api/schema').json()
        catalog = schema['catalog']
        assert len(catalog['models']) > 100
        assert 'Qwen/Qwen2.5-7B-Instruct' in catalog['models']
        assert catalog['by_model']['Qwen/Qwen2.5-7B-Instruct']['model_type'] == 'qwen2'
        assert catalog['by_model']['Qwen/Qwen2.5-7B-Instruct']['template'] == 'qwen2_5'
        assert 'qwen2' in catalog['model_types']
        assert 'qwen2_5' in catalog['templates']
        assert 'AI-ModelScope/alpaca-gpt4-data-zh' in catalog['datasets']
        assert 'modelscope/gsm8k' in catalog['datasets']
        assert 'general_qa' in catalog['eval_datasets']
        assert 'gsm8k' in catalog['eval_datasets']

        def by_key(stage):
            return {field['key']: field for field in schema['fields'][stage]}

        sft = by_key('sft')
        assert sft['model']['type'] == 'combo' and sft['model']['choices_from'] == 'models'
        assert sft['model']['label'] == '模型id或路径(--model)'
        assert sft['model_type']['choices_from'] == 'model_types'
        assert sft['template']['choices_from'] == 'templates'
        assert sft['tuner_type']['type'] == 'select'
        assert 'lora' in sft['tuner_type']['choices']
        assert 'bone' in sft['tuner_type']['choices']
        assert 'ia3' in sft['tuner_type']['choices']
        assert 'dora' not in sft['tuner_type']['choices']
        assert sft['torch_dtype']['choices'] == ['bfloat16', 'float16', 'float32']
        assert 'flash_attention_2' in sft['attn_impl']['choices']
        assert 'zero3_offload' in sft['deepspeed']['choices']

        data = by_key('data')
        assert data['dataset']['type'] == 'combo'
        assert data['dataset']['choices_from'] == 'datasets'
        assert data['dataset']['label'] == '数据集名称(--dataset)'

        rl = by_key('rl')
        assert rl['reward_funcs']['type'] == 'multiselect'
        assert 'accuracy' in rl['reward_funcs']['choices']
        assert 'itops_checklist' in rl['reward_funcs']['choices']
        assert rl['ref_model']['type'] == 'combo'
        assert rl['rlhf_type']['choices'][0] == 'dpo'
        assert 'grpo' in rl['rlhf_type']['choices']
        assert rl['sleep_level']['choices'] == ['0', '1']

        created = api.post('/api/runs', json={
            'name': 'IT 运维助手',
            'scenario': '企业内部 IT 工单分诊',
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
        }).json()
        assert created['run']['model_type'] == 'qwen2'
        assert created['run']['template'] == 'qwen2_5'
        values = created['forms']['sft']['values']
        assert values['model'] == 'Qwen/Qwen2.5-7B-Instruct'
        assert values['model_type'] == 'qwen2'
        assert values['template'] == 'qwen2_5'
