# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from pathlib import Path

from fastapi.testclient import TestClient

DOC_MARKERS = (
    ('Native', '1. Native', '2. OpenCompass'),
    ('OpenCompass', '2. OpenCompass', '3. VLMEvalKit'),
    ('VLMEvalKit', '3. VLMEvalKit', '## 环境准备'),
)


def _documented_eval_datasets():
    text = Path(__file__).resolve().parents[2].joinpath('docs/source/Instruction/Evaluation.md').read_text(encoding='utf-8')
    found = {}
    for name, start, end in DOC_MARKERS:
        chunk = text[text.find(start):text.find(end)]
        found[name] = re.findall(r"'([^']+)'", chunk)
    return found


def test_eval_datasets_are_the_documented_lists():
    from rsi import webui_catalog
    source = Path(webui_catalog.__file__).read_text(encoding='utf-8')
    assert 'evalscope' not in source
    listed = webui_catalog.eval_datasets_by_backend()
    assert listed == _documented_eval_datasets()
    assert listed['Native'] == webui_catalog.eval_dataset_choices()
    assert 'itops-heldout' not in listed['Native']
    assert 'general_qa' not in listed['OpenCompass']
    assert 'Video-MME' == listed['VLMEvalKit'][-1]


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
        assert catalog['datasets'][:14] == [
            'AI-ModelScope/alpaca-gpt4-data-zh',
            'AI-ModelScope/LongAlpaca-12k',
            'AI-ModelScope/ruozhiba',
            'AI-ModelScope/math-trn-format',
            'iic/ms_bench',
            'damo/MSAgent-Bench',
            'lvjianjin/AdvertiseGen',
            'AI-ModelScope/firefly-train-1.1M',
            'modelscope/clue',
            'DAMO_NLP/jd',
            'AI-ModelScope/synthetic_text_to_sql',
            'shenweizhou/alpha-umi-toolbench-processed-v2',
            'AI-ModelScope/blossom-math-v2',
            'AI-ModelScope/sql-create-context',
        ]
        assert 'modelscope/gsm8k' in catalog['datasets']
        assert 'openai/gsm8k' not in catalog['datasets'][:20]
        assert catalog['eval_datasets'] == catalog['eval_datasets_by_backend']['Native']
        assert catalog['eval_datasets_by_backend']['OpenCompass'][0] == 'obqa'
        assert 'AX_b' in catalog['eval_datasets_by_backend']['OpenCompass']
        assert catalog['eval_datasets_by_backend']['VLMEvalKit'][0] == 'COCO_VAL'
        assert 'MMBench-Video' in catalog['eval_datasets_by_backend']['VLMEvalKit']

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
        assert sft['torch_dtype']['choices'] == ['', 'bfloat16', 'float16', 'float32']
        assert 'flash_attention_2' in sft['attn_impl']['choices']
        assert sft['deepspeed']['choices'][0] == ''
        assert 'zero3_offload' in sft['deepspeed']['choices']
        assert schema['stage_layouts']['sft']['by_id']['deepspeed']['min_gpus'] == 2
        assert schema['stage_layouts']['sft']['by_id']['use_ddp']['min_gpus'] == 2
        assert schema['stage_layouts']['rl']['by_id']['sequence_parallel_size']['min_gpus'] == 2

        panel = schema['stage_layouts']['sft']['dataset_fields']
        data = {field['key']: field for field in panel}
        assert [field['key'] for field in panel] == [
            'source', 'dataset', 'split_dataset_ratio', 'max_length', 'padding_free',
        ]
        assert 'data' not in schema['fields']
        assert schema['stage_order'][0] == 'sft'
        assert data['dataset']['type'] == 'multicombo'
        assert data['dataset']['choices_from'] == 'datasets'
        assert data['dataset']['label'] == '数据集名称(--dataset)'
        assert data['dataset']['info'] == '选择训练的数据集，支持复选/本地路径'
        assert [item['label'] for item in data['source']['choices']] == [
            '内置示例', '公开数据集', '上传文件', '服务器本地路径',
        ]
        assert [item['value'] for item in data['source']['choices']] == ['bundled', 'public', 'upload', 'path']
        assert 'bundled' not in data
        bundled_ids = [item['id'] for item in schema['bundled']]
        assert bundled_ids[0] == 'helpdesk_train'
        assert 'itops_mix' in bundled_ids
        assert 'helpdesk_eval' not in bundled_ids
        examples = Path(__file__).resolve().parents[1] / 'examples'
        for item in schema['bundled']:
            assert (examples / f"{item['id']}.jsonl").is_file()
        assert 'dataset_num_proc' not in data
        assert 'dataloader_num_workers' not in data
        assert 'val_dataset' not in data
        assert data['split_dataset_ratio']['label'] == '验证集拆分比例(--split_dataset_ratio)'
        assert data['split_dataset_ratio']['info'] == '表示将总数据的多少拆分到验证集中'
        assert data['split_dataset_ratio']['minimum'] == 0
        assert data['split_dataset_ratio']['maximum'] == 1
        assert data['max_length']['label'] == '句子最大长度(--max_length)'
        assert data['max_length']['info'] == '设置输入模型的最大长度'
        assert data['max_length']['minimum'] == 32
        assert data['max_length']['maximum'] == 32768
        assert data['max_length']['default'] == 1024
        assert data['padding_free']['label'] == '无填充批处理(--padding_free)'
        assert data['padding_free']['info'] == '将一个batch中的数据进行展平而避免数据padding'
        assert any(block.get('dataset_panel') for block in schema['stage_layouts']['sft']['blocks'])
        for stage in ('sft', 'sft_eval', 'rl', 'rl_eval', 'promote', 'serve'):
            stage_fields = schema['fields'][stage]
            assert [row['key'] for row in stage_fields[:3]] == ['model', 'model_type', 'template']
            assert all(row['locked'] is True for row in stage_fields[:3])
        assert by_key('rl')['ref_model'].get('locked') is not True
        assert by_key('rl')['dataset'].get('locked') is not True
        assert by_key('sft')['learning_rate'].get('locked') is not True
        page = Path(__file__).resolve().parents[1].joinpath('static', 'app.js').read_text(encoding='utf-8')
        assert 'renderDatasetSettings' in page
        assert 'catalog.hf_datasets' not in page
        assert '数据集设置' in page
        assert 'data-dataset-mode="bundled"' in page
        assert 'data-field="bundled"' in page
        assert '由实验创建时决定，不能在这里修改' in page
        assert 'renderBlocks' in page
        assert 'syncStageFields' in page
        assert 'data-min-gpus' in page
        assert 'data-show-key' in page
        assert 'chip-clear' in page
        assert 'if (item.locked) return raw == null ? \'\' : raw;' in page
        assert '在线 RL' not in page

        sft_eval = by_key('sft_eval')
        assert sft_eval['eval_backend']['choices'] == ['Native', 'OpenCompass', 'VLMEvalKit']
        assert sft_eval['eval_dataset']['type'] == 'multicombo'
        assert sft_eval['eval_dataset']['choices_from'] == 'eval_datasets_by_backend'
        assert by_key('rl_eval')['eval_dataset']['type'] == 'multicombo'

        rl = by_key('rl')
        assert rl['reward_funcs']['type'] == 'multicombo'
        assert rl['reward_funcs']['choices'] == ['accuracy', 'format', 'cosine', 'repetition', 'soft_overlong']
        assert rl['reward_funcs'].get('allow_custom') is True
        assert 'rlhf_type' not in rl
        assert rl['ref_model']['type'] == 'combo'
        assert rl['ref_model'].get('locked') is not True
        assert rl['sleep_level']['choices'] == ['0', '1']
        assert rl['dataset']['type'] == 'multicombo'
        assert rl['dataset']['info'] == '选择训练的数据集，支持复选/本地路径'
        assert sft['dataset']['type'] == 'multicombo'
        assert sft['dataset']['label'] == '数据集名称(--dataset)'
        assert next(item['label'] for item in schema['stages'] if item['id'] == 'rl') == 'RL'
        assert schema['stage_layouts']['sft_eval']['by_id']['adapters']['locked'] is not True
        assert schema['stage_layouts']['sft_eval']['by_id']['adapters']['label'] == 'adapter id或路径(--adapters)'
        for stage in ('sft_eval', 'rl_eval'):
            toggle = schema['stage_layouts'][stage]['by_id']['eval_base']
            assert toggle['widget'] == 'Checkbox'
            assert toggle['label'] == '同时评测基座'
            assert toggle['default'] is False
            assert by_key(stage)['eval_base']['type'] == 'checkbox'
        assert [row['key'] for row in schema['fields']['sft_eval']] == [row['key'] for row in schema['fields']['rl_eval']]
        assert 'eval_dataset' in schema['stage_layouts']['rl_eval']['by_id']
        assert schema['stage_layouts']['sft']['by_id']['dataset']['multiselect'] is True
        assert any(block.get('label') == '数据集设置' for block in schema['stage_layouts']['sft']['blocks'])
        assert any(block.get('label') == '训练参数设置' for block in schema['stage_layouts']['sft']['blocks'])
        assert any(block.get('label') == '奖励模型设置(更多参数->GRPO高级参数设置)' for block in schema['stage_layouts']['rl']['blocks'])

        created = api.post('/api/runs', json={
            'name': 'IT 运维助手',
            'scenario': '企业内部 IT 工单分诊',
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
        }).json()
        assert created['run']['model_type'] == 'qwen2'
        assert created['run']['template'] == 'qwen2_5'
        assert created['run']['current_stage'] == 'sft'
        assert 'data' not in created['forms']
        values = created['forms']['sft']['values']
        assert values['model'] == 'Qwen/Qwen2.5-7B-Instruct'
        assert values['model_type'] == 'qwen2'
        assert values['template'] == 'qwen2_5'
        assert values['source'] == 'public'
        assert values['dataset'] == ''
        assert values['max_length'] == 2048
        for stage in ('rl', 'sft_eval', 'rl_eval'):
            locked = created['forms'][stage]['values']
            assert locked['model'] == 'Qwen/Qwen2.5-7B-Instruct'
            assert locked['model_type'] == 'qwen2'
            assert locked['template'] == 'qwen2_5'
