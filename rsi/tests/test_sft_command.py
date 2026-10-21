# Copyright (c) ModelScope Contributors. All rights reserved.
from rsi.commands import build_argv, env_prefix, selected_gpu_count
from rsi.webui_stage_spec import stage_layout


def _flags(argv):
    return {argv[index] for index in range(len(argv)) if argv[index].startswith('--')}


def test_selected_gpu_count():
    assert selected_gpu_count({}) == 1
    assert selected_gpu_count({'gpu_id': '0'}) == 1
    assert selected_gpu_count({'gpu_id': ['0']}) == 1
    assert selected_gpu_count({'gpu_id': '0,1'}) == 2
    assert selected_gpu_count({'gpu_id': 'cpu'}) == 0


def test_single_gpu_sft_drops_deepspeed_and_other_tuners():
    argv = build_argv('sft', {
        'model': 'Qwen/Qwen2.5-7B-Instruct',
        'tuner_type': 'lora',
        'lora_rank': 9,
        'lora_alpha': 33,
        'target_modules': 'all-linear',
        'deepspeed': 'zero0',
        'sequence_parallel_size': 1,
        'use_ddp': True,
        'ddp_num': 2,
        'task_type': 'causal_lm',
        'loss_type': 'cosine_similarity',
        'use_chat_template': True,
        'lisa_activated_layers': 0,
        'lisa_step_interval': 20,
        'adalora_target_r': 8,
        'vera_rank': 257,
        'boft_block_size': 4,
        'fourier_n_frequency': 2000,
        'reft_rank': 5,
        'use_galore': False,
        'galore_rank': 136,
        'use_muon': False,
        'gpu_id': '0',
        'quant_method': 'bnb',
        'quant_bits': 4,
        'bnb_4bit_quant_type': 'fp4',
    })
    flags = _flags(argv)
    assert '--deepspeed' not in flags
    assert '--sequence_parallel_size' not in flags
    assert '--loss_type' not in flags
    assert '--use_chat_template' not in flags
    assert '--adalora_target_r' not in flags
    assert '--vera_rank' not in flags
    assert '--boft_block_size' not in flags
    assert '--fourier_n_frequency' not in flags
    assert '--reft_rank' not in flags
    assert '--lisa_activated_layers' not in flags
    assert '--galore_rank' not in flags
    assert '--use_galore' not in flags
    assert '--use_muon' not in flags
    assert '--lora_rank' in flags
    assert '--lora_alpha' in flags
    assert '--target_modules' in flags
    assert '--quant_method' in flags
    assert '--bnb_4bit_quant_type' in flags
    assert 'NPROC_PER_NODE' not in env_prefix({'gpu_id': '0', 'use_ddp': True, 'ddp_num': 2})


def test_two_gpus_keep_deepspeed_and_sequence_parallel():
    params = {'gpu_id': '0 1', 'deepspeed': 'zero2', 'sequence_parallel_size': 2, 'tuner_type': 'lora', 'use_ddp': True,
              'ddp_num': 2}
    argv = build_argv('sft', params)
    assert argv[argv.index('--deepspeed') + 1] == 'zero2'
    assert argv[argv.index('--sequence_parallel_size') + 1] == '2'
    assert 'NPROC_PER_NODE=2' in env_prefix(params)


def test_sequence_parallel_size_one_is_omitted_on_two_gpus():
    argv = build_argv('sft', {'gpu_id': ['0', '1'], 'sequence_parallel_size': 1, 'deepspeed': 'zero0'})
    assert '--sequence_parallel_size' not in _flags(argv)
    assert '--deepspeed' in _flags(argv)


def test_adalora_keeps_lora_args_and_drops_vera():
    argv = build_argv('sft', {
        'tuner_type': 'adalora',
        'lora_rank': 8,
        'adalora_target_r': 8,
        'vera_rank': 256,
        'gpu_id': ['0'],
    })
    flags = _flags(argv)
    assert '--lora_rank' in flags
    assert '--adalora_target_r' in flags
    assert '--vera_rank' not in flags


def test_embedding_keeps_loss_type_and_lora_ga_follows_init():
    embedding = build_argv('sft', {'task_type': 'embedding', 'loss_type': 'cosine_similarity', 'gpu_id': 'cpu'})
    assert '--loss_type' in _flags(embedding)
    assert '--deepspeed' not in _flags(build_argv('sft', {'gpu_id': 'cpu', 'deepspeed': 'zero3_offload'}))
    kept = build_argv('sft', {
        'tuner_type': 'lora',
        'init_weights': 'lora-ga',
        'lora_ga_direction': 'ArB2r',
        'vera_rank': 8,
    })
    assert '--lora_ga_direction' in _flags(kept)
    assert '--vera_rank' not in _flags(kept)


def test_rl_single_gpu_keeps_colocate_and_drops_other_modes():
    argv = build_argv('rl', {
        'tuner_type': 'lora',
        'lora_rank': 8,
        'adalora_target_r': 8,
        'vllm_mode': 'colocate',
        'vllm_gpu_memory_utilization': 0.45,
        'vllm_server_host': '127.0.0.1',
        'loss_type': 'grpo',
        'deepspeed': 'zero0',
        'gpu_id': '0',
    })
    flags = _flags(argv)
    assert '--deepspeed' not in flags
    assert '--adalora_target_r' not in flags
    assert '--vllm_server_host' not in flags
    assert '--lora_rank' in flags
    assert '--vllm_gpu_memory_utilization' in flags
    assert '--loss_type' in flags


def test_rl_reward_weights_split_into_argv_tokens():
    """Space-separated reward_weights must be nargs floats, not one quoted token."""
    argv = build_argv('rl', {
        'tuner_type': 'lora',
        'reward_funcs': 'itops_checklist format',
        'reward_weights': '1.0 0.5',
        'gpu_id': '0',
    })
    assert '--reward_weights' in argv
    weight_at = argv.index('--reward_weights')
    assert argv[weight_at:weight_at + 3] == ['--reward_weights', '1.0', '0.5']
    assert '1.0 0.5' not in argv
    funcs_at = argv.index('--reward_funcs')
    assert argv[funcs_at:funcs_at + 3] == ['--reward_funcs', 'itops_checklist', 'format']


def test_rl_lora_drops_sticky_ref_model():
    """LoRA GRPO rejects --ref_model; sticky Gradio first-option junk must not ride along."""
    argv = build_argv('rl', {
        'tuner_type': 'lora',
        'model': 'Qwen/Qwen2.5-7B-Instruct',
        'ref_model': '01ai/Yi-1.5-34B',
        'ref_model_type': 'aya',
        'gpu_id': '0',
    })
    flags = _flags(argv)
    assert '--ref_model' not in flags
    assert '--ref_model_type' not in flags


def test_default_rl_preset_omits_ref_model_and_splits_weights():
    from rsi.schemas import presets

    preset = dict(presets().get('rl') or {})
    assert preset.get('per_device_eval_batch_size') == 1
    assert preset.get('neftune_noise_alpha') == 0
    assert not str(preset.get('ref_model') or '').strip()

    params = dict(preset)
    params.update({
        'model': 'Qwen/Qwen2.5-7B-Instruct',
        'gpu_id': '0',
        # Sticky UI junk that must not reach swift rlhf under LoRA.
        'ref_model': '01ai/Yi-1.5-34B',
        'ref_model_type': 'aya',
    })
    argv = build_argv('rl', params)
    flags = _flags(argv)
    assert '--ref_model' not in flags
    assert '--ref_model_type' not in flags
    weight_at = argv.index('--reward_weights')
    assert argv[weight_at:weight_at + 3] == ['--reward_weights', '1.0', '0.5']


def _tabs(blocks):
    found = []
    for block in blocks:
        kind = block.get('kind')
        if kind == 'tabs':
            for tab in block.get('tabs') or []:
                found.append(tab)
                found.extend(_tabs(tab.get('blocks') or []))
        elif kind in {'accordion', 'tab'}:
            found.extend(_tabs(block.get('blocks') or []))
    return found


def _top_labels(blocks):
    labels = []
    for block in blocks:
        kind = block.get('kind')
        if kind == 'accordion':
            labels.append(block.get('label') or block.get('id') or '')
        elif kind == 'row':
            for item in block.get('widgets') or []:
                labels.append(item.get('label') or item.get('id') or '')
    return labels


def test_sft_layout_hides_single_gpu_and_keeps_core_panels():
    layout = stage_layout('sft', 1)
    assert layout['by_id']['deepspeed']['min_gpus'] == 2
    assert layout['by_id']['deepspeed']['choices'][0] == ''
    assert 'zero0' in layout['by_id']['deepspeed']['choices']
    assert layout['by_id']['ddp_num']['min_gpus'] == 2
    assert layout['by_id']['gpu_id'].get('min_gpus') is None
    assert layout['by_id']['tuner_type']['choices']
    assert 'lora_rank' not in layout['by_id']
    assert 'quant_method' not in layout['by_id']
    assert not _tabs(layout['blocks'])
    labels = _top_labels(layout['blocks'])
    assert '环境变量' not in labels
    assert 'Tuner参数' not in labels
    assert '优化器参数' not in labels
    assert '任务参数' not in labels
    assert '其他参数设置' not in labels
    assert '模型设置' in labels
    assert '数据集设置' in labels
    assert '训练参数设置' in labels
    rl = stage_layout('rl', 2)
    assert rl['by_id']['deepspeed']['min_gpus'] == 2
    assert rl['by_id']['use_ddp']['min_gpus'] == 2
    assert 'lora_rank' not in rl['by_id']
    rl_labels = _top_labels(rl['blocks'])
    assert '环境变量' not in rl_labels
    assert 'Tuner参数' not in rl_labels


def test_sft_layout_drops_quant_and_advanced_accordions():
    """RSI SFT must not mirror Web-UI quant / tuner / optimizer / task shells."""
    layout = stage_layout('sft', 1)
    assert 'quant_method' not in layout['by_id']
    assert 'quant_bits' not in layout['by_id']
    assert 'bnb_4bit_quant_type' not in layout['by_id']
    assert 'envs' not in layout['by_id']
    assert 'use_galore' not in layout['by_id']
    assert 'task_type' not in layout['by_id']
    assert 'more_params' not in layout['by_id']
    rl = stage_layout('rl', 1)
    assert 'envs' not in rl['by_id']
    assert 'lora_rank' not in rl['by_id']
    # RL still has GRPO advanced (ref_model); quant stays optional-empty if present.
    if 'quant_method' in rl['by_id']:
        assert rl['by_id']['quant_method']['choices'][0] == ''


def test_default_sft_command_omits_bnb_quant_flags():
    from rsi.schemas import presets

    params = dict(presets().get('sft') or {})
    params.update({
        'model': 'Qwen/Qwen2.5-7B-Instruct',
        'gpu_id': '0',
        # Stale browser / draft values that must not ride along without quant_method.
        'bnb_4bit_quant_type': 'fp4',
        'bnb_4bit_use_double_quant': False,
        'bnb_4bit_compute_dtype': 'bfloat16',
    })
    argv = build_argv('sft', params)
    flags = _flags(argv)
    assert '--quant_method' not in flags
    assert '--quant_bits' not in flags
    assert '--bnb_4bit_quant_type' not in flags
    assert '--bnb_4bit_use_double_quant' not in flags
    assert '--bnb_4bit_compute_dtype' not in flags
    assert '--lora_rank' in flags


def test_empty_quant_method_drops_bnb_even_when_bits_set():
    argv = build_argv('sft', {
        'model': 'Qwen/Qwen2.5-7B-Instruct',
        'quant_method': '',
        'quant_bits': 4,
        'bnb_4bit_quant_type': 'fp4',
        'bnb_4bit_use_double_quant': False,
        'gpu_id': '0',
    })
    flags = _flags(argv)
    assert '--quant_method' not in flags
    assert '--quant_bits' not in flags
    assert '--bnb_4bit_quant_type' not in flags


def test_sticky_draft_bnb_is_stripped_before_sft_prepare(tmp_path, monkeypatch):
    """Old drafts with quant_method=bnb must not reach the swift command."""
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    monkeypatch.setenv('RSI_HOME', str(tmp_path / 'home'))
    from fastapi.testclient import TestClient

    from rsi.app import create_app
    from rsi.commands import strip_form_quant

    sticky = {
        'gpu_id': '0',
        'quant_method': 'bnb',
        'quant_bits': 4,
        'bnb_4bit_quant_type': 'fp4',
        'bnb_4bit_use_double_quant': False,
        'dataset': 'AI-ModelScope/alpaca-gpt4-data-zh',
        'source': 'public',
    }
    assert 'quant_method' not in strip_form_quant(sticky)
    with TestClient(create_app(tmp_path / 'home')) as api:
        created = api.post('/api/runs', json={
            'name': 'bnb-sticky',
            'scenario': 'test',
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
        })
        assert created.status_code == 200, created.text
        run_id = created.json()['run']['id']
        preview = api.post(f'/api/runs/{run_id}/stages/sft/preview', json={'params': sticky})
        assert preview.status_code == 200, preview.text
        command = preview.json().get('command') or ''
        assert '--quant_method' not in command
        assert '--bnb_4bit_quant_type' not in command
        assert '--lora_rank' in command
        # Saving a sticky draft must not keep quant keys for a later refresh.
        saved = api.post(f'/api/runs/{run_id}/drafts', json={'stage': 'sft', 'params': sticky})
        assert saved.status_code == 200, saved.text
        assert 'quant_method' not in (saved.json().get('params') or {})
        view = api.get(f'/api/runs/{run_id}').json()
        values = ((view.get('forms') or {}).get('sft') or {}).get('values') or {}
        assert 'quant_method' not in values
        assert 'bnb_4bit_quant_type' not in values
