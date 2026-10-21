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


def test_sft_layout_hides_single_gpu_and_other_tuner_entries():
    layout = stage_layout('sft', 1)
    assert layout['by_id']['deepspeed']['min_gpus'] == 2
    assert layout['by_id']['deepspeed']['choices'][0] == ''
    assert 'zero0' in layout['by_id']['deepspeed']['choices']
    assert layout['by_id']['ddp_num']['min_gpus'] == 2
    assert layout['by_id']['gpu_id'].get('min_gpus') is None
    tabs = {tab['id']: tab for tab in _tabs(layout['blocks'])}
    assert tabs['lora_tab']['show_values'] == ['lora', 'longlora', 'adalora']
    assert tabs['adalora_tab']['show_key'] == 'tuner_type'
    assert tabs['adalora_tab']['show_values'] == ['adalora']
    assert tabs['vera_tab']['show_values'] == ['vera']
    assert tabs['embed_tab']['show_key'] == 'task_type'
    assert tabs['lora_ga_tab']['show_key'] == 'init_weights'
    rl = stage_layout('rl', 2)
    assert rl['by_id']['deepspeed']['min_gpus'] == 2
    assert rl['by_id']['use_ddp']['min_gpus'] == 2


def test_sft_quant_selects_default_to_empty_not_bnb():
    """Unset Gradio quant dropdowns must not force the first HTML option (bnb)."""
    layout = stage_layout('sft', 1)
    assert layout['by_id']['quant_method']['choices'][0] == ''
    assert 'bnb' in layout['by_id']['quant_method']['choices']
    assert layout['by_id']['quant_bits']['choices'][0] == ''
    assert layout['by_id']['bnb_4bit_quant_type']['choices'][0] == ''
    rl = stage_layout('rl', 1)
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
