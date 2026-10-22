# Copyright (c) ModelScope Contributors. All rights reserved.
"""Form fields mirroring the ms-swift Web-UI / CLI argument names."""
import gzip
import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path

import yaml

from rsi.backends import cached_gpu
from rsi.llm_export_spec import export_layout
from rsi.llm_infer_spec import llm_infer_layout
from rsi.webui_stage_spec import stage_layout
from rsi.model_fetch import hf_dataset_choices, model_source
from rsi.webui_catalog import eval_datasets_by_backend, load_catalog

CONFIG_PATH = Path(__file__).resolve().parent / 'configs' / 'a100_80g.yaml'

STAGE_ORDER = ['sft', 'sft_eval', 'rl', 'rl_eval', 'promote']

STAGE_META = [
    {
        'id': 'sft',
        'label': 'SFT',
        'summary': '监督微调。先选数据集，模型默认是本轮基座。',
    },
    {
        'id': 'sft_eval',
        'label': 'SFT 评测',
        'summary': '评测这一轮的 SFT 检查点；可勾选同时评测基座以便对比。',
    },
    {
        'id': 'rl',
        'label': 'RL',
        'summary': 'GRPO。从本轮 SFT 检查点继续。',
    },
    {
        'id': 'rl_eval',
        'label': 'RL 评测',
        'summary': '用同一套评测配置检查 RL checkpoint；可勾选同时评测基座，有 SFT 评测结果时一并对比。',
    },
    {
        'id': 'promote',
        'label': '开启下一轮',
        'summary': '三选一带进下一轮的起点（实验底座 / 本轮 SFT / 本轮 RL）。实验底座模型不变，只换下一轮的 adapters（或全量 checkpoint），不会叠多份 LoRA。上一轮记录保留在实验记录里。',
    },
]

PUBLIC_DATASETS = [
    {
        'id': 'AI-ModelScope/alpaca-gpt4-data-zh',
        'note': 'Alpaca 中文指令'
    },
    {
        'id': 'AI-ModelScope/alpaca-gpt4-data-en',
        'note': 'Alpaca 英文指令'
    },
    {
        'id': 'swift/self-cognition',
        'note': '自我认知'
    },
    {
        'id': 'modelscope/gsm8k',
        'note': '数学，常用于 GRPO'
    },
    {
        'id': 'AI-MO/NuminaMath-TIR',
        'note': '数学推理'
    },
    {
        'id': 'AI-ModelScope/alpaca-gpt4-data-zh#500',
        'note': '抽样 500 条'
    },
]

# Held-out jsonl. The eval form lists these by name; the run still submits
# eval_dataset=itops-heldout plus the file path, which the gate already understands.
_EVAL_NAMES = [
    ('helpdesk_eval', 'IT 服务台验收'),
    ('itops_eval', 'IT 运维冒烟'),
    ('broad_mix_eval', '跨场景混合'),
    ('instruct_eval', '指令跟随'),
    ('dialogue_eval', '多轮对话'),
    ('math_eval', '数学'),
    ('code_eval', '代码'),
    ('exam_eval', '考试问答'),
    ('summary_eval', '摘要'),
    ('reading_eval', '阅读理解'),
    ('translate_eval', '翻译'),
    ('sql_eval', 'SQL'),
    ('tooluse_eval', '工具调用'),
    ('service_eval', '客服'),
    ('safety_eval', '安全'),
]


def bundled_eval_choices() -> list:
    root = Path(__file__).resolve().parent / 'examples'
    rows = []
    for name, label in _EVAL_NAMES:
        path = root / f'{name}.jsonl'
        if not path.is_file():
            continue
        with path.open(encoding='utf-8') as handle:
            count = sum(1 for line in handle if line.strip())
        rows.append({
            'id': name,
            'label': f'{label}（{count} 条）',
            'rows': count,
            'path': f'rsi/examples/{name}.jsonl',
        })
    return rows


# Train jsonl under rsi/examples. Eval files stay off this list so a pick cannot train on the held-out set.
BUNDLED = [
    {'id': 'helpdesk_train', 'label': 'IT 服务台验收（helpdesk_train，480 条）'},
    {'id': 'itops_mix', 'label': 'IT 运维冒烟（itops_mix，6 条）'},
    {'id': 'broad_mix', 'label': '跨场景混合（broad_mix）'},
    {'id': 'instruct', 'label': '指令跟随（instruct）'},
    {'id': 'dialogue', 'label': '多轮对话（dialogue）'},
    {'id': 'math', 'label': '数学（math）'},
    {'id': 'code', 'label': '代码（code）'},
    {'id': 'exam', 'label': '考试问答（exam）'},
    {'id': 'summary', 'label': '摘要（summary）'},
    {'id': 'reading', 'label': '阅读理解（reading）'},
    {'id': 'translate', 'label': '翻译（translate）'},
    {'id': 'sql', 'label': 'SQL（sql）'},
    {'id': 'tooluse', 'label': '工具调用（tooluse）'},
    {'id': 'service', 'label': '客服（service）'},
    {'id': 'safety', 'label': '安全（safety）'},
]


# Same set as swift.arguments.base_args.base_args.get_supported_tuners, without importing peft.
# get_supported_tuners() is this set plus swift.tuner_plugin.mapping.tuners_map.
TUNER_CHOICES = [
    'lora', 'full', 'longlora', 'adalora', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft', 'bone', 'ia3',
    'lora_llm', 'dummy'
]
TORCH_DTYPE_CHOICES = ['bfloat16', 'float16', 'float32']
DEEPSPEED_CHOICES = ['', 'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload']
ATTN_CHOICES = ['', 'sdpa', 'eager', 'flash_attention_2', 'flash_attention_3', 'flash_attention_4']
QUANT_METHOD_CHOICES = ['', 'bnb', 'hqq', 'eetq', 'quanto', 'fp8']
QUANT_BITS_CHOICES = ['', '1', '2', '3', '4', '8', 'float8']
INFER_BACKEND_CHOICES = ['transformers', 'vllm', 'sglang', 'lmdeploy', 'pt']
REWARD_FUNC_CHOICES = ['accuracy', 'format', 'cosine', 'repetition', 'soft_overlong', 'itops_checklist']


def _field(key, label, arg, type='text', group='基础', choices=None, advanced=False, step=None,
           choices_from=None, info=None, minimum=None, maximum=None, default=None, locked=False):
    item = {
        'key': key,
        'label': label,
        'arg': arg,
        'type': type,
        'group': group,
        'advanced': advanced,
    }
    if choices is not None:
        item['choices'] = choices
    if choices_from is not None:
        item['choices_from'] = choices_from
        item['allow_custom'] = True
    if step is not None:
        item['step'] = step
    if info:
        item['info'] = info
    if minimum is not None:
        item['minimum'] = minimum
    if maximum is not None:
        item['maximum'] = maximum
    if default is not None:
        item['default'] = default
    if locked:
        item['locked'] = True
    return item


def _model_block(group='模型'):
    """``--model``, ``--model_type``, and ``--template`` are chosen only when the experiment is created."""
    return [
        _field('model', '模型id或路径(--model)', '--model', 'combo', group, choices_from='models', locked=True),
        _field('model_type', '模型类型(--model_type)', '--model_type', 'combo', group, choices_from='model_types',
               locked=True),
        _field('template', '模型Prompt模板类型(--template)', '--template', 'combo', group, choices_from='templates',
               locked=True),
    ]


FIELDS = {
    'data': [
        *_model_block(),
        _field(
            'source',
            '数据来源',
            '',
            'select',
            '数据集设置',
            [{
                'value': 'bundled',
                'label': '内置示例'
            }, {
                'value': 'public',
                'label': '公开数据集'
            }, {
                'value': 'upload',
                'label': '上传文件'
            }, {
                'value': 'path',
                'label': '服务器本地路径'
            }]),
        _field('dataset', '数据集名称(--dataset)', '--dataset', 'multicombo', '数据集设置', choices_from='datasets',
               info='选择训练的数据集，支持复选/本地路径'),
        _field('split_dataset_ratio', '验证集拆分比例(--split_dataset_ratio)', '--split_dataset_ratio', 'slider',
               '数据集设置', step='0.05', minimum=0, maximum=1, default=0,
               info='表示将总数据的多少拆分到验证集中'),
        _field('max_length', '句子最大长度(--max_length)', '--max_length', 'slider', '数据集设置', step='1',
               minimum=32, maximum=32768, default=1024, info='设置输入模型的最大长度'),
        _field('padding_free', '无填充批处理(--padding_free)', '--padding_free', 'checkbox', '数据集设置',
               info='将一个batch中的数据进行展平而避免数据padding'),
    ],
    'sft': [
        *_model_block(),
        _field('system', 'System字段(--system)', '--system', 'textarea', '模型'),
        _field('torch_dtype', '训练精度(--torch_dtype)', '--torch_dtype', 'select', '模型', TORCH_DTYPE_CHOICES),
        _field('seed', '随机种子', '--seed', 'number', '模型', step='1'),
        _field('tuner_type', '训练方式(--tuner_type)', '--tuner_type', 'select', '调参', TUNER_CHOICES),
        _field('lora_rank', 'LoRA rank', '--lora_rank', 'number', '调参', step='1'),
        _field('lora_alpha', 'LoRA alpha', '--lora_alpha', 'number', '调参', step='1'),
        _field('lora_dropout', 'LoRA dropout', '--lora_dropout', 'number', '调参', step='0.01'),
        _field('target_modules', 'Target modules', '--target_modules', 'text', '调参'),
        _field('use_dora', '使用 DoRA', '--use_dora', 'checkbox', '调参'),
        _field('use_rslora', '使用 rsLoRA', '--use_rslora', 'checkbox', '调参'),
        _field('num_train_epochs', '训练轮次', '--num_train_epochs', 'number', '超参数', step='0.1'),
        _field('learning_rate', '学习率', '--learning_rate', 'text', '超参数'),
        _field('per_device_train_batch_size', '训练 batch size', '--per_device_train_batch_size', 'number', '超参数',
               step='1'),
        _field('per_device_eval_batch_size', '验证 batch size', '--per_device_eval_batch_size', 'number', '超参数',
               step='1'),
        _field('gradient_accumulation_steps', '梯度累积', '--gradient_accumulation_steps', 'number', '超参数', step='1'),
        _field('gradient_checkpointing', '梯度检查点', '--gradient_checkpointing', 'checkbox', '显存'),
        _field('warmup_ratio', 'Warmup 比例', '--warmup_ratio', 'number', '超参数', step='0.01'),
        _field('lr_scheduler_type', '学习率调度(--lr_scheduler_type)', '--lr_scheduler_type', 'text', '超参数'),
        _field('weight_decay', '权重衰减', '--weight_decay', 'number', '超参数', advanced=True, step='0.01'),
        _field('max_length', '最大长度', '--max_length', 'number', '超参数', step='1'),
        _field('logging_steps', '日志步数', '--logging_steps', 'number', '超参数', step='1'),
        _field('eval_steps', '评估步数', '--eval_steps', 'number', '超参数', step='1'),
        _field('save_steps', '保存步数', '--save_steps', 'number', '超参数', step='1'),
        _field('save_total_limit', '最多保留检查点', '--save_total_limit', 'number', '超参数', step='1'),
        _field('split_dataset_ratio', '验证集比例', '--split_dataset_ratio', 'number', '超参数', advanced=True, step='0.01'),
        _field('dataset_num_proc', '数据集进程数', '--dataset_num_proc', 'number', '资源', advanced=True, step='1'),
        _field('dataloader_num_workers', 'DataLoader workers', '--dataloader_num_workers', 'number', '资源',
               advanced=True, step='1'),
        _field('load_from_cache_file', '使用数据集缓存', '--load_from_cache_file', 'checkbox', '资源', advanced=True),
        _field('padding_free', '无填充批处理', '--padding_free', 'checkbox', '显存', advanced=True),
        _field('neftune_noise_alpha', 'NEFTune 噪声', '--neftune_noise_alpha', 'number', '超参数', advanced=True,
               step='0.5'),
        _field('attn_impl', 'Attention 实现(--attn_impl)', '--attn_impl', 'select', '显存', ATTN_CHOICES, advanced=True),
        _field('use_liger_kernel', 'Liger kernel', '--use_liger_kernel', 'checkbox', '显存', advanced=True),
        _field('deepspeed', 'DeepSpeed(--deepspeed)', '--deepspeed', 'select', '显存', DEEPSPEED_CHOICES, advanced=True),
        _field('sequence_parallel_size', '序列并行', '--sequence_parallel_size', 'number', '显存', advanced=True, step='1'),
        _field('quant_method', '量化方式(--quant_method)', '--quant_method', 'select', '显存', QUANT_METHOD_CHOICES,
               advanced=True),
        _field('quant_bits', '量化bit数(--quant_bits)', '--quant_bits', 'select', '显存', QUANT_BITS_CHOICES,
               advanced=True),
        _field('dataset', '数据集名称(--dataset)', '--dataset', 'combo', '数据', choices_from='datasets', advanced=True),
    ],
    'sft_eval': [
        *_model_block(),
        _field('eval_backend', '评测后端(--eval_backend)', '--eval_backend', 'select', '评测',
               ['Native', 'OpenCompass', 'VLMEvalKit']),
        _field('eval_dataset', '评测数据集(--eval_dataset)', '--eval_dataset', 'multicombo', '评测',
               choices_from='eval_datasets_by_backend'),
        _field('eval_limit', '每个数据集取样数', '--eval_limit', 'number', '评测', step='1'),
        _field('eval_base', '同时评测基座', '', 'checkbox', '评测', default=False,
               info='不勾选时只评测本阶段的 adapter / checkpoint；勾选后用同一套评测集再跑实验基座（无 adapter）。'),
        _field('custom_eval_config', '自定义评测配置', '--extra_eval_args', 'text', '评测'),
        _field('temperature', '温度', '--temperature', 'number', '生成', step='0.1'),
        _field('max_new_tokens', '最大生成长度', '--max_new_tokens', 'number', '生成', step='1'),
        _field('eval_num_proc', '并发数', '--eval_num_proc', 'number', '资源', step='1'),
        _field('infer_backend', '推理框架(--infer_backend)', '--infer_backend', 'select', '资源', INFER_BACKEND_CHOICES),
        _field('verbose', '详细日志', '--verbose', 'checkbox', '评测', advanced=True),
    ],
    'rl': [
        *_model_block(),
        _field('rlhf_type', 'RLHF 类型', '--rlhf_type', 'select', '算法',
               ['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo', 'gkd']),
        _field('dataset', '数据集名称(--dataset)', '--dataset', 'combo', '数据', choices_from='datasets'),
        _field('reward_funcs', '奖励函数(--reward_funcs)', '--reward_funcs', 'multiselect', '奖励', REWARD_FUNC_CHOICES),
        _field('reward_weights', '奖励权重', '--reward_weights', 'text', '奖励'),
        _field('external_plugins', '外部插件', '--external_plugins', 'text', '奖励'),
        _field('num_generations', '采样数量 G', '--num_generations', 'number', '采样', step='1'),
        _field('max_completion_length', '最大生成长度', '--max_completion_length', 'number', '采样', step='1'),
        _field('temperature', '采样温度', '--temperature', 'number', '采样', step='0.1'),
        _field('top_p', 'Top-p', '--top_p', 'number', '采样', step='0.01'),
        _field('top_k', 'Top-k', '--top_k', 'number', '采样', step='1'),
        _field('beta', 'KL 系数', '--beta', 'number', '算法', step='0.01'),
        _field('epsilon', 'clip epsilon', '--epsilon', 'number', '算法', advanced=True, step='0.01'),
        _field('epsilon_high', 'epsilon high', '--epsilon_high', 'number', '算法', advanced=True, step='0.01'),
        _field('num_iterations', 'num_iterations', '--num_iterations', 'number', '算法', advanced=True, step='1'),
        _field('learning_rate', '学习率', '--learning_rate', 'text', '超参数'),
        _field('num_train_epochs', '训练轮次', '--num_train_epochs', 'number', '超参数', step='0.1'),
        _field('per_device_train_batch_size', '训练 batch size', '--per_device_train_batch_size', 'number', '超参数',
               step='1'),
        _field('gradient_accumulation_steps', '梯度累积', '--gradient_accumulation_steps', 'number', '超参数', step='1'),
        _field('tuner_type', '训练方式(--tuner_type)', '--tuner_type', 'select', '调参', TUNER_CHOICES),
        _field('lora_rank', 'LoRA rank', '--lora_rank', 'number', '调参', step='1'),
        _field('lora_alpha', 'LoRA alpha', '--lora_alpha', 'number', '调参', step='1'),
        _field('target_modules', 'Target modules', '--target_modules', 'text', '调参'),
        _field('max_length', '最大长度', '--max_length', 'number', '超参数', step='1'),
        _field('torch_dtype', '训练精度(--torch_dtype)', '--torch_dtype', 'select', '模型', TORCH_DTYPE_CHOICES),
        _field('use_vllm', '使用 vLLM', '--use_vllm', 'checkbox', '显存'),
        _field('vllm_mode', 'vLLM 模式', '--vllm_mode', 'select', '显存', ['colocate', 'server']),
        _field('vllm_gpu_memory_utilization', 'vLLM 显存利用率', '--vllm_gpu_memory_utilization', 'number', '显存',
               step='0.01'),
        _field('vllm_tensor_parallel_size', '张量并行', '--vllm_tensor_parallel_size', 'number', '显存', step='1'),
        _field('vllm_max_model_len', 'vLLM 最大长度', '--vllm_max_model_len', 'number', '显存', step='1'),
        _field('vllm_enable_lora', 'vLLM 加载 LoRA', '--vllm_enable_lora', 'checkbox', '显存'),
        _field('vllm_max_lora_rank', 'vLLM LoRA rank 上限', '--vllm_max_lora_rank', 'number', '显存', step='1'),
        _field('sleep_level', 'Sleep level(--sleep_level)', '--sleep_level', 'select', '显存', ['0', '1']),
        _field('offload_model', '卸载模型', '--offload_model', 'checkbox', '显存'),
        _field('offload_optimizer', '卸载优化器', '--offload_optimizer', 'checkbox', '显存'),
        _field('gradient_checkpointing', '梯度检查点', '--gradient_checkpointing', 'checkbox', '显存'),
        _field('log_completions', '记录采样', '--log_completions', 'checkbox', '日志'),
        _field('seed', '随机种子', '--seed', 'number', '模型', advanced=True, step='1'),
        _field('system', 'System', '--system', 'textarea', '模型', advanced=True),
        _field('deepspeed', 'DeepSpeed(--deepspeed)', '--deepspeed', 'select', '显存', DEEPSPEED_CHOICES, advanced=True),
        _field('quant_method', '量化方式(--quant_method)', '--quant_method', 'select', '显存', QUANT_METHOD_CHOICES,
               advanced=True),
        _field('quant_bits', '量化bit数(--quant_bits)', '--quant_bits', 'select', '显存', QUANT_BITS_CHOICES,
               advanced=True),
        _field('ref_model', 'Ref 模型(--ref_model)', '--ref_model', 'combo', '算法', choices_from='models', advanced=True),
        _field('loss_scale', '损失权重', '--loss_scale', 'text', '算法', advanced=True),
        _field('dataset_num_proc', '数据集进程数', '--dataset_num_proc', 'number', '资源', advanced=True, step='1'),
        _field('dataloader_num_workers', 'DataLoader workers', '--dataloader_num_workers', 'number', '资源',
               advanced=True, step='1'),
        _field('split_dataset_ratio', '验证集比例', '--split_dataset_ratio', 'number', '数据', advanced=True, step='0.01'),
        _field('load_from_cache_file', '使用数据集缓存', '--load_from_cache_file', 'checkbox', '资源', advanced=True),
        _field('logging_steps', '日志步数', '--logging_steps', 'number', '日志', advanced=True, step='1'),
        _field('save_steps', '保存步数', '--save_steps', 'number', '日志', advanced=True, step='1'),
        _field('save_total_limit', '最多保留检查点', '--save_total_limit', 'number', '日志', advanced=True, step='1'),
        _field('warmup_ratio', 'Warmup 比例', '--warmup_ratio', 'number', '超参数', advanced=True, step='0.01'),
    ],
    'rl_eval': [
        *_model_block(),
        _field('eval_backend', '评测后端(--eval_backend)', '--eval_backend', 'select', '评测',
               ['Native', 'OpenCompass', 'VLMEvalKit']),
        _field('eval_dataset', '评测数据集(--eval_dataset)', '--eval_dataset', 'multicombo', '评测',
               choices_from='eval_datasets_by_backend'),
        _field('eval_limit', '每个数据集取样数', '--eval_limit', 'number', '评测', step='1'),
        _field('eval_base', '同时评测基座', '', 'checkbox', '评测', default=False,
               info='不勾选时只评测本阶段的 adapter / checkpoint；勾选后用同一套评测集再跑实验基座（无 adapter）。'),
        _field('custom_eval_config', '自定义评测配置', '--extra_eval_args', 'text', '评测'),
        _field('temperature', '温度', '--temperature', 'number', '生成', step='0.1'),
        _field('max_new_tokens', '最大生成长度', '--max_new_tokens', 'number', '生成', step='1'),
        _field('eval_num_proc', '并发数', '--eval_num_proc', 'number', '资源', step='1'),
        _field('infer_backend', '推理框架(--infer_backend)', '--infer_backend', 'select', '资源', INFER_BACKEND_CHOICES),
        _field('verbose', '详细日志', '--verbose', 'checkbox', '评测', advanced=True),
    ],
    'promote': [
        *_model_block(),
        _field('reason', '晋级说明', '', 'textarea', '决策'),
        _field('reuse_dataset', '沿用上一轮数据', '', 'checkbox', '下一轮'),
        _field('note', '备注', '', 'textarea', '下一轮'),
    ],
    'serve': [
        *_model_block('部署'),
        _field('adapters', 'LoRA 路径(--adapters)', '--adapters', 'text', '部署'),
        _field('merge_lora', '合并 LoRA(--merge_lora)', '--merge_lora', 'checkbox', '部署'),
        _field('infer_backend', '推理框架(--infer_backend)', '--infer_backend', 'select', '部署', INFER_BACKEND_CHOICES),
        _field('system', 'System字段(--system)', '--system', 'textarea', '生成'),
        _field('max_new_tokens', '生成序列最大长度', '--max_new_tokens', 'number', '生成', step='1'),
        _field('temperature', '采样温度', '--temperature', 'number', '生成', step='0.1'),
        _field('top_k', 'Top-k', '--top_k', 'number', '生成', step='1'),
        _field('top_p', 'Top-p', '--top_p', 'number', '生成', step='0.01'),
        _field('repetition_penalty', '重复惩罚', '--repetition_penalty', 'number', '生成', step='0.05'),
        _field('more_params', '更多参数', '', 'text', '生成'),
        _field('served_model_name', '服务模型名', '--served_model_name', 'text', '部署'),
        _field('host', 'Host', '--host', 'text', '部署'),
        _field('port', '服务端口', '--port', 'number', '部署', step='1'),
        _field('api_key', 'API key', '--api_key', 'text', '部署'),
        _field('torch_dtype', '精度(--torch_dtype)', '--torch_dtype', 'select', '部署', TORCH_DTYPE_CHOICES),
        _field('vllm_gpu_memory_utilization', 'vLLM 显存利用率', '--vllm_gpu_memory_utilization', 'number', '显存',
               advanced=True, step='0.01'),
        _field('vllm_max_model_len', 'vLLM 最大长度', '--vllm_max_model_len', 'number', '显存', advanced=True, step='1'),
    ],
}


@lru_cache(maxsize=1)
def load_preset_file() -> dict:
    with CONFIG_PATH.open(encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f'无法读取预设：{CONFIG_PATH}')
    return data


def presets() -> dict:
    return load_preset_file()


# Built once per (GPU count, model source). Cold build walks Web-UI AST + catalogs;
# the JSON is ~350KB so we also trim wire duplicates below. Wire bytes (raw + gzip +
# ETag) are computed once with the dict so the hit path skips encoder/compress.
_SCHEMA_CACHE: dict[tuple, dict] = {}
_SCHEMA_WIRE: dict[tuple, tuple[bytes, bytes, str]] = {}
_SCHEMA_LOCK = threading.Lock()


def clear_public_schema_cache() -> None:
    with _SCHEMA_LOCK:
        _SCHEMA_CACHE.clear()
        _SCHEMA_WIRE.clear()


def _wire_export_layout(device_count: int, devices: tuple = ()) -> dict:
    """Client renders ``sections`` + ``by_id``; drop triplicate ``blocks`` / ``fields``."""
    layout = export_layout(device_count, devices)
    return {key: value for key, value in layout.items() if key not in {'blocks', 'fields'}}


def _schema_cache_key() -> tuple:
    # Device *names* belong in the key so GPU label text refreshes when cards change.
    devices = tuple(cached_gpu().get('devices') or [])
    return (devices, model_source())


def _store_schema(key: tuple, schema: dict) -> dict:
    raw = json.dumps(schema, ensure_ascii=False, separators=(',', ':')).encode()
    gzipped = gzip.compress(raw, 6)
    etag = '"' + hashlib.blake2b(raw, digest_size=16).hexdigest() + '"'
    _SCHEMA_CACHE[key] = schema
    _SCHEMA_WIRE[key] = (raw, gzipped, etag)
    return schema


def _build_public_schema(devices) -> dict:
    if isinstance(devices, int):
        device_list: tuple = ()
        device_count = int(devices)
    else:
        device_list = tuple(devices or ())
        device_count = len(device_list)
    preset = presets()
    catalog = load_catalog()
    stage_layouts = {
        stage: stage_layout(stage, device_count, devices=device_list)
        for stage in ('sft', 'sft_eval', 'rl', 'rl_eval')
    }
    fields = {key: list(value) for key, value in FIELDS.items()}
    for stage, layout in stage_layouts.items():
        fields[stage] = layout['fields']
        # Top-level ``fields`` already carries the flat list; omit the copy in layouts.
        layout.pop('fields', None)
    datasets = list(catalog['datasets'])
    seen = set(datasets)
    for item in PUBLIC_DATASETS:
        if item['id'] not in seen:
            datasets.append(item['id'])
            seen.add(item['id'])
    # Keep swift/dataset registration order. Web-UI ``get_dataset_list()`` does not sort.
    sft_layout = stage_layouts['sft']
    sft_layout['dataset_fields'] = [
        field for field in FIELDS['data'] if field['key'] not in {'model', 'model_type', 'template'}
    ]
    for block in sft_layout.get('blocks') or []:
        if block.get('label') == '数据集设置':
            block['dataset_panel'] = True
    fields.pop('data', None)
    return {
        'stages': STAGE_META,
        'stage_order': STAGE_ORDER,
        'fields': fields,
        'stage_layouts': stage_layouts,
        'presets': {key: preset.get(key, {})
                    for key in ('data', 'sft', 'sft_eval', 'rl', 'rl_eval', 'serve', 'promote')},
        'public_datasets': PUBLIC_DATASETS,
        'bundled': BUNDLED,
        'bundled_eval': bundled_eval_choices(),
        'hardware': preset.get('hardware', {}),
        'base_model_default': 'Qwen/Qwen2.5-7B-Instruct',
        'model_source': model_source(),
        'llm_infer': llm_infer_layout(device_count, devices=device_list),
        'llm_export': _wire_export_layout(device_count, device_list),
        'catalog': {
            # Keep Web-UI / ModelType registration order (Qwen first). Do not sort.
            'models': list(catalog['models']),
            'model_types': list(catalog['model_types']),
            'templates': list(catalog['templates']),
            'datasets': datasets,
            'hf_datasets': hf_dataset_choices(),
            'eval_datasets': eval_datasets_by_backend()['Native'],
            'eval_datasets_by_backend': eval_datasets_by_backend(),
            'by_model': catalog['by_model'],
        },
    }


def public_schema() -> dict:
    """Return the public schema dict (single-flight cold build)."""
    key = _schema_cache_key()
    cached = _SCHEMA_CACHE.get(key)
    if cached is not None:
        return cached
    with _SCHEMA_LOCK:
        cached = _SCHEMA_CACHE.get(key)
        if cached is not None:
            return cached
        return _store_schema(key, _build_public_schema(key[0]))


def public_schema_wire() -> tuple[bytes, bytes, str]:
    """Pre-serialized JSON, gzip body, and quoted ETag for ``/api/schema``."""
    key = _schema_cache_key()
    wire = _SCHEMA_WIRE.get(key)
    if wire is not None:
        return wire
    with _SCHEMA_LOCK:
        wire = _SCHEMA_WIRE.get(key)
        if wire is not None:
            return wire
        _store_schema(key, _build_public_schema(key[0]))
        return _SCHEMA_WIRE[key]
