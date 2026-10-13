# Copyright (c) ModelScope Contributors. All rights reserved.
"""Form fields mirroring the ms-swift Web-UI / CLI argument names."""
from functools import lru_cache
from pathlib import Path

import yaml

from rsi.webui_catalog import load_catalog

CONFIG_PATH = Path(__file__).resolve().parent / 'configs' / 'a100_80g.yaml'

STAGE_ORDER = ['data', 'sft', 'sft_eval', 'rl', 'rl_eval', 'promote', 'next_round']

STAGE_META = [
    {
        'id': 'data',
        'label': '数据',
        'summary': '选择公开数据集或本地文件。格式需能被 ms-swift AutoPreprocessor 接受：messages、sharegpt、alpaca 或 query/response。',
    },
    {
        'id': 'sft',
        'label': 'SFT',
        'summary': '监督微调。参数名与 swift sft 以及 Web-UI「LLM预训练/微调」一致。模型默认是本轮基座。',
    },
    {
        'id': 'sft_eval',
        'label': 'SFT 评测',
        'summary': '评测 SFT checkpoint。参数名与 swift eval / Web-UI 评测页一致。Mock 返回可复现分数，不会调用 EvalScope。',
    },
    {
        'id': 'rl',
        'label': '在线 RL',
        'summary': '默认 GRPO（swift rlhf --rlhf_type grpo）。策略从本轮 SFT checkpoint 初始化。单卡 A100 使用 vLLM colocate。',
    },
    {
        'id': 'rl_eval',
        'label': 'RL 评测',
        'summary': '用同一套评测配置检查 RL checkpoint，便于和 SFT 分数对比。',
    },
    {
        'id': 'promote',
        'label': '晋级',
        'summary': '比较当前基座、SFT、RL。分数更高的检查点会作为建议，你可以改选，包括退回基座。',
    },
    {
        'id': 'next_round',
        'label': '下一轮',
        'summary': '用晋级模型作为新的基座，回到数据阶段。上一轮记录保留在审计里。',
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

BUNDLED = [
    {
        'id': 'itops_mix',
        'label': '内置示例：IT 运维工单（messages + solution）'
    },
]


def _field(key, label, arg, type='text', group='基础', help='', choices=None, advanced=False, step=None, source=None):
    item = {
        'key': key,
        'label': label,
        'arg': arg,
        'type': type,
        'group': group,
        'help': help,
        'advanced': advanced,
    }
    if choices is not None:
        item['choices'] = choices
    if step is not None:
        item['step'] = step
    if source:
        item['source'] = source
    return item


def _model_fields(checkpoint=False):
    """Web-UI 模型设置 labels. Rendered by the V1 page as themed comboboxes."""
    model_help = ('实际的模型id，如果是训练后的模型请填入checkpoint-xxx的目录' if checkpoint else '实际的模型id')
    type_label = '选择模型类型' if checkpoint else '模型类型'
    return [
        _field('model', '模型id或路径', '--model', 'combo', '模型设置', model_help, source='models'),
        _field('model_type', type_label, '--model_type', 'combo', '模型设置', 'SWIFT已支持的模型类型', source='model_types'),
        _field('template', '模型Prompt模板类型', '--template', 'combo', '模型设置', '选择匹配模型的Prompt模板',
               source='templates'),
    ]


FIELDS = {
    'data': [
        _field(
            'source',
            '数据来源',
            '',
            'select',
            '数据',
            '公开数据只记录 id；本地文件会做格式校验。',
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
        _field('bundled', '内置数据集', '', 'select', '数据', '仓库内的小型 jsonl，不需要下载。',
               [{
                   'value': item['id'],
                   'label': item['label']
               } for item in BUNDLED]),
        _field('dataset', '数据集 id 或路径', '--dataset', 'text', '数据', '公开数据填 org/name，本地数据填文件路径。'),
        _field('split_dataset_ratio', '验证集拆分比例', '--split_dataset_ratio', 'number', '数据',
               '本闭环用独立评测阶段，训练侧默认不划验证集。', step='0.01'),
        _field('max_length', '句子最大长度', '--max_length', 'number', '数据', '与 Web-UI 的 max_length 相同。', step='1'),
        _field('dataset_num_proc', '数据集进程数', '--dataset_num_proc', 'number', '资源', '18 核机器上预设为 8，给 vLLM 留出核心。',
               step='1'),
        _field(
            'dataloader_num_workers', 'DataLoader workers', '--dataloader_num_workers', 'number', '资源', '预设 4。',
            step='1'),
        _field('padding_free', '无填充批处理', '--padding_free', 'checkbox', '资源', '对应 Web-UI padding_free。'),
        _field('val_dataset', '验证集', '--val_dataset', 'text', '数据', '可选。', advanced=True),
    ],
    'sft': [
        *_model_fields(checkpoint=False),
        _field('system', 'System字段', '--system', 'textarea', '模型设置', '选择system字段的内容'),
        _field('torch_dtype', '训练精度', '--torch_dtype', 'select', '模型设置', 'A100 上 7B LoRA 使用 bfloat16。',
               ['bfloat16', 'float16', 'float32']),
        _field('seed', '随机种子', '--seed', 'number', '模型设置', '', step='1'),
        _field('tuner_type', '训练方式', '--tuner_type', 'select', '调参', '单卡 80G 建议 lora。全参 Adam 放不进约 60GB 可用内存。',
               ['lora', 'full', 'adalora', 'dora', 'longlora']),
        _field('lora_rank', 'LoRA rank', '--lora_rank', 'number', '调参', '', step='1'),
        _field('lora_alpha', 'LoRA alpha', '--lora_alpha', 'number', '调参', '', step='1'),
        _field('lora_dropout', 'LoRA dropout', '--lora_dropout', 'number', '调参', '', step='0.01'),
        _field('target_modules', 'Target modules', '--target_modules', 'text', '调参', '常用 all-linear。'),
        _field('use_dora', '使用 DoRA', '--use_dora', 'checkbox', '调参', ''),
        _field('use_rslora', '使用 rsLoRA', '--use_rslora', 'checkbox', '调参', ''),
        _field('num_train_epochs', '训练轮次', '--num_train_epochs', 'number', '超参数', '', step='0.1'),
        _field('learning_rate', '学习率', '--learning_rate', 'text', '超参数', 'LoRA 常用 1e-4，全参常用 1e-5。'),
        _field('per_device_train_batch_size', '训练 batch size', '--per_device_train_batch_size', 'number', '超参数', '',
               step='1'),
        _field('per_device_eval_batch_size', '验证 batch size', '--per_device_eval_batch_size', 'number', '超参数', '',
               step='1'),
        _field('gradient_accumulation_steps', '梯度累积', '--gradient_accumulation_steps', 'number', '超参数',
               '7B LoRA 预设 16，用来换有效 batch。', step='1'),
        _field('gradient_checkpointing', '梯度检查点', '--gradient_checkpointing', 'checkbox', '显存', '用计算换显存。'),
        _field('warmup_ratio', 'Warmup 比例', '--warmup_ratio', 'number', '超参数', '', step='0.01'),
        _field('lr_scheduler_type', '学习率调度', '--lr_scheduler_type', 'text', '超参数', ''),
        _field('weight_decay', '权重衰减', '--weight_decay', 'number', '超参数', '', advanced=True, step='0.01'),
        _field('max_length', '最大长度', '--max_length', 'number', '超参数', '', step='1'),
        _field('logging_steps', '日志步数', '--logging_steps', 'number', '超参数', '', step='1'),
        _field('eval_steps', '评估步数', '--eval_steps', 'number', '超参数', '', step='1'),
        _field('save_steps', '保存步数', '--save_steps', 'number', '超参数', '', step='1'),
        _field('save_total_limit', '最多保留检查点', '--save_total_limit', 'number', '超参数', '', step='1'),
        _field('split_dataset_ratio', '验证集比例', '--split_dataset_ratio', 'number', '超参数', '', advanced=True,
               step='0.01'),
        _field('dataset_num_proc', '数据集进程数', '--dataset_num_proc', 'number', '资源', '', advanced=True, step='1'),
        _field(
            'dataloader_num_workers',
            'DataLoader workers',
            '--dataloader_num_workers',
            'number',
            '资源',
            '',
            advanced=True,
            step='1'),
        _field('load_from_cache_file', '使用数据集缓存', '--load_from_cache_file', 'checkbox', '资源', '', advanced=True),
        _field('padding_free', '无填充批处理', '--padding_free', 'checkbox', '显存', '', advanced=True),
        _field('neftune_noise_alpha', 'NEFTune 噪声', '--neftune_noise_alpha', 'number', '超参数', '0 表示关闭。常用 5 或 10。',
               advanced=True, step='0.5'),
        _field('attn_impl', 'Attention 实现', '--attn_impl', 'select', '显存', '没装 flash-attn 时用 sdpa。',
               ['sdpa', 'eager', 'flash_attention_2', 'flash_attention_3'], advanced=True),
        _field('use_liger_kernel', 'Liger kernel', '--use_liger_kernel', 'checkbox', '显存', '可降低显存，需要额外安装。',
               advanced=True),
        _field('deepspeed', 'DeepSpeed', '--deepspeed', 'select', '显存', '单卡 LoRA 通常不需要。', ['', 'zero2', 'zero3'],
               advanced=True),
        _field('sequence_parallel_size', '序列并行', '--sequence_parallel_size', 'number', '显存', '单卡保持 1。', advanced=True,
               step='1'),
        _field('quant_method', '量化方式', '--quant_method', 'select', '显存', '14B 可改为 bnb 做 QLoRA。留空表示不量化。',
               ['', 'bnb', 'awq', 'gptq', 'hqq'], advanced=True),
        _field('quant_bits', '量化位数', '--quant_bits', 'number', '显存', '配合 quant_method=bnb。', advanced=True, step='1'),
        _field('dataset', '数据集', '--dataset', 'text', '数据', '默认沿用本轮数据阶段的结果。', advanced=True),
    ],
    'sft_eval': [
        *_model_fields(checkpoint=True),
        _field('eval_backend', '评测后端', '--eval_backend', 'select', '评测', '与 Web-UI 评测页一致。',
               ['Native', 'OpenCompass', 'VLMEvalKit']),
        _field('eval_dataset', '评测数据集', '--eval_dataset', 'text', '评测',
               'Mock 接受任意名字。真实 EvalScope 请改成 gsm8k、ceval 等已支持的数据集。'),
        _field('eval_limit', '每个数据集取样数', '--eval_limit', 'number', '评测', '', step='1'),
        _field('custom_eval_config', '自定义评测配置', '--extra_eval_args', 'text', '评测',
               '对应 Web-UI 的 custom_eval_config，提交时写入 extra_eval_args。'),
        _field('temperature', '温度', '--temperature', 'number', '生成', '评测默认 0。', step='0.1'),
        _field('max_new_tokens', '最大生成长度', '--max_new_tokens', 'number', '生成', '', step='1'),
        _field('eval_num_proc', '并发数', '--eval_num_proc', 'number', '资源', '单卡上不必开到 Web-UI 默认的 16。', step='1'),
        _field('infer_backend', '推理框架', '--infer_backend', 'select', '资源', '评测阶段用 transformers，把显存留给训练。',
               ['transformers', 'vllm', 'sglang', 'lmdeploy']),
        _field('verbose', '详细日志', '--verbose', 'checkbox', '评测', '', advanced=True),
    ],
    'rl': [
        _field('rlhf_type', 'RLHF 类型', '--rlhf_type', 'select', '算法', '在线 RL 预设 grpo。DPO/KTO 等是离线偏好学习。',
               ['grpo', 'dpo', 'kto', 'cpo', 'simpo', 'orpo', 'ppo', 'gkd', 'rm']),
        *_model_fields(checkpoint=False),
        _field('dataset', '数据集', '--dataset', 'text', '数据', '默认沿用数据阶段。GRPO 需要 solution 或可评分字段。'),
        _field('reward_funcs', '奖励函数', '--reward_funcs', 'text', '奖励', '空格分隔。内置示例注册了 itops_checklist。'),
        _field('reward_weights', '奖励权重', '--reward_weights', 'text', '奖励', '与奖励函数一一对应，空格分隔。'),
        _field('external_plugins', '外部插件', '--external_plugins', 'text', '奖励', 'swift rlhf 会加载这个文件。Mock 不导入它。'),
        _field('num_generations', '采样数量 G', '--num_generations', 'number', '采样', '单卡 80G 预设 4。显存富余再升到 8。',
               step='1'),
        _field('max_completion_length', '最大生成长度', '--max_completion_length', 'number', '采样', '', step='1'),
        _field('temperature', '采样温度', '--temperature', 'number', '采样', '', step='0.1'),
        _field('top_p', 'Top-p', '--top_p', 'number', '采样', '', step='0.01'),
        _field('top_k', 'Top-k', '--top_k', 'number', '采样', '', step='1'),
        _field('beta', 'KL 系数', '--beta', 'number', '算法', 'GRPO 常用 0.04。', step='0.01'),
        _field('epsilon', 'clip epsilon', '--epsilon', 'number', '算法', '', advanced=True, step='0.01'),
        _field('epsilon_high', 'epsilon high', '--epsilon_high', 'number', '算法', '', advanced=True, step='0.01'),
        _field('num_iterations', 'num_iterations', '--num_iterations', 'number', '算法', '', advanced=True, step='1'),
        _field('learning_rate', '学习率', '--learning_rate', 'text', '超参数', 'GRPO LoRA 常用 1e-6。'),
        _field('num_train_epochs', '训练轮次', '--num_train_epochs', 'number', '超参数', '', step='0.1'),
        _field('per_device_train_batch_size', '训练 batch size', '--per_device_train_batch_size', 'number', '超参数', '',
               step='1'),
        _field('gradient_accumulation_steps', '梯度累积', '--gradient_accumulation_steps', 'number', '超参数', '', step='1'),
        _field('tuner_type', '训练方式', '--tuner_type', 'select', '调参', '', ['lora', 'full']),
        _field('lora_rank', 'LoRA rank', '--lora_rank', 'number', '调参', '', step='1'),
        _field('lora_alpha', 'LoRA alpha', '--lora_alpha', 'number', '调参', '', step='1'),
        _field('target_modules', 'Target modules', '--target_modules', 'text', '调参', ''),
        _field('max_length', '最大长度', '--max_length', 'number', '超参数', '', step='1'),
        _field('torch_dtype', '训练精度', '--torch_dtype', 'select', '模型', '', ['bfloat16', 'float16', 'float32']),
        _field('use_vllm', '使用 vLLM', '--use_vllm', 'checkbox', '显存', 'GRPO 采样后端。'),
        _field('vllm_mode', 'vLLM 模式', '--vllm_mode', 'select', '显存', '只有一张卡时用 colocate，不要用 server 模式占第二张卡。',
               ['colocate', 'server']),
        _field('vllm_gpu_memory_utilization', 'vLLM 显存利用率', '--vllm_gpu_memory_utilization', 'number', '显存',
               'colocate 预设 0.45，给训练留显存。', step='0.01'),
        _field('vllm_tensor_parallel_size', '张量并行', '--vllm_tensor_parallel_size', 'number', '显存', '单卡为 1。', step='1'),
        _field('vllm_max_model_len', 'vLLM 最大长度', '--vllm_max_model_len', 'number', '显存', '', step='1'),
        _field('vllm_enable_lora', 'vLLM 加载 LoRA', '--vllm_enable_lora', 'checkbox', '显存', 'LoRA 训练时建议打开。'),
        _field('vllm_max_lora_rank', 'vLLM LoRA rank 上限', '--vllm_max_lora_rank', 'number', '显存', '需要 ≥ lora_rank。',
               step='1'),
        _field('sleep_level', 'Sleep level', '--sleep_level', 'number', '显存', '训练时释放 vLLM 显存。单卡预设 1。', step='1'),
        _field('offload_model', '卸载模型', '--offload_model', 'checkbox', '显存', 'vLLM 推理时把训练模型卸到 CPU。'),
        _field('offload_optimizer', '卸载优化器', '--offload_optimizer', 'checkbox', '显存', ''),
        _field('gradient_checkpointing', '梯度检查点', '--gradient_checkpointing', 'checkbox', '显存', ''),
        _field('log_completions', '记录采样', '--log_completions', 'checkbox', '日志', ''),
        _field('seed', '随机种子', '--seed', 'number', '模型', '', advanced=True, step='1'),
        _field('system', 'System', '--system', 'textarea', '模型', '', advanced=True),
        _field('deepspeed', 'DeepSpeed', '--deepspeed', 'select', '显存', '单卡 LoRA 通常留空。', ['', 'zero2', 'zero3'],
               advanced=True),
        _field('quant_method', '量化方式', '--quant_method', 'select', '显存', '显存不够时改为 bnb。', ['', 'bnb', 'awq', 'gptq'],
               advanced=True),
        _field('quant_bits', '量化位数', '--quant_bits', 'number', '显存', '', advanced=True, step='1'),
        _field('ref_model', 'Ref 模型', '--ref_model', 'text', '算法', 'DPO/KTO/PPO 需要。GRPO 可留空。', advanced=True),
        _field('loss_scale', '损失权重', '--loss_scale', 'text', '算法', '', advanced=True),
        _field('dataset_num_proc', '数据集进程数', '--dataset_num_proc', 'number', '资源', '', advanced=True, step='1'),
        _field(
            'dataloader_num_workers',
            'DataLoader workers',
            '--dataloader_num_workers',
            'number',
            '资源',
            '',
            advanced=True,
            step='1'),
        _field('split_dataset_ratio', '验证集比例', '--split_dataset_ratio', 'number', '数据', '', advanced=True, step='0.01'),
        _field('load_from_cache_file', '使用数据集缓存', '--load_from_cache_file', 'checkbox', '资源', '', advanced=True),
        _field('logging_steps', '日志步数', '--logging_steps', 'number', '日志', '', advanced=True, step='1'),
        _field('save_steps', '保存步数', '--save_steps', 'number', '日志', '', advanced=True, step='1'),
        _field('save_total_limit', '最多保留检查点', '--save_total_limit', 'number', '日志', '', advanced=True, step='1'),
        _field('warmup_ratio', 'Warmup 比例', '--warmup_ratio', 'number', '超参数', '', advanced=True, step='0.01'),
    ],
    'rl_eval': [
        *_model_fields(checkpoint=True),
        _field('eval_backend', '评测后端', '--eval_backend', 'select', '评测', '',
               ['Native', 'OpenCompass', 'VLMEvalKit']),
        _field('eval_dataset', '评测数据集', '--eval_dataset', 'text', '评测', '与 SFT 评测使用同一名字，方便对比。'),
        _field('eval_limit', '每个数据集取样数', '--eval_limit', 'number', '评测', '', step='1'),
        _field('custom_eval_config', '自定义评测配置', '--extra_eval_args', 'text', '评测', ''),
        _field('temperature', '温度', '--temperature', 'number', '生成', '', step='0.1'),
        _field('max_new_tokens', '最大生成长度', '--max_new_tokens', 'number', '生成', '', step='1'),
        _field('eval_num_proc', '并发数', '--eval_num_proc', 'number', '资源', '', step='1'),
        _field('infer_backend', '推理框架', '--infer_backend', 'select', '资源', '',
               ['transformers', 'vllm', 'sglang', 'lmdeploy']),
        _field('verbose', '详细日志', '--verbose', 'checkbox', '评测', '', advanced=True),
    ],
    'promote': [
        _field('reason', '晋级说明', '', 'textarea', '决策', '会写进审计。例如：RL accuracy 高于 SFT，采纳 RL。'),
    ],
    'next_round': [
        _field('reuse_dataset', '沿用上一轮数据', '', 'checkbox', '下一轮', '只预填表单，仍需要你在数据阶段确认。'),
        _field('note', '备注', '', 'textarea', '下一轮', '可选。'),
    ],
    'serve': [
        *_model_fields(checkpoint=True),
        _field('served_model_name', '服务模型名', '--served_model_name', 'text', '部署', 'OpenAI 请求里的 model 字段。'),
        _field('infer_backend', '推理框架', '--infer_backend', 'select', '部署', 'A100 上真实部署用 vllm。',
               ['vllm', 'transformers', 'sglang', 'lmdeploy']),
        _field('host', 'Host', '--host', 'text', '部署', '本机 Mock 用 127.0.0.1。对外服务再用 0.0.0.0。'),
        _field('port', 'Port', '--port', 'number', '部署', '与 swift deploy 示例一致时为 8000。', step='1'),
        _field('api_key', 'API key', '--api_key', 'text', '部署', '留空则不校验。'),
        _field('torch_dtype', '精度', '--torch_dtype', 'select', '部署', '', ['bfloat16', 'float16', 'float32']),
        _field('max_new_tokens', '最大生成长度', '--max_new_tokens', 'number', '生成', '', step='1'),
        _field('temperature', '温度', '--temperature', 'number', '生成', '', step='0.1'),
        _field('vllm_gpu_memory_utilization', 'vLLM 显存利用率', '--vllm_gpu_memory_utilization', 'number', '显存',
               '纯推理可以给到 0.9。', advanced=True, step='0.01'),
        _field('vllm_max_model_len', 'vLLM 最大长度', '--vllm_max_model_len', 'number', '显存', '', advanced=True, step='1'),
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


def public_schema() -> dict:
    preset = presets()
    return {
        'stages': STAGE_META,
        'stage_order': STAGE_ORDER,
        'fields': FIELDS,
        'presets': {key: preset.get(key, {})
                    for key in ('data', 'sft', 'sft_eval', 'rl', 'rl_eval', 'serve', 'next_round')},
        'public_datasets': PUBLIC_DATASETS,
        'bundled': BUNDLED,
        'hardware': preset.get('hardware', {}),
        'base_model_default': 'Qwen/Qwen2.5-7B-Instruct',
        'catalog': load_catalog(),
    }
