# Copyright (c) ModelScope Contributors. All rights reserved.
"""Chinese labels and help copied from the ms-swift Gradio Web-UI locales."""

# key -> (label, info). Info None keeps the schema help.
COMMON = {
    'model': ('模型id或路径', '实际的模型id'),
    'model_type': ('模型类型', 'SWIFT已支持的模型类型'),
    'template': ('模型Prompt模板类型', '选择匹配模型的Prompt模板'),
    'system': ('System字段', '选择system字段的内容'),
    'dataset': ('数据集名称', '选择训练的数据集，支持复选/本地路径'),
    'max_length': ('句子最大长度', '设置输入模型的最大长度'),
    'split_dataset_ratio': ('验证集拆分比例', '表示将总数据的多少拆分到验证集中'),
    'padding_free': ('无填充批处理', '将一个batch中的数据进行展平而避免数据padding'),
    'val_dataset': ('验证集', '可选的验证集名称或本地路径'),
    'tuner_type': ('训练方式', '选择训练的方式'),
    'seed': ('随机数种子', '选择随机数种子'),
    'torch_dtype': ('训练精度', '选择训练精度'),
    'use_liger_kernel': ('使用Liger kernel', 'Liger kernel可以有效降低显存使用'),
    'sequence_parallel_size': ('序列并行大小', '当前支持CPT/SFT/DPO/GRPO'),
    'deepspeed': ('DeepSpeed', '可以选择下拉列表，也支持传入路径'),
    'per_device_train_batch_size': ('训练batch size', '训练的batch size'),
    'per_device_eval_batch_size': ('验证batch size', '验证的batch size'),
    'learning_rate': ('学习率', '设置学习率'),
    'eval_steps': ('交叉验证步数', '设置每隔多少步数进行一次验证'),
    'num_train_epochs': ('数据集迭代轮次', '设置对数据集训练多少轮次'),
    'gradient_accumulation_steps': ('梯度累计步数', '设置梯度累计步数以减小显存占用'),
    'attn_impl': ('Flash Attention类型', '没装 flash-attn 时用 sdpa'),
    'neftune_noise_alpha': ('NEFTune噪声系数', '使用NEFTune提升训练效果, 一般设置为5或者10'),
    'save_steps': ('存储步数', '设置每个多少步数进行存储'),
    'output_dir': ('存储目录', '设置输出模型存储在哪个文件夹下'),
    'lora_rank': ('LoRA的秩', 'LoRA 的秩。单卡 80G 预设 8'),
    'lora_alpha': ('LoRA的缩放因子', 'LoRA 的缩放因子'),
    'lora_dropout': ('LoRA的丢弃概率', 'LoRA 的丢弃概率'),
    'use_rslora': ('使用rsLoRA', ''),
    'use_dora': ('使用DoRA', ''),
    'target_modules': ('指定tuner模块', ''),
    'quant_method': ('量化方式', '如果指定了量化位数，本参数默认为bnb。留空表示不量化'),
    'quant_bits': ('量化bit数', '设置量化bit数, 0代表不进行量化'),
    'logging_steps': ('日志打印步数', '设置日志打印的步数间隔'),
    'lr_scheduler_type': ('LrScheduler类型', '设置LrScheduler类型'),
    'warmup_ratio': ('学习率warmup比例', '设置学习率warmup比例'),
    'weight_decay': ('权重衰减', '设置weight decay'),
    'save_total_limit': ('最多保留检查点', '最多保留多少个 checkpoint'),
    'dataset_num_proc': ('数据集进程数', '18 核机器上预设为 8，给 vLLM 留出核心'),
    'dataloader_num_workers': ('DataLoader workers', '预设 4，给 vLLM 留出核心'),
    'load_from_cache_file': ('使用数据集缓存', ''),
    'gradient_checkpointing': ('梯度检查点', '用计算换显存'),
    'eval_backend': ('评测后端', '选择评测后端'),
    'eval_dataset': ('评测数据集', '选择评测数据集，支持多选 (先选择评测后端)'),
    'eval_limit': ('评测数据个数', '每个评测集的取样数'),
    'eval_output_dir': ('评测输出目录', '评测结果的输出目录'),
    'custom_eval_config': ('自定义数据集评测配置', '可以使用该配置评测自己的数据集，详见github文档的评测部分'),
    'temperature': ('采样温度', ''),
    'max_new_tokens': ('生成序列最大长度', ''),
    'eval_num_proc': ('并发数', '单卡上不必开到 Web-UI 默认的 16'),
    'infer_backend': ('推理框架', ''),
    'verbose': ('详细日志', ''),
    'rlhf_type': ('人类对齐算法类型', '选择对齐算法。在线 RL 预设 grpo'),
    'reward_funcs': ('奖励函数', 'GRPO算法奖励函数'),
    'reward_weights': ('奖励函数权重', '各奖励函数的权重之间用空格隔开'),
    'external_plugins': ('外部插件', 'swift rlhf 会加载这个文件。Mock 不导入它'),
    'num_generations': ('采样数量', '每个prompt采样的数量，即论文中的G值'),
    'max_completion_length': ('最大生成长度', 'GRPO算法中的最大生成长度'),
    'top_p': ('Top-p', ''),
    'top_k': ('Top-k', ''),
    'beta': ('KL正则项系数', 'GRPO 常用 0.04'),
    'use_vllm': ('使用vLLM', '是否使用vLLM作为GRPO生成的推理后端'),
    'vllm_mode': (
        'vLLM集成模式',
        'Server模式使用`swift rollout`拉起的vLLM服务进行采样;Colocate模式使用程序内部署的vLLM。单卡用 colocate',
    ),
    'vllm_gpu_memory_utilization': ('GPU显存利用率', 'vLLM透传参数'),
    'vllm_tensor_parallel_size': ('张量并行大小', 'vLLM透传参数'),
    'vllm_max_model_len': ('模型支持的最大长度', 'vLLM透传参数'),
    'vllm_enable_lora': ('vLLM加载LoRA', 'LoRA 训练时建议打开'),
    'vllm_max_lora_rank': ('vLLM最大LoRA秩', '需要大于等于 lora_rank'),
    'sleep_level': ('Sleep level', '训练时释放vLLM显存'),
    'offload_model': ('卸载模型', 'vLLM 推理时把训练模型卸到 CPU'),
    'offload_optimizer': ('卸载优化器', ''),
    'log_completions': ('记录采样', ''),
    'ref_model': ('Ref模型id或路径', '实际的模型id或路径。GRPO 可留空'),
    'loss_scale': ('损失权重设置', ''),
    'epsilon': ('clip epsilon', ''),
    'epsilon_high': ('epsilon high', ''),
    'num_iterations': ('num_iterations', ''),
    'source': ('数据来源', '公开数据只记录 id；本地文件会做格式校验'),
    'bundled': ('内置数据集', '仓库内的小型 jsonl，不需要下载'),
    'host': ('服务host', 'swift deploy 的 --host。本机 Mock 用 127.0.0.1'),
    'port': ('端口', 'swift deploy 的 --port，示例为 8000'),
    'api_key': ('接口token', '留空则不校验'),
    'served_model_name': ('服务模型名', 'OpenAI 请求里的 model 字段，对应 --served_model_name'),
    'reason': ('晋级说明', '会写进审计。例如：RL accuracy 高于 SFT，采纳 RL'),
    'choice': ('晋级检查点', '分数更高的检查点会作为建议，可以改选，包括退回基座'),
    'reuse_dataset': ('沿用上一轮数据', '只预填表单，仍需要在数据阶段确认'),
    'note': ('备注', '可选'),
}

# Checkpoint pickers use the inference-page wording.
STAGE = {
    ('sft_eval', 'model'): ('模型id或路径', '实际的模型id，如果是训练后的模型请填入checkpoint-xxx的目录'),
    ('rl_eval', 'model'): ('模型id或路径', '实际的模型id，如果是训练后的模型请填入checkpoint-xxx的目录'),
    ('rl', 'model'): ('模型id或路径', '实际的模型id，如果是训练后的模型请填入checkpoint-xxx的目录'),
    ('serve', 'model'): ('模型id或路径', '实际的模型id，如果是训练后的模型请填入checkpoint-xxx的目录'),
    ('sft_eval', 'infer_backend'): ('推理框架', '评测阶段用 transformers，把显存留给训练'),
    ('rl_eval', 'infer_backend'): ('推理框架', '评测阶段用 transformers，把显存留给训练'),
    ('serve', 'infer_backend'): ('推理框架', 'A100 上真实部署用 vllm'),
    ('rl', 'vllm_gpu_memory_utilization'): ('GPU显存利用率', 'vLLM透传参数。colocate 预设 0.45，给训练留显存'),
    ('serve', 'vllm_gpu_memory_utilization'): ('GPU显存利用率', 'vLLM透传参数。纯推理可以给到 0.9'),
    ('sft_eval', 'model_type'): ('选择模型类型', 'SWIFT已支持的模型类型'),
    ('rl_eval', 'model_type'): ('选择模型类型', 'SWIFT已支持的模型类型'),
    ('serve', 'model_type'): ('选择模型类型', 'SWIFT已支持的模型类型'),
}


def label_for(stage: str, key: str, fallback_label: str, fallback_help: str):
    spec = STAGE.get((stage, key)) or COMMON.get(key)
    if not spec:
        return fallback_label, fallback_help
    label, info = spec
    if info is None:
        info = fallback_help
    return label, info


def control_label(label: str, arg: str) -> str:
    """Same suffix the Web-UI monkeypatch appends: 模型id或路径(--model)."""
    if arg and not label.endswith(')'):
        return f'{label}({arg})'
    return label
