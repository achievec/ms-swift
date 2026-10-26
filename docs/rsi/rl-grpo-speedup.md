# RSI GRPO RL 训练提速分析

基于一次完整 `swift.log` 的量化结论：墙钟约 2.17 s/it **不是**主要瓶颈；约 89% 的 step 上 `reward=1` 且 `reward_std=0`（组内无 GRPO advantage），同时 Format reward 恒为 0。

**分析日志**：`/Users/liyuhao03/test/20261019_rsi/swift.log`（RSI 驱动的 GRPO / `swift rlhf --rlhf_type grpo`，vLLM colocate）。

---

## 1. 背景与现象

场景：ITOps 短答风格数据 + `itops_checklist` + 内置 `format`，7B LoRA，单卡 A100 级显存，RSI 默认 GRPO 预设（`num_generations=4`，`max_completion_length=1024` 等）。

肉眼可见的现象：

- 训练条进度正常前进，单步约 2 s，显存稳定，无 OOM。
- 绝大多数 step 的 `reward` / `rewards/ITOpsChecklist/mean` 顶满为 1，`reward_std` 与 `frac_reward_zero_std` 显示组内无区分度。
- `rewards/Format/mean` 全程为 0（与 weight 0.5 并存，等于常驻无效项）。
- completion 极短（十几 token），却仍按 `max_completion_length=1024` 预留生成预算。

结论先说：**先修样本效率（advantage / 奖励设计），再抠墙钟**；只把 s/it 压下去，有效学习步数仍然很少。

---

## 2. 日志量化

统计口径：日志中已打印的 train metrics（进度约到 334/1000）。

| 指标 | 观测值 |
| --- | --- |
| 进度 | 334 / 1000 |
| 已用时 / 剩余 ETA | ~12 min / 剩余约 ~24 min |
| `train_speed` | ~2.18 s/it |
| `step_time` mean | ~2.14 s |
| 显存 `memory(GiB)` | ~50.04，稳定，无 OOM |
| 数据规模 | `num_rows=1000` |
| 有效 batch | `per_device_train_batch_size=1` × `gradient_accumulation_steps=4` × `num_generations=4` → 约 **1 prompt × 4 generations / step** |
| completion length | mean ~17，max ~33 |
| `completions/clipped_ratio` | 恒为 0（尽管 `max_completion_length=1024`） |
| `reward` mean | ~0.968 |
| `reward == 1` 占比 | ~88.9% |
| `frac_reward_zero_std == 1` 占比 | ~89.2% |
| `rewards/Format/mean` | 恒为 0 |
| `kl` mean | ~2.24 |
| checkpoint | `save_steps=50`；334 step 内约 6 次保存 |

相关命令侧配置（摘自同一日志）：

- `--max_completion_length 1024`
- `--reward_funcs itops_checklist format`，`--reward_weights 1.0 0.5`
- `--log_completions true`
- `--vllm_enable_prefix_caching false`
- `--dynamic_sample false`

---

## 3. 根因

### 3.1 样本效率：组内无 advantage

GRPO 用同 prompt 下多条 generation 的相对奖励做 advantage。当组内奖励几乎全是 1（或方差为 0）时：

- `reward_std=0` / `frac_reward_zero_std=1` → **该 step 对 policy 几乎无有效梯度方向**（或仅剩 KL / 数值噪声）。
- ~89% step 处于这种饱和态 → 墙钟在跑，但有效 RL 信号很少。

更可能的数据/奖励原因：

1. **任务过易**：SFT 后 checklist 已接近满分，短答易「全对」。
2. **奖励过松**：`itops_checklist` 对当前输出过于宽容，组内拉不开差距。
3. **未做零方差过滤**：`dynamic_sample=false`，零 std 组仍计入训练循环。

### 3.2 Format reward 失效且与短答冲突

- 日志里 Format mean **始终为 0**，却仍占 `reward_weights` 的 0.5。
- ITOps 短答通常不走「带标签的长推理格式」；Format 与业务目标冲突时，要么永远为 0（浪费），要么把策略往错误格式拉。
- **不要指望靠把 Format 权重调到 0.5 来「顺便」学格式**；要么修好与数据匹配的 format 规则，要么从 `reward_funcs` 中去掉。

### 3.3 墙钟：预算虚高，吞吐未吃满

- 实际 completion ≪ 1024，`clipped_ratio=0` → `max_completion_length=1024` 主要增加 **vLLM 预分配 / 调度预算与开销**，对当前短答收益很小。
- batch=1、gas=4、G=4 → 每步有效 prompt 很少；显存仍有余量（~50 GiB / 80 GiB 级），可适度加大 batch/gas（需保持可被 `num_generations` 整除）。
- `log_completions=true`、`save_steps=50`、未开 prefix caching，都会叠加 IO / 同步成本；单卡 colocate 下 server 模式 vLLM 需多卡才划算。

---

## 4. 提速建议（按优先级）

### A. 墙钟（降低 s/it、缩短总 wall time）

1. **`max_completion_length`：1024 → 64~128**  
   与观测 max~33 对齐，留余量即可；优先改，风险低。
2. **提高 `per_device_train_batch_size` / `gradient_accumulation_steps`**  
   保持 `generation_batch`（或等价组合）能被 `num_generations` 整除；在显存允许下提高每步有效样本。
3. **开启 vLLM prefix caching**（`--vllm_enable_prefix_caching true`）  
   prompt 重复度高时收益明显。
4. **关闭 `log_completions`**（训练稳态阶段）  
   减少逐条写盘与日志膨胀。
5. **`save_steps`：50 → 100 或 200**  
   降低 checkpoint IO（已有 `save_total_limit` 时仍建议拉开间隔）。
6. **若有多卡**：考虑 server 模式独立 vLLM；单卡继续 colocate。

### B. 样本效率（提高「有效」RL 步比例）

1. **换更难数据或更严奖励**  
   提高组内区分度，降低 `reward==1` 与零 std 占比。
2. **启用 `dynamic_sample` 或过滤零 std 组**  
   避免 ~89% 的无效 step 空转。
3. **移除或修复 Format reward**  
   与短答 ITOps 冲突时不要保留 weight 0.5；修好规则或直接去掉 `format`。
4. **饱和时早停**  
   监控 `frac_reward_zero_std` / reward mean；长期顶满则缩短 epoch / max_steps，把算力留给更难集。
5. **不要依赖 Format weight=0.5**  
   权重不等于有效信号；零均值的 Format 只会稀释或干扰主奖励。

**建议顺序**：先做 B（否则 A 只是更快地空转）→ 再做 A 的 1/3/4/5 → 最后视显存调 batch/gas 与多卡。

---

## 5. 建议的默认 / 预设改动（可选）

面向 RSI `a100_80g` 一类 GRPO 预设，可考虑将默认值调成更贴近短答 ITOps 的配置（需改表单默认或 YAML，并跑一轮回归）：

| 项 | 现状（本次日志） | 建议默认 |
| --- | --- | --- |
| `max_completion_length` | 1024 | 128（或 64） |
| `log_completions` | true | false（调试时再开） |
| `save_steps` | 50 | 100~200 |
| `vllm_enable_prefix_caching` | false | true |
| `reward_funcs` / weights | `itops_checklist,format` + `1.0,0.5` | 默认仅 `itops_checklist`；Format 改为显式可选 |
| `dynamic_sample` | false | 对易饱和任务默认 true，或文档标明需人工打开 |
| batch / gas | 1 × 4 | 在整除约束下略增（例如 batch=2 或 gas 上调），以显存为准 |

改预设后应用同一类日志再比一次：`train_speed`、`frac_reward_zero_std`、`reward` mean、Format mean、completion length。**成功标准**应同时看墙钟与「非零 std 步占比」，而不是只看 s/it。

---

## 6. 一句话回顾

本次 run 的主要浪费不是 2 s/it，而是约九成 step 没有 GRPO advantage，再加上永远为 0 的 Format 与虚高的 1024 completion 预算。先让奖励和数据重新产生组内方差，再收紧生成长度与 IO，墙钟和样本效率才会一起上来。
