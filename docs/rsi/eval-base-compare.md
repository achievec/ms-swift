# 评测阶段的基座对比

`sft_eval` / `rl_eval` 默认勾选 **同时评测基座**（`eval_base`）。

## 行为

1. 先按表单评测本阶段检查点（LoRA 走 `--adapters`，全量检查点走 `--model`）。
2. 若 `eval_base` 为真，再用**同一套** `eval_dataset` / `eval_limit` / `eval_backend` 等参数评测实验 `base_model`，且 **不带 adapters**。
3. 基座输出写在本阶段 `eval_output/base/`；阶段检查点仍写在 `eval_output/`（报告 HTML / JSON 与原先一致）。
4. 步骤结果里的 `compare` 是小表数据（分数 + 报告路径），不把 HTML 塞进 `/api/runs`：
   - `sft_eval`：基座 / SFT
   - `rl_eval`：基座 /（若本轮已有成功的 `sft_eval`）SFT / RL  
     SFT 一行只复用已有评测结果，不再跑一遍。

关闭勾选则只评本阶段检查点。若当前已经在评「无 adapter 的基座本身」，会自动跳过第二趟。

## 页面

阶段结果区会显示「基座对比」表，并可分别打开基座 / SFT / RL 的完整报告（`/report?kind=base|sft|rl`）。
