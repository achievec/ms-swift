# 评测阶段的基座对比与自选对比

`sft_eval` / `rl_eval` 默认勾选 **同时评测基座**（`eval_base`）。此外还提供 **自选对比**：在同实验内勾选任意已有评测报告，拼成指标对照表。

## 自动基座对比（`eval_base`）

1. 先按表单评测本阶段检查点（LoRA 走 `--adapters`，全量检查点走 `--model`）。
2. 若 `eval_base` 为真，再用**同一套** `eval_dataset` / `eval_limit` / `eval_backend` 等参数评测实验 `base_model`，且 **不带 adapters**。
3. 基座输出写在本阶段 `eval_output/base/`；阶段检查点仍写在 `eval_output/`（报告 HTML / JSON 与原先一致）。
4. 步骤结果里的 `compare` 是小表数据（分数 + 报告路径），不把 HTML 塞进 `/api/runs`：
   - `sft_eval`：基座 / SFT
   - `rl_eval`：基座 /（若本轮已有成功的 `sft_eval`）SFT / RL  
     SFT 一行只复用已有评测结果，不再跑一遍。

关闭勾选则只评本阶段检查点。若当前已经在评「无 adapter 的基座本身」，会自动跳过第二趟。

若 `swift eval` 进程非 0 退出，但 `eval_output`（或基座的 `eval_output/base`）下已有 evalscope 写出的 `report.html` / 分数 JSON，RSI 仍按**成功**处理该趟（日志里会注明退出码），以便继续跑基座、并在界面保留报告链接。基座趟若硬失败，本阶段检查点结果仍会保留，并在结果区提示基座错误。

### 页面（自动表）

阶段结果区会显示 **基座对比** 表：

| 指标 | 基座 | SFT | RL |
|------|------|-----|----|
| Accuracy / ROUGE-L F1 / ROUGE-1 F1 / BLEU-1… | … | … | … |

- 缺席的列会省略（例如还没跑 RL、或未勾选基座时只有单列则不画对照表）。
- 数值行会用浅色标出最优 / 最弱（越高越好）。
- 表下可分别打开 **基座 / SFT / RL** 的完整 HTML 报告：`/api/runs/.../stages/{sft_eval|rl_eval}/report?kind=base|sft|rl`（`kind` 空则本阶段检查点报告）。

## 自选对比（灵活挑选）

自动基座表只覆盖「本阶段刚跑完的那一组」。若要和同实验里其它报告比（例如本轮 RL vs 本轮 SFT、或历史重跑、或其它轮次），用结果区下方的 **自选对比**：

1. 选择范围：**本轮** 或 **全部轮次**（含 `superseded` 历史步骤）。
2. 勾选两份及以上报告（芯片上可看 acc / BLEU-1 摘要，并可打开单份 HTML）。
3. 点 **对比** → 指标 × 所选报告 的对照表（同样标最优 / 最弱，并附报告链接）。

列表与分数来自独立接口，**不**塞进 `/api/runs/{id}`：

```http
GET /api/runs/{id}/eval-reports
GET /api/runs/{id}/eval-reports?round={n}
```

每条报告大致包含：`id`、`step_id`、`round_index`、`stage`、`role`、`label`、`status`、`metrics`（紧凑分数）、`report_url`、`has_report`。

打开某份 HTML 时可带历史步骤：`.../report?round=&kind=&step=`（`step` 指向具体 step id，含已作废重跑）。
