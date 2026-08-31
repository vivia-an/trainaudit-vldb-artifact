# Template Induction Experiment — 完成

目录：`benchmark/eval/template_induction/`
报告：**`report/experiment_report.md`**（先读这个）
协议：`template_coding_protocol.md`（2026-07-16 冻结）
冻结 catalog：`frozen_template_catalog.json`（sha256 见 `.sha256`，35 templates / 24 singletons）

## Pipeline 全部完成

- [x] S0 协议冻结
- [x] S1 数据划分（seed 128 / dev 186 / test 78；Real-SE 18 例强制入 test；RNG=20260716）
- [x] S2 E1 抽取 128+186，校验通过
- [x] S3 E2 复标 79（25%）→ relation κ=0.600 / phase κ=0.968 / topology κ=0.834
- [x] S4 Seed 归纳 → 27 templates，覆盖 119/128
- [x] S5 Dev 8 批饱和 → 35 templates，累计覆盖 0.876；**停止规则从未触发**（无 stopping point，regret 不可计算）
- [x] S6 Catalog 冻结（stream 耗尽处冻结）
- [x] S7 Rater A/B 双盲 78 例 + adjudication（7 例分歧）
- [x] S8 指标：C_schema=0.923 / C_joint=0.910 / novelty=0.038；template assignment κ=0.954
- [x] S9 Merge audit：595 对 → 23 候选 → **0 可合并**
- [x] S10 Uncovered：U1×1 / U4×1 / U5×3 / U6×2
- [x] S11 重排模拟（200 orders）→ `analysis/saturation_curve.pdf`
- [x] S12 `report/experiment_report.md`

## 三个核心结论

1. **可复现**：template assignment κ=0.954（两位独立、框架不同的 rater 对冻结 catalog）。
2. **未饱和**：模板数 27→35 且末批仍 +2，预注册停止条件全程未触发。覆盖率却在第 25 例就稳在 ~0.86。→ 关系词表接近完备，但**划分粒度未收敛**（尾部不断裂出 2-case 模板）。**不要声称 saturated。**
3. **可泛化**：held-out C_schema=92.3%，novelty 3.8%，0 个新 recurring relation；但 OLMo 仅 0.80（seed 池偏 Megatron/DeepSpeed）。

## 对论文的影响（见报告 §12）

- 本实验独立归纳出 **35 templates**，与论文 P1–P16 不一致 → 需做 **mapping study**：检查 35 个是否都能表示为 P-template + guards。
- 论文**不应**加入 saturation 声明。
- κ=0.954 是新的、强的可复现性证据（区别于 taxonomy κ=0.566）。
- 数据修复：补 `fix_date`（可解锁 temporal split）、修 ~11 条 `root_cause` 为空/复制 `description` 的记录。
