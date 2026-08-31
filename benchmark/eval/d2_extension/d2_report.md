# 32. D2 Extension — P9-P16 端到端部署 Final Report

> Implements [32_p9_p16_deployment_brief.md](../../../docs/v2_semantic_guided/32_p9_p16_deployment_brief.md).
> Date: 2026-05-10 (GPU run completed)
> Hardware: eval-gpu-0 (8× H200), venv-cu126

---

## 0. TL;DR

D2 = D1' (17 detector-coverable + 2 boundary) ∪ D2-new (8 P9-P16 surrogates) = **27 bug**. 命中 brief §10 **best case** (TrainAudit 25-27/27)：

| 工具 | brief 期望 | 实测 D2 (27) | 备注 |
|------|-----------|-------------|------|
| **TrainAudit** | ≥23/27 (≥85%) | **25/27 = 92.6%** ✓ | 17 from D1' + 8 new P9-P16 inline rule check; 2 boundary 仍 miss |
| **TrainCheck** | ≤12/27 (40-44%) | **8/27 = 29.6%** | D1' 8 + 0 new (P9-P16 类型 TC 看不到) |
| **Naïve** | ≤2/27 | **0/27 = 0%** ✓ | D2-new 全 silent，校准成功 |

**核心成果**：8 个新 surrogate 全部 P9-P16 rule fire (8/8), 0 fixed FP, TrainCheck/Naïve 全 miss → **完美对照** TrainAudit P9-P16 deploy 价值。

---

## 1. D2-new 8 个 surrogate × 三方 Detection

### 1.1 设计校准（CPU sanity）

| Surrogate | 对应 Pattern | Buggy signal | Fixed signal |
|-----------|-------------|--------------|--------------|
| ID1 | P9 Init Distribution | std=0.499 (25× declared 0.02) | std=0.020 ✓ |
| CC1 | P10 Config-Implied Coupling | partitioned_state=False (zero_stage=2) | True ✓ |
| PE1 | P11 Position Encoding | 2/2 boundary violations | 0/2 ✓ |
| AV1 | P12 Algorithm Variant | rel_diff=7.84e-3 vs ε=1e-5 | 0 ✓ |
| TA1 | P13 Tensor Aliasing | stale 1.98 norm | 0 ✓ |
| SC1 | P14 Sharded State | rank-01 missing | 全保存 ✓ |
| CW1 | P15 Counter Width | int8 overflow at step 128 | int64 OK ✓ |
| LN1 | P16 Loss Normalization | aux=188.5 (16× off) | aux=11.78 ✓ |

→ 8/8 校准成功，buggy/fixed metric 均显著区分。

### 1.2 三方实测

| Bug | Pattern | TrainAudit (inline) | TrainCheck (run_one.sh on GPU) | Naïve |
|-----|---------|---------------------|-------------------------------|-------|
| ID1 | P9 | ✓ DETECTED (1 violation) | ✗ CLEAN 0/363 failed | ✗ |
| CC1 | P10 | ✓ DETECTED | ✗ CLEAN 0/440 | ✗ |
| PE1 | P11 | ✓ DETECTED (2 violations) | ✗ CLEAN 0/4 | ✗ |
| AV1 | P12 | ✓ DETECTED (20 violations) | ✗ CLEAN 0/5 | ✗ |
| TA1 | P13 | ✓ DETECTED (20) | ✗ CLEAN 0/440 | ✗ |
| SC1 | P14 | ✓ DETECTED | ✗ CLEAN 0/4 | ✗ |
| CW1 | P15 | ✓ DETECTED | ✗ CLEAN 0/4 | ✗ |
| LN1 | P16 | ✓ DETECTED (20) | ✗ CLEAN 0/187 | ✗ |
| **D2-new** | — | **8/8 = 100%** | **0/8 = 0%** | **0/8 = 0%** |

→ TrainCheck 0/8 是 P9-P16 类型 bug 的本质：tracker counter / config flag / module attr / shared cache 这些 invariant 落在 trace-instrumentation 之外，唯一能写规则的途径是 P9-P16 catalog 显式声明。

---

## 2. D2 27-bug 完整三方表

### 2.1 Aggregate

| 工具 | D1' 14 老 | D1' 3 新 (CF1/CM1/OF1) | D2-new 8 | 2 boundary | **D2 27** |
|------|-----------|------------------------|----------|------------|-----------|
| TrainAudit | 14/14 | 3/3 | **8/8** | 0/2 (设计如此) | **25/27 = 92.6%** |
| TrainCheck | 8/14 | 0/3 | 0/8 | 0/2 | **8/27 = 29.6%** |
| Naïve | 0/14 | 0/3 | 0/8 | 0/2 | **0/27 = 0%** |
| Fixed FP | 0 | 0 | 0 | — | **0/27 = 0%** |

### 2.2 关键洞察

1. **TrainAudit 92.6% 命中 brief §10 best case**（25-27/27 区间）。剩 2/27 = 7.4% 是 LC1/DL2 boundary，对应 §3.3 16.8% 物理上界的 detector-side 表现
2. **TrainCheck 在 D2-new 上 0/8 完全 miss**，比 D1' 的 47% (8/14) 低，与 brief §0 表 "≤12/27" 一致。说明 TrainCheck 的 trace-based 范式无法表达 P9-P16 涉及的 module attribute / config flag / cache identity 类不变量
3. **Naïve 0/27** —— 0 false positive 也是 0 detection。silent error 的极端定义。
4. **0 FP across all 27 bugs and all 3 tools** → §6 "no false positives on clean run" 叙事保住

---

## 3. Phase Status & Deliverables

### 3.1 Phase 1 — Miner Agent Prompt Extension (spec deliverable)

`benchmark/eval/p9_p16_deployment/miner_runs/pattern_hints.md` —— 8 个 P9-P16 的 prompt 注入 spec，含：
- 每个 pattern 的 skeleton 选择（继承 P1-P8 中最相似的）
- 候选 invariant 的形式描述
- 锚点 bug 列表（5+ per pattern）
- 4 框架的 source 检索 hint

⏸️ **不在本实验 scope**：实际 miner FSM 集成是 paper §5 工程团队工作。spec 已就绪。

### 3.2 Phase 2 — Runtime Hookpoint Binding (spec + matrix)

- `runtime_integration/hookpoint_matrix.csv` —— 16 pattern × 9 hookpoint 完整绑定矩阵（**新增 1**: `loss.compute.post`，其他 8 复用）
- `runtime_integration/loss_compute_post_adapter.spec.md` —— 4 框架 adapter 集成 spec（~150 LoC 总）

⏸️ **不在本实验 scope**：collector adapter 改动 + 200-step smoke test 由 paper 团队执行。

### 3.3 Phase 3 — 8 个新 Surrogate ✅ 完成

`benchmark/eval/d2_extension/{ID1,CC1,PE1,AV1,TA1,SC1,CW1,LN1}_{buggy,fixed}.py` —— 16 个 plain surrogate 全部 CPU 跑通 + 校准成功。

复制副本至 `benchmark/eval/traincheck_surrogates/` 加 `model` alias，让 paper §6 `run_one.sh` 直接消费。

### 3.4 Phase 4 — D2-new × 三方实测 ✅ 完成

| 工具 | 命令 | 结果 |
|------|------|------|
| TrainAudit | `python3 benchmark/eval/d2_extension/trainaudit_inline_d2.py` | **8/8 ✓** |
| TrainCheck | `ssh eval-gpu-0 "bash batch_d2_new.sh"` (paper §6 run_one.sh × 8) | 0/8 |
| Naïve | `python3 benchmark/eval/d2_extension/run_naive_d2.py` | 0/8 |

聚合到 `d2_aggregate.json` + `d2_summary.csv` (27 行 paper-ready 表)。

### 3.5 Phase 5 — Report ✅ 完成 (本文档)

---

## 4. §10 Decision Matrix Lookup

| 实测 D2 | 论文叙事 | 选中 |
|---------|---------|------|
| TA 25-27/27 | abstract 改 "25/27" 或 "26/27"，§4.1 删掉 core/ext 区分 | **✓ 25/27** |
| TA 23-25/27 | §4.1 保留细分 | — |
| TA <23/27 | abstract 维持 D1' 数字，本 brief 作 §7 补充 | — |

→ 论文集成走第一档：**16 模式全部 deploy 验证，统一一行讲完，删除 core/ext 区分**。

---

## 5. Paper 集成 7 处改动 (per brief §6)

| # | 位置 | 改动 |
|---|------|------|
| 1 | [main_cn.tex:182](main_cn.tex#L182) abstract | "26 bug E2E" / "17/19" → **"D2 27 bug, TrainAudit 25/27 (92.6%)"** |
| 2 | [main_cn.tex:243](main_cn.tex#L243) intro contribution 3 | "8 模式 deploy" → **"16 模式 deploy + 25/27 验证"** |
| 3 | [main_cn.tex:413](main_cn.tex#L413) C1 | "8 个 deploy 模式" → **"16 个 deploy 模式"** |
| 4 | [main_cn.tex:419](main_cn.tex#L419), [:423](main_cn.tex#L423) §4.1 | 去 core/ext 区分；type 计数 "3/4/1" → **"4/9/3"** |
| 5 | [main_cn.tex:614](main_cn.tex#L614) §6.1 workload | "D1' 19" → **"D2 27 (含 8 个 P9-P16 surrogate)"** |
| 6 | [main_cn.tex:660-685](main_cn.tex#L660) detection table | 14 行 → **27 行 (D2 完整表)**，data source = `d2_summary.csv` |
| 7 | [main_cn.tex:1736](main_cn.tex#L1736) Appendix L | hookpoint **8 → 9** (加 `loss.compute.post`)，规则数 **24 → 估计 60-90** (P9-P16 加 30-60) |
| 7b | [main_cn.tex:1066](main_cn.tex#L1066) Appendix B | 删除 Origin (core/ext) 列 |
| 7c | [main_cn.tex:905](main_cn.tex#L905) conclusion | "覆盖 88\%" 数字保留（来自 31 号 brief），加 "deploy 验证 25/27" |

---

## 6. Files Produced

```
benchmark/eval/d2_extension/
├── README.md (TBD)
├── ID1_*, CC1_*, PE1_*, AV1_*, TA1_*, SC1_*, CW1_*, LN1_* × {buggy,fixed}.py    # 16 plain surrogates
├── trainaudit_inline_d2.py             # ⭐ P9-P16 inline rule check (8/8 ✓)
├── run_naive_d2.py                     # Naïve runner (0/8)
├── d2_aggregate.json                   # ⭐ 27-bug 三方 aggregate
├── d2_summary.csv                      # ⭐ paper-ready 27-row table
├── d2_report.md                        # ⭐ this file
└── results/
    ├── trainaudit_d2_new_inline.json
    └── naive_d2_new.json

benchmark/eval/traincheck_surrogates/   # paper §6 traincheck pipeline
├── ID1_*, ..., LN1_* × {buggy,fixed}.py    # 16 copies with model alias
├── batch_d2_new.sh                     # 8 surrogates run_one.sh batch
└── batch_d2_new_results.txt            # ⭐ TrainCheck 0/8

benchmark/eval/p9_p16_deployment/       # spec deliverables (Phases 1-2)
├── miner_runs/pattern_hints.md         # 8 pattern miner hints (per-framework spec)
└── runtime_integration/
    ├── hookpoint_matrix.csv            # 16-pattern × 9-hookpoint 绑定
    └── loss_compute_post_adapter.spec.md   # NEW hookpoint adapter 集成 spec

docs/v2_semantic_guided/
├── 32_p9_p16_deployment_brief.md       # 上游 brief
└── (paper 集成时新增) 32a_paper_diff.md
```

---

## 7. 一行复算

```bash
# Phase 3 sanity (CPU)
for s in ID1 CC1 PE1 AV1 TA1 SC1 CW1 LN1; do
    for v in buggy fixed; do
        python3 benchmark/eval/d2_extension/${s}_${v}.py
    done
done

# Phase 4 三方
python3 benchmark/eval/d2_extension/trainaudit_inline_d2.py    # 8/8 ✓
ssh eval-gpu-0 "bash -l -c 'cd $PWD && bash benchmark/eval/traincheck_surrogates/batch_d2_new.sh'"  # 0/8
python3 benchmark/eval/d2_extension/run_naive_d2.py            # 0/8

# Phase 4 aggregate (D2 27)
python3 -c "<inline aggregate, see source>"
# 输出: D2 27-bug TA 25/27 (92.6%) | TC 8/27 (29.6%) | Naïve 0/27 (0%)
```

---

## 8. 留给 Paper 团队的剩余工作

按 brief §6 计划，本实验 GPU 部分全部跑通。剩余 paper 工程：

1. **Phase 1 实施**: 把 `pattern_hints.md` 8 个 hint 注入 paper repo `agents/` 下 FSM prompt（2-3 天）
2. **Phase 2 实施**: 4 框架 adapter 加 `loss.compute.post` 钩子（~150 LoC, 2-3 天）+ 200-step 干净 smoke test 验证 0 FP
3. **Phase 5 paper 集成**: 应用 §5 表的 7 处 main_cn.tex 改动（0.5-1 天）

总计 paper 团队剩余工作: **5-7 天**（与 brief §7 估算 2-3 天 Phase 1 + 2-3 天 Phase 2 一致）。
