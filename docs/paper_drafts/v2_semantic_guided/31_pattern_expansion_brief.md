# 31. Pattern Catalog 扩充实验 Brief（给实验 agent）

> 目标：扩充当前 8 pattern (P1-P8, 覆盖 48.7%) → 12-16 pattern，让模式目录覆盖与 §3.3 三层 invariant 框架的 83.2% 理论上界一致。
> 上游：[27_master_summary.md](27_master_summary.md)、[28_392_extension_brief.md](28_392_extension_brief.md)（E1 pattern coverage = 48.7%）、[29_v2_full_392_report.md](29_v2_full_392_report.md)（v2 staircase = 83.2%）
> 完成日期：待定。
> 论文同步集成由我做。

---

## 0. TL;DR

§3.3 三层框架在 392 池上理论覆盖 **83.2%**；当前 8 pattern 实际覆盖 **48.7%**——中间 **34.5%（135 bug）** 是"三层能表达但 8 pattern 没归纳"的扩充空间。本实验通过 grounded theory 二次迭代提炼 4-8 个新 pattern (P9-P16)，目标让 P1-PN 集合覆盖 ≥83%。

| 数字 | 当前 | 目标 |
|---|---|---|
| Pattern 数量 | 8 | 12-16 |
| 392 bug 覆盖 | 191/392 = 48.7% | ≥326/392 = 83% |
| Uncovered (8-pattern) | 201 (51.3%) | ↓ 至 66 (16.8% physically unobservable) |

最终交付：4-8 个新 pattern 的 4 元组 `<r, σ, m, ρ>` 定义 + per-pattern coverage + 更新版 Appendix C 表 + 论文 §4.1 数字更新依据。

---

## 1. 背景与决策上下文

### 1.1 当前数字状况

```
392 bug
├── 191 (48.7%)  ← §4.1: 当前 8 个 pattern (P1-P8) 直接命中  ← 现状
├── 135 (34.5%)  ← 三层框架可表达，但 8 pattern 未归纳         ← 本实验目标
└──  66 (16.8%)  ← runtime 物理不可观测（§3.3 边界）           ← 不做
        ──────
        326 (83.2%) ← §3.3: 三层 invariant 框架的理论覆盖上界
```

### 1.2 当前 8 pattern 已覆盖什么

| ID | Pattern Name | 392 hit |
|---|---|---|
| P1 | Dtype Preservation | 43 (11.0%) |
| P2 | Scaling Consistency | 32 (8.2%) |
| P3 | Cross-Rank Replication | 42 (10.7%) |
| P4 | Invocation Frequency | 19 (4.8%) |
| P5 | State Restoration | 26 (6.6%) |
| P6 | Structural Integrity | 22 (5.6%) |
| P7 | Residual Stream Integrity | 5 (1.3%) |
| P8 | Counter Consistency | 6 (1.5%) |

合计去重 191 (48.7%)，均覆盖框架级、cross-rank 级、状态恢复类 bug。

### 1.3 待覆盖的 135 bug 是什么类型

根据 28号 E1 报告 §1.2 与 v2 全集报告 §1.2，未被 8 pattern 覆盖的 135 bug 主要是 **source-only 算法 bug**：
- **init-distribution**（OLMo / OLMo-core 系密集）：参数初始化分布约束（fan-in/fan-out scaling 错、权重 init seed 错）
- **RoPE / position encoding**：positional embedding 在 packed mode、sliding window、长序列下的几何/边界违反
- **sliding-window / attention mask**：mask 边界、window size 与 batch 错配
- **config-param consistency**：配置参数（如 `num_layers`、`n_kv_heads`）与实际 module 状态不一致
- **algorithmic regression**：MoE topk 选择、SDP attention 实现切换、normalization 顺序等算法层 bug

这些 bug **不是** P1-P8 当前规则能直接 instantiate 的——它们的语义违反不在 cross-rank checksum / dtype preservation / invocation frequency 等已有规则的语言里。**但它们在三层框架内是可表达的**（v2 staircase 标注证明 minimum_sufficient_layer ≠ none）——只是缺规则。

---

## 2. 实验流程

### 2.1 Phase 1：提取 135 个 uncovered bug

```bash
python3 -c "
import json
from pathlib import Path
e1 = json.loads(Path('benchmark/eval/extension_v3/pattern_coverage_392.json').read_text())
manifest = json.loads(Path('benchmark/eval/manifest_v2.json').read_text())
m_idx = {b['bug_id']: b for b in manifest['bugs']}

# uncovered = patterns_hit empty
uncovered = []
for bug_id, e in e1['per_bug'].items():
    if not e.get('patterns_hit'):
        b = m_idx[bug_id]
        uncovered.append({
            'bug_id': bug_id,
            'framework': b['framework'],
            'category': b['category'],
            'root_cause': b.get('root_cause', '')[:300],
        })

# v2 hint: 这些 bug 在 v2 协议下被归到哪个 P
v2 = json.loads(Path('benchmark/eval/v2_full/annotations_392_v2.json').read_text())
v2_idx = {b['bug_id']: b for b in v2['per_bug']}
for u in uncovered:
    u['v2_hint'] = v2_idx.get(u['bug_id'], {}).get('pattern_id', 'none')

Path('benchmark/eval/pattern_expansion/uncovered_135.json').parent.mkdir(parents=True, exist_ok=True)
Path('benchmark/eval/pattern_expansion/uncovered_135.json').write_text(json.dumps(uncovered, indent=2))
print(f'TODO: {len(uncovered)} uncovered bugs')
"
```

预期输出：~135 bug，含 framework × category × root_cause + v2 协议下的 pattern_id 提示。

### 2.2 Phase 2：Inductive grounded theory 二次迭代

仿照 §3.1 13 类 taxonomy 的归纳流程，但这次是为 **pattern**（不是 category）做归纳：

1. **Open coding**（per-bug 5-10 标签）：
   - 对 135 bug 自由打"语义违反结构"标签（如 `init-fan-in-scaling`、`rope-pos-overflow`、`mask-window-mismatch`、`config-attr-stale`、`topk-degenerate` 等）
   - 不限 8 pattern 范围，自由命名
2. **Axial clustering**：
   - 按"规则形式 r 的语法相似性"聚合（不按 framework 或 category）
   - 例如：`init-fan-in` + `init-fan-out` + `weight-init-seed-consistency` → 都是"参数初始化的统计性质守护"，聚为一个候选 pattern
3. **Pattern 收敛**：
   - 每个候选 pattern 必须满足：(i) 至少 5 个 bug 命中（避免单点）；(ii) 能写成清晰的 `<r, σ, m, ρ>` 4 元组；(iii) 与 P1-P8 不重复
   - 三轮迭代收敛至 **4-8 个新 pattern**（P9-P16）

### 2.3 Phase 3：全集验证（Pattern catalog v2）

把 P1-PN（N=12 至 16）pattern 集喂给 LLM，在 392 全集上重跑（同 28 号 E1 strict 协议）：

- 输入：392 bug × P1-PN pattern 描述
- 输出：每 bug 的 `patterns_hit` 列表（P1..PN 的子集，可空）
- 目标：`any_pattern_hit / 392 ≥ 83%`，且 `(any_pattern_hit + unobservable) / 392 ≈ 100%`

**Sanity check**：原 8 pattern 的 hit count 应保持稳定（±3 bug/pattern），不应因 prompt 改动让 P1-P8 hit 数大幅波动。如果出现 ≥10% 漂移，说明 prompt 协议变了，要诊断。

---

## 3. 4-8 个新 Pattern 候选方向（hint，不限制）

实验 agent 自由 inductive 归纳，但作为 hint 我列出可能的方向（基于 135 bug 的初步分析）：

| Hint ID | 候选 Name | 候选 r (规则) | 覆盖大致估计 |
|---|---|---|---|
| **P9?** | Init Distribution Consistency | 参数初始化的统计矩（mean/var/fan-in scaling）应满足声明的分布约束 | ~25-35 bug |
| **P10?** | Position Encoding Integrity | positional embedding 在 packed/window/long-context 下应满足 boundary constraint | ~15-25 bug |
| **P11?** | Mask / Window Geometry | attention mask shape 与 window size、kv-cache 边界一致 | ~10-20 bug |
| **P12?** | Config-Param Consistency | 配置字段（如 `num_layers`、`vocab_size`）与对应 module 实例化状态一致 | ~15-25 bug |
| **P13?** | Algorithm Variant Selection | 多算法分支（如 SDP / xformers / vanilla attention）的运行时选择与 config 声明一致 | ~10-15 bug |
| **P14?** | Reduction Order / Numerical Stability | reduction 顺序（如 softmax over heads vs over experts）与数值稳定性约束 | ~10-15 bug |
| **P15?** | Optimizer State Schema | optimizer state 字典的 key 完整性 / 类型 schema 与 optimizer 声明一致 | ~5-10 bug |
| **P16?** | Routing / Dispatch Integrity | MoE/MoD 的 router decision 与下游 expert/dispatcher 状态一致 | ~5-10 bug |

→ Agent **不必**采纳全部 hint，但收敛后的 N 应在 4-8 之间。

---

## 4. 输出物

```
benchmark/eval/pattern_expansion/
├── uncovered_135.json                # Phase 1 输出
├── open_coding_labels.json           # Phase 2.1 自由标签
├── axial_clusters.json               # Phase 2.2 聚类
├── new_patterns/
│   ├── P9.yaml                       # 4 元组定义 + 描述 + 5 个示例 bug ID
│   ├── P10.yaml
│   └── ...
├── pattern_catalog_v2.json           # P1-PN 完整 catalog
├── pattern_coverage_v2_392.json      # Phase 3 全集 LLM 标注
├── pattern_coverage_v2_summary.csv   # paper-ready 表
└── pattern_expansion_report.md       # ⭐ 实验报告
```

每个新 pattern 的 yaml 格式（仿照已有 P1-P8）：

```yaml
id: P9
name: Init Distribution Consistency
type: C   # A=trace SQL / B=runtime hook / C=static assert
rule_r: |
  for each Parameter p with declared init_dist (e.g., normal(0, std=0.02)):
    p.data.std() ∈ [declared_std × 0.5, declared_std × 1.5]
    p.data.mean() ∈ [-declared_std × 0.1, declared_std × 0.1]
scope_sigma: |
  p.requires_grad == True
  p.declared_init_dist != None
mode_m: C
precond_rho: |
  step == 0 (post-init, pre-train)
  framework_init_done == True
example_bugs:
  - O-NEW-12   # OLMo SwiGLU init wrong fan
  - OC-NEW-7   # olmo-core MoE expert init scaling
  - ...        # 至少 5 个示例
covers_count_estimate: 28
related_existing_pattern: P1 (Dtype Preservation, 但 P1 只看 dtype 不看分布)
```

---

## 5. 论文集成（实验完成后我做）

预计要改：

1. **§4.1 [main_cn.tex:423](../../main_cn.tex#L423)**：8 pattern → N pattern (12-16)，覆盖率 48.7% → 实测百分比（目标 ≥83%）
2. **Appendix C 8 模式表 [main_cn.tex:1059-1075](../../main_cn.tex#L1059)**：表格扩展为 N 行，"8" → "N"
3. **abstract / intro / conclusion**：如果数字闭环到 §3.3 的 83.2%，可保留模糊措辞 "covers majority of surveyed bugs"，或显式写 "covers 83% of surveyed bugs"
4. **possibly figure**：如果 pattern 数量大幅增加，可能需要重画 pattern 分组图（按 Type A/B/C 或按语义维度分组）

---

## 6. 时间预算

| Phase | 工作 | 工作量 |
|---|---|---|
| Phase 1 | 提取 135 uncovered bug + v2 hint 加注 | 0.2 天 |
| Phase 2.1 | Open coding（人工 + LLM 辅助打标签） | 1-2 天 |
| Phase 2.2 | Axial clustering + pattern 收敛 | 1 天 |
| Phase 2.3 | 4-8 个新 pattern 4 元组定义 | 1 天 |
| Phase 3 | LLM 全集 rerun + sanity check | 0.5 天 |
| 报告 | 实验报告 + 论文集成所需材料 | 0.5 天 |
| **合计** | | **4-5 天** |

Token 预算：~10-15M（392 × 1 prompt = 392 calls，主要 cost 在 Phase 3 的 catalog v2 全集 rerun），成本 \$20-40。

---

## 7. 不要做的事

- ❌ **不要重新评估 P1-P8** — 保持现状（48.7% 基线已固定）
- ❌ **不要扩展到 16.8% unobservable bug** — 物理边界，不是 pattern 缺口（§3.3 已闭环）
- ❌ **不要追求 100% 覆盖** — 16.8% gap 必须保留
- ❌ **不要新增超过 8 个新 pattern** — 总 catalog ≤16，超过 reader 会迷失
- ❌ **不要让单个新 pattern 命中 < 5 个 bug** — 太碎，不构成有效模式
- ❌ **不要改 manifest_v2.json**
- ❌ **不要把 v2 协议直接当 ground truth** — v2 是 liberal 协议，agreement 与 28号 E1 仅 19.6%；用作 hint 而非 truth
- ❌ **不要省略 4 元组的 precond_rho** — 缺 precond 是 §3.3 论证三层联合必要的核心

---

## 8. 失败处理

| 实测覆盖 | 应对 |
|---|---|
| ≥83% | ✓ 与 §3.3 闭环，论文直接说覆盖 83% |
| 70-83% | 报告差距，论文 §4.1 写实测数字，承认"部分 source-only bug 难以归纳为通用 pattern" |
| <70% | grounded theory 失败，可能需要重审：要么 §3.3 的 83.2% 是 over-estimated（v2 标注协议偏 liberal）；要么 source-only bug 本质上不能用三层 invariant 表达。诚实报告，论文需重写 §4.1 |

如果 N 收敛到 >12（即新增 ≥5 个 pattern 但仍未达 83%），暂停实验，让我重新评估 §3.3 与 §4.1 的对齐策略。

---

## 9. 一行复算

```bash
# Phase 1: 提取 uncovered
python3 benchmark/eval/pattern_expansion/extract_uncovered.py

# Phase 2: open coding + clustering（subagent 多轮）
# (subagent prompt 在本 brief §2.2)

# Phase 3: catalog v2 全集 rerun
python3 benchmark/eval/pattern_expansion/run_catalog_v2.py

# Aggregate
python3 benchmark/eval/pattern_expansion/aggregate.py
```

预期最终产出：`pattern_coverage_v2_summary.csv` 含 N 行（每行一个 P1..PN 的 hit count + %），最后一行 `any_pattern` 累积覆盖目标 ≥83%。

---

## 10. 决策矩阵（实验完成后我看哪一档）

| 新 pattern 数 N | 总覆盖 | 论文集成 |
|---|---|---|
| N=4 (12 total) | ≥83% | §4.1 改为 "12 pattern 覆盖 ≥83%"；Appendix C 加 4 行 |
| N=5-6 (13-14 total) | ≥83% | §4.1 改为 "13-14 pattern 覆盖 ≥83%"；Appendix C 重写 |
| N=7-8 (15-16 total) | ≥83% | §4.1 段重组（按 Type A/B/C 或语义维度分组）；Appendix C 重新分节 |
| N≥4 但覆盖 <83% | 任意 | §4.1 写实测数字 + 承认差距，与 §3.3 16.8% boundary 仍闭环 |
