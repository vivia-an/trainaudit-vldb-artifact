---
title: 任务 — Three-way Baseline 对比 (TrainAudit / TrainCheck / Naïve)
created: 2026-05-07
status: D1 DONE / real-23 BLOCKED
owner: claude (D1 portion 2026-05-07)
estimated_effort: 2–3 工作日（实际 D1 portion ~5h，real-23 仍阻塞）
output_destination: docs/v2_semantic_guided/22_paper_evidence_index.md §2.6 (已加)
---

## 完成状态（2026-05-07）

**D1 同集合定量对比已完成（17-bug 全集合）**：见 `paper_table_baseline_3way.md` + doc 22 §2.6。

| 方法 | D1 buggy 检出 | D1 fixed FP |
|------|---------------|-------------|
| TrainAudit | **17/17 = 100.0%** | 1/17 = 5.9% (OC-NEW-3 已知 surrogate FP) |
| TrainCheck | **10/17 = 58.8%** | N/A（旧记录 fixed 作 reference，不是独立 FP） |
| Naïve | **0/17 = 0.0%** | 0/17 |

Subset：`benchmark/eval/synthetic_17.json` (B1/B2/B3/B8/B11/B12/B13/M-012/M-014/M-020/M-024/M-NEW-5/O-005/O-NEW-1/O-NEW-9/OC-NEW-2/OC-NEW-3)。注意之前一版只跑了 14 bugs（synthetic_14），漏了 DeepSpeed B2/B3/B8 三个 T1 tier surrogate；本次补齐到 17。

**部分完成（real-23 Naïve）**：

`bash benchmark/eval/sweep_naive_real.sh` on eval-gpu-0 跑了完整的 23 bugs。结果：

| 子集 | 数量 | Naïve verdict |
|------|------|---------------|
| Phase 1-2 verified | 13 | 13/13 FAIL（driver-pre-training crash 或 driver-runs-no-metric） |
| Hunt real-run | 6 | 6/6 FAIL（import 错误 / 纯 structural script） |
| Hunt structural | 4 | 4/4 N/A by design（AST emulation，无 metric stream） |

**仍阻塞（doc 26 §3.2 / D7 未完）**：23 real bugs 的 TrainAudit + TrainCheck subprocess。每 bug 需要：
- 包装 framework 训练为 traincheck-collect 可识别的 pyscript（torchrun + multi-process 接口需重设计）
- 解决 commit-vs-API 兼容矩阵（M-* 系列 buggy commit 与 venv-cu126 torch 2.7.1 不兼容）
- SSH 到 eval-gpu-0 / olmo-gpu-0 跑（每个 bug 数分钟）

预估 0.5-1 天/bug × 23 bugs ≈ 1.5-2 周。需另开 session 推进，且大概率拆分成 13 verified + 10 hunt 两批。

# 任务: Paper §4.1 三方 Baseline 对比

## 1. 背景

Paper `main_cn.tex` §4.1 (Detection Efficacy) 当前的 baseline 数字（TrainAudit 30/33 (90.9%) / TrainCheck 2/33 (6.1%) / Naïve 6/33 (18.2%)）来自**已废弃的 33-fault 注入集**，无 doc 22 锚点（CLAUDE.md 已标"旧静态"）。

新评估集合已就绪：
- **D1 synthetic harness**：15 buggy + 13 fixed-commit pairs（CPU-friendly，已有 surrogate runners）
- **23 real bugs**：13 Phase 1–2 verified（doc 22 §2.1）+ 10 Hunt phase E2E confirmed（doc 22 §2.5 + doc 25）

TrainAudit 自身在两个集合上的检测数据都已就绪（doc 22 §2.0、§2.1、§2.5）。本任务**只补 TrainCheck 和 Naïve 两条 baseline 在同一集合上的数据**，让 paper §4.1 三方对比表用同集合可比数字。

## 2. 交付物 (Definition of Done)

完成本任务即产出：

| # | 产物 | 路径 | 用途 |
|---|------|------|------|
| D1 | Naïve baseline 实现 | `benchmark/eval/baseline_naive.py`（新建） | 跑 D1 + 23 real，输出 CSV |
| D2 | TrainCheck baseline 完整集成 | `benchmark/eval/baseline_traincheck.py`（修改 `--mode synthetic` 实现） | 跑 D1 + 23 real，输出 CSV |
| D3 | Naïve CSV | `benchmark/eval/baseline_naive_results.csv` | 每个 bug 一行 |
| D4 | TrainCheck CSV | `benchmark/eval/baseline_traincheck_results.csv` | 每个 bug 一行 |
| D5 | 三方对比 markdown 表 | `benchmark/eval/paper_table_baseline_3way.md` | 直接进 paper §4.1 |
| D6 | doc 22 新增 §2.6 row | `docs/v2_semantic_guided/22_paper_evidence_index.md` | 真值索引同步 |
| D7 | 复现命令记录 | `docs/v2_semantic_guided/22_paper_evidence_index.md §2.6` 内 | 命令、commit hash、seed |

## 3. 评估集合定义

### 3.1 D1 synthetic harness — 28 instance

- **Manifest**：`benchmark/eval/synthetic_14.json`（当前 14 bugs，需要扩到 15）
- **Bugs（14 已定义 + 待补 1）**：
  ```
  B1, B11, B12, B13, M-012, M-014, M-020, M-024, M-NEW-5,
  O-005, O-NEW-1, O-NEW-9, OC-NEW-2, OC-NEW-3
  ```
  → 待与 doc 22 §2.0 对账，补足到 15 buggy。
- **每个 bug 跑 2 次**：buggy 版本（应检出）+ fixed 版本（应不检出 → FP test）
  - 13 个 bug 有 fixed pair（28 instances 中的 13 fixed）
  - 2 个 bug 只有 buggy（无 fixed pair；论文里只算 detection 不算 FP）
- **Runner 入口**：`benchmark/eval/synthetic_runners.py::run_synthetic_bug(bug_id)`
- **TrainAudit 对照数据**：15/15 (100%) buggy 检出，0/13 (0%) fixed FP（doc 22 §2.0）

### 3.2 23 real bugs — 23 instance

#### Phase 1–2 verified (13)，来源 doc 22 §2.1

| Bug ID | Framework | Commit | Tier | Rule |
|--------|-----------|--------|------|------|
| B11 | DeepSpeed | `005afe12` | T0 | clip-grad-bounded |
| B12 | OLMo-core | `6e330ba2` | T0 | initial-lr-present |
| O-NEW-1 | OLMo | `67c9e315` | T0 | norm-output-unit-rms |
| M-014 | Megatron | `5153efea0` | T0 | softmax-degenerate |
| O-005 | OLMo | `204ad53c` | T0 | checkpoint-preserve-rng |
| OC-NEW-2 | OLMo-core | `2b6cf996` | T0 | optim-step-counter-monotonic |
| B13 | OLMo | `562c0fe0` | T1 | residual-stream-preserved |
| O-002 | OLMo | `c482df74` | T1 | residual-stream-preserved |
| M-NEW-5 | Megatron | `87d9d2506` | T1 | router-has-calculate-per-token-loss |
| M-012 | Megatron | `a58768725f` | T1 | expert-bias-fp32 |
| M-020 | Megatron | `99f999a466` | T1 | layer-count-strict |
| M-024 | Megatron | `20b395424d` | T1 | jitter-preserves-dtype |
| OC-NEW-3 | OLMo-core | `f34e7ddc` | T1 | sqrt-decay-front-loaded |

#### Hunt phase E2E confirmed (10)，来源 doc 22 §2.5 + doc 25

| Cand ID | Framework | 评价方式 |
|---------|-----------|----------|
| CAND_OLMOCORE_RNGCKPT | OLMo-core | E2E real |
| CAND_OLMOCORE_EVAL_NOZEROGRAD | OLMo-core | E2E real (hunt iter 2) |
| CAND_OLMOCORE_FSDP_EXPERTS | OLMo-core | structural AST + source |
| CAND_DEEPSPEED_WARMUPCOSINE_MULTIGROUP | DeepSpeed | E2E real |
| CAND_DEEPSPEED_BF16_ZERO0_DUAL_BUG | DeepSpeed | E2E real @ H200 |
| CAND_DEEPSPEED_ZERO_OFFLOAD_MULTI_BACKWARD | DeepSpeed | E2E real (v0.18.7, hunt iter 10) |
| CAND_DEEPSPEED_BF16_BOUNDARY_GRAD_LEAK | DeepSpeed | E2E real (v0.18.7 + 2 H200) |
| CAND_MEGATRON_CUDAGRAPH_BUFFER_CORRUPTION | Megatron | structural on H200 (hunt iter 4) |
| CAND_OLMO_CKPT_SAVE_OVERWRITE_DROP | OLMo | structural AST + monkey-patch |
| CAND_OLMO_ADAPTIVE_CLIP_EMA_RESET | OLMo | structural AST + loop replay |

- **Runner 入口**：`benchmark/eval/run_all.py --mode subprocess`，对应 `benchmark/bugs/<id>/trainaudit_run.sh`
- 每个 bug 用其 `trainaudit_run.sh` 接口，需要适配两条 baseline 接同一 stdin 接口或写 sibling script

## 4. 任务一: Naïve baseline 实现

### 4.1 定义

Naïve metric monitoring 检测器，做最朴素的标量监控：

| 信号 | 阈值规则 | 检出条件 |
|------|----------|----------|
| Loss spike | `loss[t] > k1 * median(loss[t-W:t])` | k1=10, W=20 步 |
| Loss NaN/Inf | `not isfinite(loss[t])` | 任意一步 |
| Grad NaN/Inf | `any(not isfinite(g))` for g in grads | 任意一步 |
| Grad-norm spike | `grad_norm[t] > k2 * max(grad_norm[t-W:t])` | k2=10, W=20 步 |

任何一条触发即判 `DETECTED`；否则跑完 N 步后判 `CLEAN`。

### 4.2 实现位置

新建 `benchmark/eval/baseline_naive.py`，结构对齐 `baseline_traincheck.py`：

```python
def run_naive_on_bug(bug_id: str, mode: str) -> Dict[str, Any]:
    """
    mode='synthetic': 复用 synthetic_runners 的 run_synthetic_bug，
                      但替换 detector 为 naive_detector
    mode='subprocess': 跑 benchmark/bugs/<id>/trainaudit_run.sh，
                       但传 env DETECTOR=naive，让 driver 跳过 TrainAudit 规则
    """
```

### 4.3 步骤

1. 抽出 `synthetic_runners.py` 训练循环里的 `(loss, grad_norm)` 时序（或新建一个 hook）
2. 实现 `naive_detector(loss_seq, gradnorm_seq) -> verdict`
3. 在 D1 14 bugs（每个 buggy + fixed）上跑 → 28 instances
4. 在 23 real bugs 上跑（subprocess mode）：
   - 改 `benchmark/bugs/<id>/trainaudit_run.sh` 加 `--detector=naive` 开关，或写并列脚本 `naive_run.sh`
   - 每个 driver 的 dump 文件已经包含 loss / grad_norm 字段，直接读
5. 输出 CSV (schema 见 §4.4)

### 4.4 输出 CSV schema

`benchmark/eval/baseline_naive_results.csv`，每行一 instance：

```
bug_id, framework, category, phase, set,
verdict,                # DETECTED | CLEAN | FAIL
trigger_signal,         # loss_spike | loss_nan | grad_nan | gradnorm_spike | (空)
trigger_step,           # 触发步数（DETECTED 时）
duration_s,
note
```

`set ∈ {D1, real}`，`phase ∈ {buggy, fixed}`（real bugs 全是 buggy）。

### 4.5 期望结果（论文里要看的数）

- **D1 buggy 检出率**：M_d1 / 15
- **D1 fixed FP 率**：F_d1 / 13
- **23 real 检出率**：M_real / 23

预期 Naïve 在 D1 上检出率会显著低于 TrainAudit（因为很多 semantic faults 不会立即触发 metric 异常）。具体数字未知，跑出来就是数字。

## 5. 任务二: TrainCheck baseline 完整集成

### 5.1 现状

`benchmark/eval/baseline_traincheck.py` 已有：
- ✅ `_import_traincheck()` 验证 import + Instrumentor 可用
- ✅ `harness-check` mode 输出 import 状态
- ⏳ `--mode synthetic` **是 stub**：返回 `traincheck_verdict = "PENDING_INTEGRATION"`，仅记录 TrainAudit verdict 作占位
- ✅ CSV / Markdown 输出 schema 已就绪

TrainCheck 安装在 `exp/traincheck/TrainCheck/`，已 pip-importable，submodule（instrumentor / checker / infer_engine / collect_trace / invariant）全部 OK。

### 5.2 待完成的集成工作

需要完成 `run_synthetic_with_traincheck()` 的真实实现，目前 stub 在 `baseline_traincheck.py:75-109`。

TrainCheck API 期望：
1. **Instrumentor.start(model)** — 在训练循环开始前调用，包装 model + optim + autograd 钩子
2. **训练循环** — TrainCheck 自己往 `collect_trace` 写文件 trace
3. **Checker pipeline** — `traincheck.checker.run(trace_path)` 输出 verdict（具体 API 需要看 TrainCheck 源码或文档）

### 5.3 步骤

**5.3.1 D1 synthetic 模式**：

1. 阅读 `exp/traincheck/TrainCheck/README.md` + `traincheck/checker/`，理解：
   - Instrumentor.start() 签名 + lifecycle（哪些 hook，写到哪）
   - Checker 期望的 trace 格式（文件路径、json schema）
   - Verdict 的对接接口（返回 list of violations？exit code？stdout 解析？）
2. 修改 `synthetic_runners.py::run_synthetic_bug(bug_id)`：
   - 增加可选参数 `detector='trainaudit'|'traincheck'|'naive'`
   - `traincheck` 分支：包装训练循环为 `with Instrumentor.start(...)` 上下文
   - 训练完成后调 `traincheck.checker.run(trace_path)` 解析 verdict
3. 替换 `baseline_traincheck.py:run_synthetic_with_traincheck()` 中的 `traincheck_verdict = "PENDING_INTEGRATION"` 为真实 verdict
4. 跑全集：`python benchmark/eval/baseline_traincheck.py --mode synthetic --subset benchmark/eval/synthetic_15.json`

**5.3.2 23 real bugs subprocess 模式**：

1. 在每个 `benchmark/bugs/<id>/trainaudit_run.sh` 同目录加 `traincheck_run.sh`：
   - 启动框架训练 + Instrumentor.start
   - 输出统一契约行：`[<bug_id>] BUG DETECTED|CLEAN|FAIL: <details>`
2. 实现 `baseline_traincheck.py --mode subprocess`（仿照 `run_all.py` 的 subprocess 逻辑）
3. 跑：`python benchmark/eval/baseline_traincheck.py --mode subprocess --subset <23-real.json>`

### 5.4 输出 CSV schema

`benchmark/eval/baseline_traincheck_results.csv`，每行一 instance：

```
bug_id, framework, category, phase, set,
trainaudit_verdict,     # 对照（DETECTED|CLEAN|FAIL）
traincheck_verdict,     # 本次（DETECTED|CLEAN|FAIL）
traincheck_violations,  # 触发的 invariant id 列表（DETECTED 时）
duration_s,
note
```

### 5.5 Fallback：若 5.3.1 无法完成

如果 TrainCheck 的 trace/checker 接口集成时间超过 1.5 天还没跑通，**降级到 related-work 定性对比**：
- 不在 paper 同集合表里塞 TrainCheck 数字
- 改在 §5 Related Work 段落 cite TrainCheck 论文已发表的 detection profile（"shape/dtype mismatch focus, no semantic constraint"）
- `baseline_traincheck.py --mode related-work` 已有占位输出
- doc 22 §2.6 row 标 `traincheck = qualitative-only`

**这条 fallback 必须先和论文负责人确认再启用**。优先尝试同集合定量对比。

## 6. 任务三: 三方对比表 + 写回 doc 22

### 6.1 paper_table_baseline_3way.md (D5)

输出位置：`benchmark/eval/paper_table_baseline_3way.md`

格式（直接可贴进 paper §4.1）：

```markdown
| 方法 | D1 buggy 检出 | D1 fixed FP | 23 real 检出 |
|------|---------------|-------------|----------------|
| TrainAudit (本工作) | 15/15 (100%) | 0/13 (0%) | 23/23 (100%) |
| TrainCheck         | M_tc/15      | F_tc/13     | M_tc_real/23  |
| Naïve monitoring   | M_n/15       | F_n/13      | M_n_real/23   |

(具体每行 verdict 见 baseline_*_results.csv per-bug)
```

### 6.2 写回 doc 22 §2.6（D6 + D7）

在 `docs/v2_semantic_guided/22_paper_evidence_index.md` 新增 §2.6 行，包含：

```
| 锚点 | row 描述 | value | last_updated | 复现命令 |
|------|----------|-------|--------------|----------|
| §2.6 baseline-d1-trainaudit | TrainAudit on D1 | 15/15 + 0/13 | 2026-05-XX | python benchmark/eval/run_all.py --mode synthetic --subset benchmark/eval/synthetic_15.json |
| §2.6 baseline-d1-traincheck | TrainCheck on D1 | M/15 + F/13 | 2026-05-XX | python benchmark/eval/baseline_traincheck.py --mode synthetic --subset benchmark/eval/synthetic_15.json |
| §2.6 baseline-d1-naive | Naïve on D1 | M/15 + F/13 | 2026-05-XX | python benchmark/eval/baseline_naive.py --mode synthetic --subset benchmark/eval/synthetic_15.json |
| §2.6 baseline-real-trainaudit | TrainAudit on 23 real | 23/23 | 2026-05-XX | python benchmark/eval/run_all.py --mode subprocess --subset benchmark/eval/real_23.json |
| §2.6 baseline-real-traincheck | TrainCheck on 23 real | M/23 | 2026-05-XX | python benchmark/eval/baseline_traincheck.py --mode subprocess --subset benchmark/eval/real_23.json |
| §2.6 baseline-real-naive | Naïve on 23 real | M/23 | 2026-05-XX | python benchmark/eval/baseline_naive.py --mode subprocess --subset benchmark/eval/real_23.json |
```

每行带：seed、PyTorch 版本、CUDA 版本、commit hash 列表（确保可复现）。

## 7. 验收标准

- [ ] D3 + D4 两个 CSV 完整，每个 bug 一行（D1 28 行 + real 23 行 = 51 行/baseline）
- [ ] 没有 `verdict=FAIL` 行 > 5%（否则需要排查环境或 driver 问题）
- [ ] D5 表格数字与 D3/D4 CSV 一致
- [ ] D6 doc 22 §2.6 已 commit，所有命令可被复现
- [ ] `python benchmark/eval/baseline_traincheck.py --mode harness-check` 不再返回 `PENDING_INTEGRATION`，能输出真实 verdict
- [ ] D5 的数字差异可解释：TrainAudit ≥ TrainCheck ≥ Naïve（合常理）；如果 Naïve 检出 > TrainCheck，需要在 note 里说明

## 8. 已知风险与 fallback

| 风险 | 概率 | Mitigation |
|------|------|-----------|
| TrainCheck 的 Instrumentor lifecycle 与 synthetic_runners 不兼容 | 中 | 跑 fallback §5.5（related-work 定性） |
| 23 real bugs 中部分需要 GPU/特定 torch 版本，跑 subprocess mode 失败 | 高（doc 22 §5 已记） | 这部分 row 标 `BLOCKED:env`，paper 写"subprocess subset"附说明 |
| Naïve baseline 在 synthetic surrogate 上完全检不出（→ 0%） | 中 | 这就是想要的论证（"Naïve 漏 80%+"），不是问题 |
| D1 synthetic 17 vs 14 bug 数量对账不一致 | 低 | 先和 doc 22 §2.0 对账确认 D1 完整定义 |
| `synthetic_runners.py` 没暴露 loss/gradnorm 时序给 naive 用 | 中 | 加一个 `metrics_callback` 参数或用 detector 接口注入 |

## 9. 工作量预估

| 子任务 | 预估 |
|--------|------|
| Naïve baseline 实现 + D1 synthetic | 0.5 天 |
| Naïve baseline 23 real subprocess | 0.5 天 |
| TrainCheck integration（学 API + 改 synthetic_runners） | 1 天 |
| TrainCheck 23 real subprocess | 0.5 天 |
| 汇总表 + doc 22 写回 + 验收 | 0.5 天 |
| **合计** | **3 天** |

如果 TrainCheck integration 卡 1.5 天没跑通，启用 §5.5 fallback，整体压回 2 天。

## 10. 严禁事项

- ❌ **不要把已发表的 TrainCheck paper 数字（2/33 6.1%）拼接到本任务的同集合表里** —— 会被 reviewer 抓"vibe baseline"
- ❌ **不要伪造或估算 verdict** —— 跑不出来就标 `BLOCKED:env`，禁止写"likely DETECTED"之类
- ❌ **不要跳过 doc 22 写回** —— paper 任何数字必须有 doc 22 锚点
- ❌ **不要碰 paper main_cn.tex / main.tex** —— 本任务只产数据 + 表，论文同步是后续人工触发的工作流

## 11. 完成后通知

任务完成后，更新本 doc 顶部 frontmatter `status: DONE`，并在 doc 22 §2.6 row 的 `last_updated` 写当天日期。然后通知论文负责人启动 §4.1 重写流程。

---

## 附录 A: 已存在的资产清单（受任务复用）

| 文件 | 用途 |
|------|------|
| `benchmark/eval/run_all.py` | TrainAudit 跑 D1 / real 的入口（subprocess + synthetic） |
| `benchmark/eval/baseline_traincheck.py` | TrainCheck baseline skeleton（待完成 synthetic mode） |
| `benchmark/eval/synthetic_runners.py` | D1 synthetic surrogate runner（in-process） |
| `benchmark/eval/synthetic_14.json` | D1 subset manifest（需扩到 15） |
| `benchmark/eval/manifest.json` | 295 bug 总池（含 framework / category / commit） |
| `benchmark/eval/fault_injection.py` | 老的 33-fault 注入逻辑（含 loss/grad 监测代码可借鉴给 naive） |
| `benchmark/bugs/<id>/trainaudit_run.sh` | 每个 real bug 的 driver |
| `exp/traincheck/TrainCheck/` | TrainCheck 完整 fork（已 import OK） |
| `docs/v2_semantic_guided/22_paper_evidence_index.md` | 真值索引，§2.6 待新增 |
| `docs/v2_semantic_guided/25_hunt_silent_errors_writeup.md` | 10 个 hunt E2E confirmed 的详情 |

## 附录 B: 联系点

如果任务执行中遇到设计层面歧义（例如 TrainCheck 的 verdict 解析没标准答案），把问题列在本 doc §12 末尾（新增），不要自作主张。
