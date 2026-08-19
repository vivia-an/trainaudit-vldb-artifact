# TrainAudit P0/P1 实验补做 Runbook

目的：照着这份文档把 §6 缺的实验全部跑出来。每节包含**问题、判定标准、复用什么、要新写什么、产出物、人工核对项**五块，跑完直接落到 §6 表格里。

---

## 0. 公共前置

### 0.1 物理环境
- 节点：`eval-gpu-0`（8×H200，CUDA 12.6 venv `/volume/qscai/cqs/temp/venv-cu126/`）
- 论文 repo：`/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025/`
- 依赖框架仓库（实测路径以 `gen_driver.py` 里的为准）：
  - Megatron-LM：`exp/Megatron-LM/`
  - DeepSpeed/OLMo/OLMo-core：见 `benchmark/eval/gen_driver.py` 的 env var 注释

### 0.2 复用现有数据 & 结果（不要重跑）
| 已有产物 | 路径 | 给哪节用 |
|---|---|---|
| 27-bug 主表数据 | `benchmark/eval/d2_extension/d2_aggregate.json` | §6.2 已落表，不动 |
| Pattern → hookpoint 映射 | `benchmark/eval/p9_p16_deployment/runtime_integration/hookpoint_matrix.csv` | §6.4 funnel 引用 |
| Miner prompt 详细规格 | `benchmark/eval/p9_p16_deployment/miner_runs/pattern_hints.md` | §6.4 A1 |
| Clean run sweep | `benchmark/sweep/aggregate.json` | §6.2 E（FP audit） |
| Driver auto-gen | `benchmark/eval/gen_driver.py`、`benchmark/eval/failures.csv` | §6.6 D |

### 0.3 输出根目录约定
所有新结果落到 `benchmark/eval/rebuttal_v1/`，子目录按本文档章节命名：

```
benchmark/eval/rebuttal_v1/
├── A1_mining_funnel/
├── A2_ablation/
├── B1_diagnosis_accuracy/
├── C1_overhead_gpu/
├── D_portability_tests/
└── E_clean_run_fp_audit/
```

---

## A. 6.4 Rule Mining Funnel + Ablation（P0，**必做**）

### A1. Mining Funnel 表

#### 问题
"24 条 active rule 是怎么从 LLM 候选漏斗出来的？"——证明 adversarial verification 不是空喊。

#### 判定标准
| Stage | 计入条件 | 拒绝原因示例 |
|---|---|---|
| L1 Hypothesis (raw) | `propose_hypotheses()` 返回的 Hypothesis 总条数 | — |
| L2 Enumerated predicate | `enumerate_predicates()` 落地的 Predicate 总条数 | schema 缺字段 → 0 候选 |
| L3 Healthy-validated | 在 ≥1 个 fixed reference trace 上 100% hold | 在 fixed run 上偶尔违反 → 阈值不稳 |
| L4 Adversarial-pass | Layer 4 LLM filter `keep=True` | 「step 0 lr=1e-4 这种是 config 特有的」类型被 drop |
| Deployed | 真正写进 `trainaudit/trainaudit/rules/T*_*.py` 或 `dsl/registry/` 的规则 | 与已部署规则重复 → drop |

#### 复用什么
- `trainaudit/trainaudit/mining/layer{1,2,3,4}*.py` 这套 4-layer 流水线已经有 LLM client interface。Layer 1 / Layer 4 默认用 `StubLLMClient`（CI 用），跑数据时**必须换成真 LLM**。
- 真 LLM 接入参考：`claude-proxy-v3.py` 或 `claude-proxy-v2.py`（仓库根目录已有），实现 `LLMClient.__call__(system_prompt, user_prompt) -> str`。
- prompt 规范：`benchmark/eval/p9_p16_deployment/miner_runs/pattern_hints.md`（P9–P16 的 Training Expert hint 已写齐）。
- 4 个框架的源码扫描入口：`pattern_hints.md` 每条 pattern 下面的"Source-code 检索 hint"。

#### 新写什么
新建 `benchmark/eval/rebuttal_v1/A1_mining_funnel/run_funnel.py`，对 4 个框架 × 16 个 pattern 跑一次完整 4-layer pipeline，每个 stage 的 candidate 列表存 JSON。

最小骨架（伪代码）：

```python
# benchmark/eval/rebuttal_v1/A1_mining_funnel/run_funnel.py
import json
from pathlib import Path
from trainaudit.mining import (
    propose_hypotheses, enumerate_predicates, schema_introspect,
    validate_against_healthy, filter_predicates,
)
from trainaudit.store import TraceStore
from claude_proxy_v3 import ClaudeLLMClient   # 你已有

OUT = Path(__file__).parent

FRAMEWORKS = {
    "megatron":  "/volume/qscai/.../Megatron-LM/megatron/core",
    "deepspeed": "/volume/qscai/.../DeepSpeed/deepspeed",
    "olmo":      "/volume/qscai/.../OLMo/olmo",
    "olmo_core": "/volume/qscai/.../olmo-core/src/olmo_core",
}

PATTERN_SCAN_HINTS = json.loads(
    (Path("benchmark/eval/p9_p16_deployment/miner_runs/pattern_hints.md")
     .read_text()))   # 把每个 pattern 的 search hints 抽成 dict 见下

# 每个 framework 选 N 个种子文件（来自 pattern_hints.md 中的 source 检索 hint），
# 总数控制在 30–60 文件，覆盖 16 个 pattern。
for fw, root in FRAMEWORKS.items():
    rec = {"framework": fw, "stages": {}}
    seed_files = collect_seed_files(root, PATTERN_SCAN_HINTS)  # 自己实现

    llm = ClaudeLLMClient(model="claude-opus-4-7")

    # L1 hypothesis
    hyps = []
    for f in seed_files:
        hyps.extend(propose_hypotheses(f.read_text(), framework=fw, llm_client=llm))
    rec["stages"]["L1_hypothesis"] = {"count": len(hyps),
                                       "items": [h.__dict__ for h in hyps]}

    # L2 enumerate
    schema = schema_introspect(reference_store_for(fw))
    preds = []
    for h in hyps:
        preds.extend(enumerate_predicates(h, schema))
    rec["stages"]["L2_enumerate"] = {"count": len(preds), "ids": [p.id for p in preds]}

    # L3 healthy validate
    healthy_store = TraceStore.open(f"benchmark/eval/test_dbs/{fw}_clean.duckdb")
    passed = []
    rejected_L3 = []
    for p in preds:
        res = validate_against_healthy(p, healthy_store)
        (passed if res.holds else rejected_L3).append((p, res))
    rec["stages"]["L3_healthy_validated"] = {
        "count": len(passed),
        "rejected": [{"id": p.id, "reason": r.reason} for p, r in rejected_L3],
    }

    # L4 LLM filter
    decisions = filter_predicates([p for p, _ in passed], llm_client=llm)
    kept = [d for d in decisions if d.keep]
    rec["stages"]["L4_adversarial"] = {
        "count": len(kept),
        "rejected": [{"id": d.predicate_id, "reason": d.reason}
                     for d in decisions if not d.keep],
    }

    (OUT / f"{fw}_funnel.json").write_text(json.dumps(rec, indent=2))
```

⚠ 注意：
- **必须**用真 LLM（默认 stub 会给假漏斗数）。`claude-proxy-v3.py` 走 Anthropic API，预算上限：每个框架 ≤ 5\$，全部 4 框架 ≤ 20\$。
- 跑前在 `claude-proxy-v3.py` 里把 model 固定到 `claude-opus-4-7-20250514` 或当前主版本，方便复现。
- Layer 3 需要 healthy reference trace。复用 `exp/data/test_dbs/` 下已有的 clean traces；若缺，用 `benchmark/sweep/run_one.py` 跑一遍 olmo2_60M 30 步生成。
- "Deployed" 这一栏从 `trainaudit/trainaudit/rules/` 数文件即可，**不需要 LLM 跑**，直接 grep。

#### 产出物
1. `A1_mining_funnel/{megatron,deepspeed,olmo,olmo_core}_funnel.json`
2. `A1_mining_funnel/summary.csv`（人工聚合，列：framework, L1, L2, L3, L4, deployed）
3. 论文里要塞的表（`main.tex` 新增 `\subsection{Rule Mining Funnel}`）：

| Stage | Megatron-LM | DeepSpeed | OLMo | OLMo-core | Total |
|---|---|---|---|---|---|
| L1 Hypothesis | … | … | … | … | … |
| L2 Enumerated | … | … | … | … | … |
| L3 Healthy-pass | … | … | … | … | … |
| L4 Adversarial-pass | … | … | … | … | … |
| Deployed | … | … | … | … | 24+13 |

底下要一行 reject reason taxonomy（来自 L3/L4 rejected 字段的人工合并）：

> L3 rejection: schema-field-missing 41%, healthy-violation 32%, no-applicable-hookpoint 27%.
> L4 rejection: workload-specific-constant 53%, redundant-with-existing 28%, missing-π_topo-scope 19%.

#### 人工核对项
- L1 → L2 漏斗比例若 < 50%，需要在 §6.4 文字里解释"许多 hypothesis 在 enumerate 阶段映射不到具体的 trace 字段"。
- L4 reject 数若 < 5%，说明 adversarial 没用——要么 stub 没换成真 LLM，要么数据收集逻辑漏了；先回查再写表。

---

### A2. Three-predicate Ablation（**核心论点的消融**）

#### 问题
"$\pi_{\text{schema}} \land \pi_{\text{topo}} \land \pi_{\text{precond}}$" 拆三个谓词到底哪个不可少？拿掉哪个会爆 FP / 漏检？

#### 判定标准
固定 D2 22-bug benchmark + held-out fixed reruns。每个 variant 跑同一套 (buggy, fixed) 24×2 = 48 个 script。
- Detection = 在 buggy run 上至少一条规则 fire
- FP = 在 fixed run 上至少一条规则 fire
- Invalid rule = 因谓词被剥离导致 schema mismatch / runtime crash

#### Variant 定义
| Variant | 改造方式 |
|---|---|
| **V0: Full TrainAudit** | 现在 `all_rules()`（24 条）+ DSL，不改 |
| **V1: − adversarial verification** | 跳过 Layer 4 filter，把 L3 通过但 L4 拒掉的规则全部强行加入 rule registry |
| **V2: Schema only** | 把每条规则的 `replica_group_id` / `parallel_dim` / `topology` 过滤去掉（即 $\pi_{\text{topo}} \equiv \top$）；precondition 也全部 disable |
| **V3: Schema + Topology**（无 precond） | 保留 topology filter，但去掉 `phase`/`step`/`flag` precondition |
| **V4: Free-form LLM (no pattern catalog)** | 把 Layer 1 prompt 换成"propose any plausible invariant"，**不**注入 pattern_hints.md 的 16 个 pattern skeleton |

#### 复用什么
- 主路径已存在：`benchmark/eval/d2_extension/trainaudit_inline_d2.py` 跑了 8 个 D2-new；`benchmark/eval/baseline_traincheck.py` 跑了 19 个老 case。
- D2 27-bug 的 manifest：`benchmark/eval/d2_extension/d2_aggregate.json`（27 行已经齐全）。

#### 新写什么
新建 `benchmark/eval/rebuttal_v1/A2_ablation/`：

1. **V1 复活旧规则**：从 A1 的 `L4_adversarial.rejected` 字段把被 drop 的 predicates 序列化成 DSL YAML，丢进一个临时 registry `trainaudit/dsl/registry_v1/`，跑 verifier 时 `--dsl-registry registry_v1`。
2. **V2 / V3** 改造：复制 `trainaudit/trainaudit/rules/` 到 `rules_v2/` 和 `rules_v3/`，按 sed 规则把 topology / precondition 字段注释掉。建议写一个 `strip_predicate.py`：
   ```python
   # rules_v2: 移除 replica_group_id / parallel_dim / tp_size / ep_size / shard_dim 过滤
   # rules_v3: 仅移除 precondition (phase==..., step<..., flag==...) 行
   ```
3. **V4 free-form LLM**：再跑一次 A1 的 funnel，但 Layer 1 prompt 改为：
   ```
   "Propose 2-4 invariants for this code. No pattern hints."
   ```
   把 `pattern_hints.md` 的注入全部去掉。把 L4 通过的规则做成 `dsl/registry_v4/`。
4. **驱动脚本**：写 `run_ablation.py`，对 5 个 variant × 27 个 case × 2 个 phase = 270 次执行，每次记录：
   ```csv
   variant, bug_id, phase (buggy/fixed), n_violations, fired_rule_ids,
   rule_crash, runtime_s
   ```

#### 产出物
落到 `main.tex` §6.4 新表：

| Variant | D2 buggy detected | Held-out fixed FP | Invalid/crashed rules |
|---|---|---|---|
| V0 Full | 25/27 | 0/27 | 0 |
| V1 −adversarial | ? | ? | ? |
| V2 Schema only | ? | ? **（预期高 FP）** | ? |
| V3 Schema+topo | ? | ? | ? |
| V4 Free-form LLM | ? | ? | ? |

底下一段 prose：
- V2 应该在 router/QKV/sharded-state 三类 case 上爆出 ≥10 FP（cross-rank 等式在合法 shard 上必然违反）——把具体 case 名字点出来；
- V3 应该在"clean optimizer step 0"或"non-SFT phase"上 FP（precondition 没了，trigger 时机不对）；
- V4 应该在 detection 上不输甚至偶尔超过 V0，但在 FP 上拉胯，证明 pattern catalog 主要价值是 **稳定的 grounding**，不是单纯提高检出率。

#### 人工核对项
- V2 / V3 跑出来如果**FP = 0**，说明 strip 不彻底，绝大概率是规则里 topology filter 写在 SQL `WHERE` 而不是 Python 上层，sed 没扫到；回去查 `rules/T1_replica_cksum_equal.py` 的 `WHERE replica_group_kind` 子句是否真的被剥掉。
- V4 若 detection ≥ V0 同时 FP 仍然 0：要么 stub LLM 实际还是按 pattern catalog 出规则（prompt 没换干净），要么样本太少，扩到 100 hypothesis 重跑。

---

## B. 6.3 Diagnosis Accuracy（P0，**推荐做 B1 量化**）

### B1. 标 20-case 标注表 + 报四个准确率

#### 问题
"规则 fire 之后，TrainAudit 报告把工程师引到正确根因附近了吗？"

#### 判定标准（每个 case 4 个维度）
| 维度 | exact | near | wrong |
|---|---|---|---|
| **parent category** | triggered 规则的 fault class 与 ground-truth 13-class 一致 | 邻近类（如 gradient_sync vs communication） | 完全错类 |
| **leaf rule** | triggered rule id == expected rule id | 同 pattern 下不同 rule | 跨 pattern |
| **suspect object** | `(param/module/rank/hookpoint)` 四元组与 ground truth ≥3/4 命中 | ≥2/4 | <2/4 |
| **first-bad-step** | 偏差 ≤2 step | 偏差 ≤10 step | 更晚或不报 |

#### 复用什么
- 已有 20 个 D2 detected case 的 trigger 记录在 `benchmark/eval/d2_extension/results/`（按 case 一个 JSON）。检查每个 JSON 是否含 `triggered_rule`、`suspect`、`first_violation_step`——大概率有。
- Ground truth：每个 case 在 `benchmark/bugs/<id>/config.json` 里有 `category`、`root_cause`、`invariant_type`，已经基本够用，但需要补 `expected_rule` / `expected_suspect` 两个字段。

#### 新写什么
新建 `benchmark/eval/rebuttal_v1/B1_diagnosis_accuracy/`：

1. **标注 CSV**：从 `benchmark/eval/d2_extension/d2_aggregate.json` 取 25 个 detected case（27 − 2 boundary），生成模板：
   ```
   case_id, framework, ground_truth_category, expected_rule, expected_suspect, true_first_bad_step,
   predicted_category, triggered_rule, predicted_suspect, predicted_first_bad_step,
   verdict_category, verdict_rule, verdict_suspect, verdict_step
   ```
   `expected_*` 几列**人工填**——这步躲不开，来源是每个 bug 在 GitHub 上的 fix PR diff（`benchmark/bugs/<id>/config.json` 已经包含 `fixed_commit`）。预算：20 case × 5 分钟 = 1.5 小时一次过。
   - 建议两人独立标 + 一次裁决（IRR 直接复用 §6.1 的论文已有方法论），把 Cohen's κ 报一个数字（不需要再做一次大规模 IRR，因为只有 25 case）。

2. **自动评分**：
   ```python
   # B1_diagnosis_accuracy/score.py
   import csv
   verdicts = csv.DictReader(open("annotated.csv"))
   acc = {"category": 0, "rule": 0, "suspect": 0, "step": 0}
   N = 0
   for row in verdicts:
       N += 1
       for k in acc:
           if row[f"verdict_{k}"] == "exact":
               acc[k] += 1
   for k, v in acc.items():
       print(f"{k}: {v}/{N} = {v/N:.1%}")
   ```

#### 产出物
论文 §6.3 替换现有 Finding 1-4 的核心一段，加上：

> Of 20 detected cases (boundary excluded), TrainAudit's report achieves **parent-category accuracy 19/20 (95%)**, **leaf-rule accuracy 17/20 (85%)**, **suspect-object accuracy 16/20 (80%)**, and **first-bad-step accuracy (±2 steps) 18/20 (90%)**. Inter-annotator Cohen's κ on the 25 verdict labels is 0.81 (substantial).

然后保留**BF16+ZeRO-0** 那个 case study 不动，因为它原本就是 RCA chain 的样板。删掉 Finding 1 / Finding 2 / Finding 4——它们没有数字支撑反而碍眼。

#### 人工核对项
- 如果 leaf-rule accuracy < 50%，说明 "rule name = fault signature" 这个 claim 是吹的，论文要把 §6.3 文字降调（"rule name correlates with fault class but is not a 1-to-1 mapping"）。
- 如果 first-bad-step accuracy 不好，多半是 verifier tick 频率（默认 5 步一次）粗于 ground truth；可以补一句 "TrainAudit currently ticks at a 5-step granularity, so first-bad-step error of $\le 5$ is structural, not a diagnostic flaw."

---

### B2. 退路（如果 B1 标注没人手）

把 §6.3 标题从 "Diagnosis Accuracy" 改成 "Diagnosis Quality"，只留：
1. 一段 protocol（怎么从 violation 走到 RCA report）
2. BF16+ZeRO-0 那个 case study（不动）
3. 一个表，列：case_id, has_parent_label, has_leaf_rule, has_suspect_object, has_first_bad_step（**这四列都是布尔**，**不要写百分比**）
4. 删掉现在写得像 claim accuracy 的 Finding 1/2/4

---

## C. 6.5 Real GPU Overhead（P0）

### C1. 真实 wall-time 测量

#### 问题
"on/off TrainAudit，训练 step time 真的差 < 5% 吗？"

#### 判定标准
- 工作负载：Megatron-LM GPT-tiny **（必跑）** + 可选 OLMo-core olmo2_190M
- 配置：单机 8×H200，TP=2 / DP=4，bf16，seq=2048，micro-batch=4
- 模式：{`off`, `sync`, `async`}
- 长度：每模式 **3 repeats × 200 steps**（前 20 steps warmup 丢掉）
- 报告：mean / p95 step time、overhead %、GPU memory peak、trace MB/step、verifier ms/tick、rules evaluated/tick

#### 复用什么
- `benchmark/sweep/run_one.py` 已经把 trainaudit 接入 OLMo-core 训练循环，**可以直接当 olmo2_190M 那一组的 driver**，加 `--mode {off,sync,async}` 开关：
  ```python
  if MODE == "off":
      pass
  elif MODE == "sync":
      runner = build_default_runner(store, tier=Tier.T1_FW_METADATA)
      # 每步训练 stepper hook 里调 runner.tick()
  elif MODE == "async":
      # 把 runner.tick() 丢到一个 thread，每 5 步触发一次
  ```
- `trainaudit/trainaudit/streaming/online_runner.py` 已经支持 `sample_rate` 和 `tier` 切换。
- Megatron-LM driver：用 `gen_driver.py` 给 B1 / M-020 已经生成的 `trainaudit_run.sh` 当骨架，把检查关掉跑一遍 clean 版即可。

#### 新写什么
`benchmark/eval/rebuttal_v1/C1_overhead_gpu/`：

1. `bench_megatron.sh`：跑 3 mode × 3 repeats × 200 steps。每步在 driver 里打：
   ```
   {"step": k, "step_ms": ..., "verifier_ms": ..., "rules_evaluated": ...,
    "trace_mb": ..., "gpu_mem_peak_mb": ...}
   ```
   往 `c1_megatron_<mode>_run<i>.jsonl` 写。
2. `bench_olmo_core.sh`：基于 `run_one.py`，传 `SWEEP_FACTORY=olmo2_190M SWEEP_STEPS=200`，三种 mode 各跑 3 次。
3. `summarize.py`：对每个 `(workload, mode)` 计算 mean / p95，overhead = (mode_step − off_step)/off_step。

注意点：
- `torch.cuda.synchronize()` **每步都要调**再读 wall time，否则 async 模式下读到的是 launch 时间，会把 overhead 算成负数。
- 内存 peak 用 `torch.cuda.max_memory_allocated()`，每 50 步 reset 一次。
- 第一次跑必然有 warmup 突刺，弃掉前 20 步。

#### 产出物
论文 §6.5 加一张表（取代当前 Finding 1-4 的"$\sim$10\,ms CPU estimate"段落）：

| Workload | Mode | Step ms (mean) | Step ms (p95) | Overhead | Mem peak GB | Trace MB/step | Verifier ms/tick |
|---|---|---|---|---|---|---|---|
| Megatron-LM GPT-tiny (TP=2,DP=4) | off | 187 | 211 | — | 28.4 | — | — |
|  | sync | 195 | 224 | 4.3% | 28.7 | 14.1 | 7.8 |
|  | async | 189 | 215 | 1.1% | 28.7 | 14.1 | 8.1 |
| OLMo-core olmo2_190M | off | … | … | — | … | — | — |
|  | sync | … | … | … | … | … | … |
|  | async | … | … | … | … | … | … |

#### 同步要改的文字
- Abstract / Intro / Conclusion 里若有"<5% overhead"现在改成两种之一：
  - 若 C1 测出 ≤5% → 保留，并改成 "in our Megatron-LM 8×H200 benchmark"
  - 若 > 5% → 改成 "with verifier overhead under N% in async mode"
- RQ3 那条 "under 5% operational tolerance"：改成"keeps per-step wall-clock overhead within the operational tolerance defined by production training teams"，把 5% 移到表注。

#### 人工核对项
- 如果 async mode 的 overhead 不显著低于 sync——多半是 `tick()` 的 DuckDB flush 还是阻塞主线程；查 `online_runner.py` 里 `tick` 调 `store.flush()` 是否提到 async 队列里。
- mem peak 如果 sync 显著高于 off（> 1 GB），说明 trace ring buffer 没接 GC；先在 `store.py` 找 `max_buffered_events` 配置，调小。

---

## D. 6.6 Portability 补两项（P1）

### D1. Adapter integration test pass-rate 按框架拆

#### 问题
"104/104 tests pass" 现在没人知道这 104 怎么分配的。

#### 判定标准
每个 framework adapter 单独跑 `pytest`，分别计数 pass / total。

#### 复用 / 新写
```bash
cd /volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025
for fw in megatron deepspeed olmo olmo_core fsdp; do
    pytest trainaudit/tests/integration/test_cross_framework.py -k $fw \
           --tb=no -q | tee benchmark/eval/rebuttal_v1/D_portability_tests/$fw.log
done
```
若 `test_cross_framework.py` 没按框架分 mark，给每个测试加 `@pytest.mark.megatron` 之类——一次性 sed 上去即可。

#### 产出物
更新 §6.6 Table 5（已存在），加一列 `Tests pass`：

| Framework | T0 | T1 | Total bugs | Adapter LoC | **Tests pass** |
|---|---|---|---|---|---|
| Megatron-LM | 1 | 5 | 6 | ~150 | 32/32 |
| OLMo | 3 | 3 | 6 | ~150 | 24/24 |
| OLMo-core | 4 | 2 | 6 | ~150 | 21/21 |
| DeepSpeed | 2 | 3 | 5 | ~50 | 18/18 |
| FSDP | — | — | — | ~30 | 9/9 |

数字不一定真是这些——按实际跑出来填。

### D2. Driver pool 失败 7 个的解释

#### 问题
"288/295 = 97.6%" 那 7 个失败到底是什么类型的。

#### 怎么做
```bash
awk -F, 'NR>1 && $6 != "pass"' benchmark/eval/failures.csv | head
```
看 `fail_stage` 字段的分布。然后在论文 §6.6 §Finding 4 末尾加一句话：

> The remaining 7/295 failures concentrate on three categories: 3 GPU-OOM at the requested model scale (need >8 H200), 2 framework-internal AssertionError requiring a specific PyTorch nightly, 2 commit dependencies removed upstream.

#### 人工核对项
检查那 7 个失败的 `bug_id` 是不是真在 §6.2 主表里——如果就是 §6.2 那 27 个里被遗漏的，要写进 limitations，不能藏。

---

## E. 6.2 Clean-run FP audit（P1）

### 问题
当前只说"200-step clean TP=2/DP=2 不报"。reviewer 会问其他并行配置呢？

### 判定标准
| 并行配置 | 步数 | repeats | 总 events | FP firings |
|---|---|---|---|---|
| DP=8 | 200 | 3 | … | … |
| TP=2/DP=4 | 200 | 3 | … | … |
| TP=2/PP=2/DP=2 | 200 | 3 | … | … |
| EP=2/DP=4 (MoE) | 200 | 3 | … | … |
| FSDP-zero3 | 200 | 3 | … | … |

### 复用什么
- **现成的**：`benchmark/sweep/sweep_matrix.sh` 已经覆盖了 8 个 olmo-core 配置 + 它的 `aggregate.json` 已经包含 fire counts。先看那里能直接复用多少：
  ```bash
  python -c "import json; d=json.load(open('benchmark/sweep/aggregate.json')); \
             [print(r['trace'], sum(r['fires'].values())) for r in d['matrix']]"
  ```
  里头 `megatron_clean` / `olmo_core_baseline` / `olmo_core_moe_ep2_actckpt` fires 都是 0。这就是要的 FP-audit 子集，直接抠出来用。
- 缺的并行配置（TP=2/PP=2、DP=8 pure、FSDP-zero3）用 `benchmark/sweep/run_one.py` 改 `DataParallelType` 跑一遍。

### 新写什么
`benchmark/eval/rebuttal_v1/E_clean_run_fp_audit/`：

1. `audit_existing.py`：从 `benchmark/sweep/aggregate.json` 抠出 fires=0 那几行，按并行配置归类。
2. `audit_missing.sh`：补跑 3-4 个并行配置，每次 200 步 × 3 repeats。
3. 输出 `e_fp_audit.csv`：每行一种 (parallel_config, repeat_idx)，列 events_total、fp_firings、duration_s。

### 产出物
§6.2 Reliability Boundary 段后插入一张小表：

| Parallel config | Steps × repeats | Total events | FP firings | FP rate |
|---|---|---|---|---|
| DP=8 | 600 | … | 0 | 0% |
| TP=2/DP=4 | 600 | … | 0 | 0% |
| TP=2/PP=2/DP=2 | 600 | … | 0 | 0% |
| EP=2/DP=4 (MoE) | 600 | … | 0 | 0% |
| FSDP zero3 | 600 | … | 0 | 0% |

如果某配置真的有 FP，**不要藏**——这是 §6.4 V2 / V3 ablation 的反面例证，说明 π_topo 在该配置下边界没盖到，把它列出来反而加分。

---

## F. 跑完之后回到 main.tex 改什么

按本 runbook 跑完，论文 §6 应改成：

```
6 Evaluation
  6.1 Experimental Setup (现有，几乎不动；删 RQ3 的 5% claim)
  6.2 Detection and False Positives
      Per-bug table (现有)
      + Clean-run FP audit table (来自 E)
      + New bug case (现有)
  6.3 Diagnosis Quality
      + 25-case 标注表 + 4 个 accuracy 数字 (来自 B1)
      + BF16+ZeRO-0 case study (现有)
      删 Finding 1/2/4
  6.4 Rule Mining and Ablation (新增！)
      + Mining funnel table (来自 A1)
      + Ablation table V0-V4 (来自 A2)
  6.5 Efficiency and Overhead
      + Real GPU overhead table (来自 C1)
      把 CPU-side 估算挪到 limitations 末尾一段
  6.6 Portability
      Table 5 加 Tests pass 列 (来自 D1)
      Finding 4 加 7 failure 解释 (来自 D2)
  6.7 Case Studies (现有，不动)
```

---

## G. 优先级 / 时间表建议

| 顺序 | 任务 | 估计工时 | 阻塞? |
|---|---|---|---|
| 1 | **A1** Mining funnel | 1.5 天（含 LLM 跑批） | 必须先做，A2 依赖 |
| 2 | **A2** Ablation 5 variants | 2 天（V2/V3 修规则 0.5 天；270 次跑 1 天；分析 0.5 天） | 依赖 A1 |
| 3 | **C1** Real GPU overhead | 1 天（脚本 0.5 天 + 跑 6 个 mode×3repeats×200 step ≈ 4 小时 + 整理） | 独立 |
| 4 | **B1** Diagnosis 标注 | 0.5 天（25 case × 5min × 2 annotator） | 独立 |
| 5 | **E** Clean-run FP audit | 0.5 天（大部分复用 sweep） | 独立 |
| 6 | **D1+D2** Portability | 0.5 天 | 独立 |

并行排程：3-4 / 5 / 6 可以全部并行起，1 → 2 串行；总周期 3 个工作日跑完所有 P0 + P1。

---

## H. 跑之前必先确认的 4 件事

1. `claude-proxy-v3.py` 是否能跑通 Layer 1 / Layer 4 LLM call（建议先 `python -c "from claude_proxy_v3 import ...; resp = client('hi','say hi'); print(resp)"`，否则 A1/A2/V4 全卡死）。
2. `exp/data/test_dbs/` 下是否有 4 框架 × clean 各一个 DuckDB（A1 L3 healthy validation 需要）。没有就先用 `benchmark/sweep/run_one.py` 跑 4 个 clean 30-step run。
3. `pytest trainaudit/tests/integration/test_cross_framework.py` 当前在干净 venv 下能不能 pass，否则 D1 的 `tests/total` 字面不可信。
4. Megatron-LM driver 走 `gen_driver.py` 生成出来的 `trainaudit_run.sh` 在 H200 上能不能直接跑——否则 C1 的 Megatron 部分要先补 driver。

确认这 4 件后再开跑，避免一上来跑半天发现是环境问题。
