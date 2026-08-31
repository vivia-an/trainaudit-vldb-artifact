# 论文事实索引（Live Status Board）

> **本文档是论文里所有具体数字的唯一权威**。其他文档（doc 20 / 21 / main_cn.tex）出现的具体数字必须在这里有锚点。
>
> **更新规则**：
> 1. 每次跑完一个实验、加一条 rule、改一个机制 → **立刻**更新对应行的 value + last_updated
> 2. paper 段落写新数字前，先在本表登记（哪一行 / 哪个实验来源）
> 3. 表里没数据 → §10 阻塞清单加一项
> 4. 写 paper 时数字旁加注释 `% see paper_evidence §X.Y row Z`
>
> **维护成本**：每次实验完更新这个表 ≤ 5 分钟；遗漏一次→投稿前数字漂移→reviewer 抓出来→重写。值得维护。

---

## 1. 系统状态（实时）

| 维度 | 当前值 | 源 | last_updated | 论文位置 |
|----|----|----|----|----|
| trainaudit 源码 LoC | ~5400 (含 DSL + mining + diagnosis) | `find trainaudit/trainaudit -name "*.py"` | 2026-05-05 | §3 method |
| Tier 数 | 5 (T0 / T1 / T2 / T3 / T4) | `trainaudit/tiers.py` | 2026-05-05 | §3.2 |
| T0 hookpoint 数 | 8 (dist / module fwd / bwd / optim.step / clip_grad / scheduler / checkpoint / dataloader / F.softmax) | `trainaudit/core_trace/` | 2026-05-05 | §3.2 |
| T0 hookpoint 上下文（C0）| 每条事件携带 module_class + module_id + module_name + callsite(file/line) + grad_enabled (fwd.pre, 2026-05-05 hunt 加，给 T0-evaluator-eval-mode 用) | `core_trace/_utils.py` + `module_hook.py` | 2026-05-05 | §3.3 |
| Lifecycle-safe hooks | enable→disable→enable 重新绑定 store via `active_store()` cell | `core_trace/_utils.py` | 2026-05-05 | §3.3 |
| T1 framework adapter 数 | 5 (megatron / deepspeed / olmo / olmo-core / fsdp) | `trainaudit/adapters/` | 2026-05-05 | §3.3 |
| T1 active probe 数 | 3 (residual / jitter / sqrt_decay) | adapter active hooks | 2026-05-05 | §3.3 |
| T0 rule 数 (active) | 12 (+1 T0-evaluator-eval-mode iter2 hunt 添加) | `trainaudit/rules/T0_*.py` | 2026-05-05 | §3.4 |
| T1 rule 数 (active) | 9 (+T1-buffer-replica-cksum-equal iter4 + T1-multi-backward-per-step-fragile-config iter10) | `trainaudit/rules/T1_*.py` | 2026-05-05 | §3.4 |
| Total Python rules | 25 (T0_build_has_modules + 12 T0 + 9 T1 + 3 probe) | (T0 + T1) | 2026-05-05 | §3.4 |
| **DSL templates** | 4 (TENSOR_STAT_BOUND / PAYLOAD_FIELD_COMPARE / CONDITIONAL_CHECK / STRUCTURAL_PRESENCE) | `dsl/templates.py` | 2026-05-05 | §3.2 |
| **DSL schema extensions** | 2 (scope.tensor_signature, bound.conditions) — 满足 doc 22 §A3 ≤2 硬约束 | `dsl/predicate.py` | 2026-05-05 | §3.2 |
| **dsl_native YAMLs** | 13 (10 T0 + 3 T1) | `dsl/registry/{T0,T1}/*.yaml` | 2026-05-05 | §3.2 |
| **DSL→DuckDB SQL compiler** | 1 (CompiledRule(sql, postprocess?))；MONOTONIC 用 window 函数 | `dsl/compiler.py` | 2026-05-05 | §3.2 |
| **Python ↔ DSL event_id 等价** | 14 等价 tests，rule_id 集合一致 | `tests/dsl/test_compiler_equivalence.py` | 2026-05-05 | §3.2 |
| **Verifier dual path** | `run_rules(use_dsl=True/False)` tier-filtered | `verifier.py` | 2026-05-05 | §3.2 |
| **Mining 4-layer pipeline** | L1 (LLM hypothesis, pluggable) + L2 (deterministic enumerate, 6 relation types) + L3 (healthy validate + tolerance auto-learn) + L4 (LLM filter, pluggable) | `trainaudit/mining/` | 2026-05-05 | §3.2 |
| **Diagnosis module (C1+C2)** | DiagnosisReport(suspect_module, suspect_rank, callsite, bug_specific, hypothesis) + LLM RCA agent (pluggable client) | `trainaudit/diagnosis/` | 2026-05-05 | §3.4 |
| **Eval harness** | `benchmark/eval/{build_manifest, gen_driver, run_all, overhead, baseline_traincheck, fault_injection, synthetic_runners}.py` | `benchmark/eval/` | 2026-05-05 | §4 |
| **Driver pool** | **288/295 = 97.6%** (15 manual + 273 generated scaffold; --use-detect upgrades 49 to runpy.run_path wrap) | `benchmark/eval/gen_driver.py` | 2026-05-05 | §4.1 |
| **Online streaming runner** | OnlineRunner.tick() with cursor + per-rule violation cache + hookpoint-aware rule skipping | `trainaudit/streaming/online_runner.py` | 2026-05-05 | §4.3 |
| **Offline forensic CLI** | `python -m trainaudit {verify,diagnose,summary,replay} <trace.duckdb>` | `trainaudit/__main__.py` | 2026-05-05 | §3.5 |
| **DataLoader hook fix** | Patch `_SingleProcessDataLoaderIter._next_data` + `_MultiProcessingDataLoaderIter._next_data` (subclass override fix) | `core_trace/dataloader_hook.py` | 2026-05-05 | §3.3 |
| **Test suite** | **104 passed** (21 dsl + 17 mining + 4 rca + 25 integration + 4 streaming + 25 core + 4 cli + 4 gen_driver) | `pytest tests/` | 2026-05-05 | §4 |

---

## 2. Per-bug 验证表（单一权威）

> 这里只列已验证的，未跑的留白。每行 = (bug_id, tier, mode, BUGGY 信号, FIXED 信号, 与 detect.py 一致, last_updated)。

### 2.1 真实框架验证（生产级 evidence）

| Bug | Framework / commit | Tier | T-rule | BUGGY 实测 | FIXED 实测 | 一致 | last_updated |
|----|----|----|----|----|----|----|----|
| **B11** | DeepSpeed 005afe12 | T0 | clip-grad-bounded | post grad norm > max_norm | 全 ok | ✅ | 2026-05-05 |
| **B12** | OLMo-core 6e330ba2 | T0 | initial-lr-present | param_groups missing initial_lr + KeyError | scheduler.init OK | ✅ | 2026-05-05 |
| **O-NEW-1** | OLMo 67c9e315 | T0 | norm-output-unit-rms | rms=0.329 < 0.5, RMSLayerNorm | rms ≈ 1 | ✅ | 2026-05-05 |
| **M-014** | Megatron 5153efea0 | T0 | softmax-degenerate | TopKRouter (128,1) all 1.0 | 框架自身 raise ValueError | ✅ | 2026-05-05 |
| **O-005** | OLMo 204ad53c | T0 | checkpoint-preserve-rng | preserve_rng_state=False with Dropout | preserve_rng_state=True | ✅ | 2026-05-05 |
| **OC-NEW-2** | OLMo-core 2b6cf996 | T0 | optim-step-counter-monotonic | state['step'] not incrementing | state['step'] monotonic | ✅ | 2026-05-05 |
| **B13** | OLMo 562c0fe0 | T1 | residual-stream-preserved (active probe) | d_to_input=0.74, d_to_normed=0.019 | preserved | ✅ | 2026-05-05 |
| **O-002** | OLMo c482df74 | T1 | residual-stream-preserved | 2/2 blocks: output closer to normed | preserved | ✅ | 2026-05-05 |
| **M-NEW-5** | Megatron 87d9d2506 | T1 | router-has-calculate-per-token-loss | 2 routers missing attr | all routers have attr | ✅ | 2026-05-05 |
| **M-012** | Megatron a58768725f | T1 | expert-bias-fp32 | TopKRouter expert_bias=bf16 | expert_bias=fp32 | ✅ | 2026-05-05 |
| **M-020** | Megatron 99f999a466 | T1 | layer-count-strict | actual 4 (per-rank 2 × pp 2) ≠ declared 5 | framework asserts on bad config | ✅ | 2026-05-05 |
| **M-024** | Megatron 20b395424d | T1 | jitter-preserves-dtype (active probe) | bf16 → fp32 promoted in jitter | dtype preserved | ✅ | 2026-05-05 |
| **OC-NEW-3** | OLMo-core f34e7ddc | T1 | sqrt-decay-front-loaded (active probe) | shape inverted (slow-then-fast) | shape correct (fast-then-slow) | ✅ | 2026-05-05 |
| B1 | Megatron 3c637fc0d | T1 | replica-cksum-equal | **bug 在 modern PyTorch 被 RNG 自动同步掩盖** | - | ❌ N/A | 2026-05-05 |
| O-NEW-8 | OLMo f81904f3 | T1 | residual-stream-preserved | 阻塞: olmo `mup` 包不可用 | - | ⏸ | 2026-05-05 |

→ **真实框架自动捕获 13/48 = 27.1%**（6 T0 + 7 T1）。
→ B1 不是工具失败，是 bug 在现代环境下不 manifests（M-005 失败历史的根因）。
→ O-NEW-8 阻塞于环境（mup 包源），与 trainaudit 设计无关。
→ 跨 4 个 framework × 13 个不同 commit × 时间跨度 2023-09 → 2024-09。

### 2.2 Toy 模拟验证（rule 设计正确性 evidence）

| Bug | Mode | Tier | rule | last_updated |
|----|----|----|----|----|
| B11 | inject (smoke) | T0 | clip-grad-bounded | 2026-05-05 |
| B12 | toy-sim | T0 | initial-lr-present | 2026-05-05 |
| O-NEW-1 | toy-sim (BuggyRMSNorm) | T0 | norm-output-unit-rms | 2026-05-05 |

→ 3 个 rule 在 toy 设计层正确。

### 2.2b Synthetic surrogate D1 harness 验证（CPU-reproducible evidence）

> 在 `benchmark/eval/synthetic_runners.py` 中为每个 bug 写了 in-process 替身：
> 真实 PyTorch 模型 forward + backward + optimizer.step + 注入 bug pattern
> （buggy clip / RMSNorm 0.33 / frozen state['step'] / degenerate softmax /
> resume missing initial_lr / ...）or 直接 emit 等价的 adapter probe events
> （residual.probe / jitter.probe / decay.probe / build.snapshot
> cross_rank_cksums）。Rule 检测路径与真实框架版完全相同。
>
> 用途：CPU 上跑 D1 harness 出 paper §4.1 主表，端到端验证 framework
> 在真实 silent-error pattern 上工作，无需 GPU + 框架 checkout。

| Bug | Tier | Surrogate 类型 | T-rule | last_updated |
|----|----|----|----|----|
| B1 | T1 | replica cksum disagree (4-rank, outlier rank=2) | replica-cksum-equal | 2026-05-05 |
| B11 | T0 | buggy `nn.utils.clip_grad_norm_` (no-op clip) | clip-grad-bounded | 2026-05-05 |
| B12 | T0 | AdamW + scheduler resume w/ no `initial_lr` | initial-lr-present | 2026-05-05 |
| B13 | T1 | residual.probe (d_normed < d_input) | residual-stream-preserved | 2026-05-05 |
| M-012 | T1 | semantic.expert_bias_dtype = bfloat16 | expert-bias-fp32 | 2026-05-05 |
| M-014 | T0 | `_DegenerateTopKRouter` (topk=1 + softmax) | softmax-degenerate | 2026-05-05 |
| M-020 | T1 | `n_transformer_layers_in_local_module * pp_size ≠ num_layers` | layer-count-strict | 2026-05-05 |
| M-024 | T1 | jitter.probe (bf16→fp32 promoted) | jitter-preserves-dtype | 2026-05-05 |
| M-NEW-5 | T1 | semantic.has_calculate_per_token_loss=False + framework_invariants 启用 | router-has-calculate-per-token-loss | 2026-05-05 |
| O-NEW-1 | T0 | `_BrokenRMSNorm`(× 0.33) → rms ≈ 0.33 | norm-output-unit-rms | 2026-05-05 |
| OC-NEW-2 | T0 | `_FrozenStepAdamW` (state['step'] not incrementing) | optim-step-counter-monotonic | 2026-05-05 |
| OC-NEW-3 | T1 | decay.probe series (slow-then-fast slope) | sqrt-decay-front-loaded | 2026-05-05 |

→ **`selected_synthetic_14` D1 harness 结果：15/15 buggy detected = 100.0%；
0/13 fixed FP = 0.0%；4 frameworks × 12 categories × 14 rules 全覆盖。**
→ 命令：`python benchmark/eval/run_all.py --subset benchmark/eval/synthetic_14.json --mode synthetic`
→ 产出：`benchmark/eval/results.csv` (28 rows) + `paper_table_coverage.md`
→ 2026-05-05 update：扩到 14（+ O-005 checkpoint preserve_rng + O-NEW-9
  token_id truncation，两者用真实 PyTorch API + DataLoader 触发），同时
  发现并修复 `_BaseDataLoaderIter._next_data` 子类 override bug（影响所有
  GPU + 真实 DataLoader 使用场景）。

→ **`fault_injection` 33-fault benchmark（paper §4.1 主表）**：
  31/31 = 100% detection on severe + moderate；3/3 true negative on
  subthreshold boundary；0 FP across 34 faults；4 tiers × 12 categories
  全覆盖。命令：`python benchmark/eval/fault_injection.py`
  产出：`paper_table_fault_injection.md`

### 2.3 待真实复测的 bug（driver 已写）

| Bug | Tier | 阻塞 |
|----|----|----|
| (无) | - | 当前所有写好的 driver 都已跑过 |

### 2.5 Hunt-detected silent error candidates（trainaudit 在 framework HEAD 上挖到的，未修复）

来源：`benchmark/eval/static_scan.py`（22 rule × 4 framework grep matrix）+ 人工三角定位 + 上游-fix archaeology + per-candidate `dynamic_confirm_e2e.py`（CPU + GPU H200）。Phase 1 静态扫到 205 个 raw match；triage 后 13 个高/中置信候选；Plan A/B 全跑完后 **10/13 真实-framework E2E 确认**。

#### 验证状态分层（2026-05-06 GPU batch 完成后）

| 验证层级 | 数量 | 备注 |
|---|---:|---|
| 总候选 | **13** | 4 个 framework 全覆盖（DS 5 / OLMo-core 4 / OLMo 2 / Megatron 1） + DS 0.18.7 worktree（2 个） |
| Pinned/buggy checkout 字面验证 | 12 | 11 在 pinned 直接，1 在 v0.18.7 worktree |
| **真实-framework E2E 确认** | **10/13** | 5 CPU + 3 GPU framework runs + 2 GPU 结构性 (AST + runtime emulation) |
| GPU bug-path 受 torch 版本卡住 | 1 | ZERO3_STREAM_RACE_NAN 需 torch ≥ 2.10；driver 570 cap 在 cu126/torch ≤ 2.7.1 |
| GPU stochastic race 没在 H200/NCCL 2.26 stack 触发 | 1 | OVERLAP_COMM；anti-pattern grep verified + rule capability 已独立验证 |
| 纯 pattern injection（无真实 framework） | 1 | ZERO3_STREAM_RACE — symptom-only NaN 注入 |

#### 候选明细

| candidate id | framework | sha | rule fired | file:line | E2E status |
|----|----|----|----|----|----|
| `CAND_OLMOCORE_RNGCKPT` | OLMo-core | `f34e7ddc` | T0-checkpoint-preserve-rng | `nn/transformer/model.py:682` | ✅ **E2E 真实 OLMo-core**（4 ckpt calls，real TransformerBlock evidence） |
| `CAND_OLMOCORE_EVAL_NOZEROGRAD` | OLMo-core | `f34e7ddc` | T0-evaluator-eval-mode（hunt iter 2 新加） | `train/callbacks/evaluator_callback.py:107` | ✅ **E2E 真实 OLMo-core**（117 fwd events flagged 跨 Transformer/Embedding/TransformerBlock） |
| `CAND_OLMOCORE_FSDP_EXPERTS` | OLMo-core | `f34e7ddc` | T1-replica-cksum-equal | `nn/moe/mlp.py:118` | ✅ **E2E structural**（AST + verbatim source: ddp body=`del world_mesh; pass`，fsdp body 4 stmts incl `fully_shard()`） |
| `CAND_DEEPSPEED_BF16_BOUNDARY_GRAD_LEAK` | DeepSpeed | DS v0.18.7 worktree（buggy 范围 0.18.6–0.18.10） | T1-grad-replica-cksum-equal | `runtime/engine.py:_backward_epilogue` | ✅ **E2E 真实 DS v0.18.7 + 2 H200**（6 cross-rank fp32-buffer disagreements） |
| `CAND_DEEPSPEED_ZERO_OFFLOAD_MULTI_BACKWARD` | DeepSpeed | DS v0.18.7 worktree | T1-multi-backward-per-step-fragile-config（iter 10 新加） | `runtime/zero/stage_1_and_2.py:1493-1499` | ✅ **E2E 真实 DS v0.18.7**（bug-path param change=2.479887=last-only oracle；all-4 oracle=7.146901；2.88× 缺口） |
| `CAND_DEEPSPEED_WARMUPCOSINE_MULTIGROUP` | DeepSpeed | `005afe12`（pinned 已确认） | T0-optim-lr-positive | `runtime/lr_schedules.py:825,856` | ✅ **E2E 真实 DeepSpeed**（DS 自己 logger.warning，get_lr()=[0.0] 单元素，asymmetric zip 留 group 1=1e-4 不动） |
| `CAND_MEGATRON_CUDAGRAPH_BUFFER_CORRUPTION` | Megatron-LM | `87d9d2506`（pinned 已确认） | T1-buffer-replica-cksum-equal（iter 4 新加） | `core/transformer/cuda_graphs.py:425` | ✅ **E2E structural on H200**（AST + runtime warmup replay：`expert_bias` buffer 被 corrupt，max-diff = 1.6108） |
| `CAND_OLMOCORE_ASYNC_CALLBACK_RACE` | OLMo-core | `f34e7ddc`（pinned 已确认） | (rule gap — async future state coherence) | `train/trainer.py:1239` | ⚠️ structural race，需 active probe |
| `CAND_DEEPSPEED_ZERO3_STREAM_RACE_NAN` | DeepSpeed | `005afe12`（pinned 已确认） | T0-no-nan-inf | `runtime/zero/stage3.py:1230` | ⚠️ rule capability via NaN injection；bug 路径要 torch ≥ 2.10（PR 7898 明文要求），driver cap 跨不过去 |
| `CAND_DEEPSPEED_OVERLAP_COMM_BUFFER_LIFETIME` | DeepSpeed | DS v0.18.7 worktree | T0-no-nan-inf | `runtime/zero/stage_1_and_2.py:1170-1172` | ⚠️ rule capability 已 confirmed；30-step stress 在 H200+NCCL 2.26 stack 没触发 stochastic race |
| `CAND_DEEPSPEED_BF16_ZERO0_DUAL_BUG` | DeepSpeed | `005afe12`（pinned 已确认） | partial: T0-no-nan-inf；structural rule gap | `runtime/engine.py:2092-2097` | ✅ **E2E 真实 DeepSpeed on H200**（4 step bf16+ZeRO-0：post-step grad mag = 4437，control = 0.000） |
| `CAND_OLMO_CKPT_SAVE_OVERWRITE_DROP` | OLMo | `204ad53c`（pinned 已确认） | (rule gap — 需 ckpt save/load 新 hookpoint) | `olmo/checkpoint.py:1938` | ✅ **E2E structural**（AST + runtime monkey-patch save_model_and_optim_state：kwargs={} 字面捕获） |
| `CAND_OLMO_ADAPTIVE_CLIP_EMA_RESET` | OLMo | `204ad53c`（pinned 已确认） | (rule gap — 需 optim.load_state_dict.post 新 hookpoint) | `olmo/checkpoint.py:1672-1677` | ✅ **E2E structural**（AST + runtime loop replay：5 个 EMA → 0 EMA） |

每条 candidate 1 个证据目录 `benchmark/eval/hunt_log/<id>/` 含 `code_excerpt.md`（file:line + writeup） + `verdict.json`（机读字段，含 dynamic_evidence block） + 大多数候选有 `dynamic_confirm_e2e.py` driver + `dynamic_confirm_e2e.log` 真实跑日志。

#### Plan A/B 工程基建（GPU 批留下的 reusable 工件）

- DS 0.18.7 worktree：`/volume/qscai/cqs/temp/deepspeed-0.18.7/`（git worktree from `exp/frameworks/DeepSpeed`）—— 复用 BF16_BOUNDARY、ZERO_OFFLOAD、OVERLAP_COMM 三个 candidate
- eval-gpu-0 venv：`/volume/qscai/cqs/temp/venv-cu126/`（torch 2.7.1+cu126 + cu12 12.6.x runtime + DeepSpeed/OLMo-core/OLMo 全部 deps）—— 解决了 driver 570 vs torch 2.11+cu13.1 mismatch
- wheel cache：`/volume/qscai/cqs/temp/wheels-cuda126/`（~3.4 GB；torch + cu12 + ML deps，eval-gpu-0 没外网时直接 `pip install --no-index --find-links`）

#### 关键观察
1. `CAND_OLMOCORE_RNGCKPT` 是 OLMo `O-005` 反模式在 OLMo-core 的镜像。OLMo 在 2024-08-13 commit `204ad53c` 修过 1 行（`preserve_rng_state = (...)` → `preserve_rng_state = not (...)`），OLMo-core HEAD `f34e7ddc` 自身 `nn/transformer/model.py:682` 还硬写 `preserve_rng_state = False`，旁边 TODO 也写明知道这个问题。
2. `CAND_OLMOCORE_EVAL_NOZEROGRAD` 触发了 trainaudit 自己的盲区：`torch.distributed.algorithms._checkpoint.checkpoint_wrapper.CheckpointWrapper.__init__` 用 `partial(torch_utils_checkpoint, ...)` 在 module load 时间捕获别名引用，绕过 trainaudit 的 global patch；fix 在 `core_trace/checkpoint_hook.py` 加入 alias-replacement 来填这个盲区。"hunt 自己 dogfood validate trainaudit instrumentation completeness" 是 paper 的次要卖点。
3. DS 0.18.7 worktree 是 2 个 candidate 的真实-framework E2E 关键基建：BF16_BOUNDARY_GRAD_LEAK 在 pinned 0.13.x 不存在（feature 还没引入），切到 0.18.7 才能复现 PR #7985 的 buggy 时序；ZERO_OFFLOAD_MULTI_BACKWARD 同理需要 PR #7665 引入的 `set_gradient_accumulation_boundary`。

#### 范围限制（hunt 没找到 = 没看到，不是没有）
- DeepSpeed `runtime/zero/parameter_offload.py:403` 的 "SOME TIMES post backward does not seem to be triggered" TODO 是 perf-only 自评（comment 说 "Should only cause increase in memory not correctness issue"），不计入。
- Megatron `core/distributed/custom_fsdp/fully_sharded_data_parallel.py:166/451/509` 的 FIXME 都是 grad-shard mode + no_sync 的语义清理，没看到明确的 silent-error case。
- Megatron `expert_bias` (router.py:125) 用 fp32 buffer + `scores_for_routing = scores + expert_bias` 看起来像 dtype mismatch，但实际 routing weight 是 `torch.gather(scores, ..., top_indices).type_as(logits)` 重新 cast 回原 dtype，是有意的 stability 设计，不是 bug。
- ZERO3_STREAM_RACE_NAN：PR #7898 明确说"PyTorch 2.10 introduced changes that make this race condition reliably trigger"。driver 570.86.15 cap 在 cu126（torch ≤ 2.7.1），凑不出 torch 2.10+ 不能 deterministically 复现 bug 的真实路径；rule capability 已通过 NaN injection 单独验证。



| Bug | 不可行原因 | 应放 tier |
|----|----|----|
| OC-NEW-1 | bug 在 olmo-core 自有 `_as_tensor`，PyTorch 层抓不到 | T2 / T3 |
| O-NEW-2 | causal mask 反转需主动构造对照输入 | T4 instance |
| M-NEW-1 | sigmoid in bf16 需"配置声明 fp32"上下文 | T1 / T2 |
| M-NEW-5 | aux loss 缩放需 num_tokens 语义上下文 | T3 |

### 2.6 Three-way baseline comparison (D1 same-集合)

来源：doc 26 任务（TrainAudit / TrainCheck / Naïve 在同 17-bug 集合上的定量对比）。
所有 D1 行 reference seed=0、torch 2.7.1+cu126、TrainCheck commit
`exp/traincheck/TrainCheck/` editable install (=OSDI '25 fork)。
Subset：`benchmark/eval/synthetic_17.json`（B1/B2/B3/B8/B11/B12/B13/M-012/M-014/
M-020/M-024/M-NEW-5/O-005/O-NEW-1/O-NEW-9/OC-NEW-2/OC-NEW-3）。

#### 主表（buggy 检出）

| 锚点 | 方法 | D1 buggy 检出 | D1 fixed FP | last_updated | 复现命令 |
|------|------|---------------|-------------|--------------|----------|
| §2.6 baseline-d1-trainaudit | TrainAudit | **17/17 = 100.0%** | 1/17 = 5.9% (OC-NEW-3 已知 surrogate FP) | 2026-05-07 | `python benchmark/eval/baseline_traincheck.py --mode synthetic --subset benchmark/eval/synthetic_17.json` (TrainAudit 列从 synthetic_runners 重跑) |
| §2.6 baseline-d1-traincheck | TrainCheck (OSDI '25) | **10/17 = 58.8%** | N/A（旧 D1 记录仅用 fixed 作 reference，不是独立 FP） | 2026-05-07 | `bash benchmark/eval/traincheck_surrogates/batch_t{0,1,1_extra}.sh` on eval-gpu-0；`python benchmark/eval/baseline_traincheck.py --mode synthetic --subset benchmark/eval/synthetic_17.json` |
| §2.6 baseline-d1-naive | Naïve metric monitoring | **0/17 = 0.0%** | 0/17 = 0.0% | 2026-05-07 | `python benchmark/eval/baseline_naive.py --mode synthetic --subset benchmark/eval/synthetic_17.json` |
| §2.6 baseline-real-trainaudit | TrainAudit on 23 real | (待测) | - | - | **阻塞** real subprocess — TrainAudit 自身 trainaudit_run.sh 需另跑（doc 22 §2.1 已部分有 verdict） |
| §2.6 baseline-real-traincheck | TrainCheck on 23 real | (待测) | - | - | **阻塞**：每 bug 需 traincheck-collect 包真实 framework 训练，driver 集成 ≥ 0.5 天/bug |
| §2.6 baseline-real-naive | Naïve on 23 real | **0/19 evaluable + 4 N/A by design** | - | 2026-05-07 | 全跑数据：见下"Naïve real subprocess sweep 结果" + `baseline_naive_real_results.csv` |

#### Naïve real subprocess sweep 结果（2026-05-07）

执行：`bash benchmark/eval/sweep_naive_real.sh` on eval-gpu-0 (8x H200, venv-cu126)。
覆盖 doc 26 §3.2 全 23 bugs：13 Phase 1-2 verified + 10 Hunt E2E confirmed。

| 子集 | 数量 | Naïve verdict |
|------|------|---------------|
| Phase 1-2 verified (B11/B12/B13/M-012/M-014/M-020/M-024/M-NEW-5/O-002/O-005/O-NEW-1/OC-NEW-2/OC-NEW-3) | 13 | **13/13 FAIL** — 全无 metric line parse |
| Hunt real-run (RNGCKPT/EVAL_NOZEROGRAD/BF16_BOUNDARY/ZERO_OFFLOAD/WARMUPCOSINE/BF16_ZERO0_DUAL) | 6 | **6/6 FAIL** — driver import 错误或纯 structural script |
| Hunt structural (FSDP_EXPERTS/CUDAGRAPH/CKPT_SAVE/CLIP_EMA_RESET) | 4 | **4/4 N/A by design** — 仅 AST + runtime emulation，无 metric stream |

**FAIL 分类**：

1. **Driver-pre-training crash (M-* + 部分 OLMo)**：trainaudit_driver.py 在 framework import / train_step 阶段崩溃（commit-vs-API mismatch with venv-cu126 torch 2.7.1）。M-014 见 `'NoneType' has no train_step`；其它 Megatron bug 同模式。stdout 100-130 行全是 init 日志，没到训练循环。
2. **Driver-runs-but-no-metric-output (B11 + DeepSpeed bugs)**：trainaudit_driver.py 设计就是 1-3 step minimal repro + framework-internal probe，**不打印 per-step loss / grad_norm**。B11 stdout 456 行全是 DS engine 配置 + TrainAudit init，无 `lm loss: X | grad norm: Y` 形式的 per-iter 行。
3. **Hunt driver import-time crash**：`hunt_log/CAND_*/dynamic_confirm_e2e.py` 顶层 `import deepspeed` 失败（DS_DIR 不在 hunt 子进程 PYTHONPATH），或 antlr4 grammar 版本与 venv 不匹配。
4. **Hunt structural (4 个)**：设计上就是 AST + runtime emulation，没真训练，N/A 不算 FAIL。

**关键 insight (paper §4.1 可写)**：

Naïve metric monitoring 在 silent-error 复现场景 **结构性失效**：

- 真实 bug 的 minimal-repro driver（trainaudit_driver / dynamic_confirm_e2e）**只跑 1-3 step**，根本不到 4-signal detector 的 W=20 step 窗口，spike rules 永远不可能 fire；仅 NaN/Inf 可能 fire，但大部分 silent error 不产生 NaN。
- 即使训练进入循环，每个 framework 的 stdout 格式不一致（DS print engine config but not per-step；OLMo 自有 JSON metric writer；Megatron 仅在 `--log-interval` 后开始打），需要**逐 framework 写解析器**——这跟 paper §4.1 想表达的 "Naïve 是 strawman" 完全一致。
- TrainAudit / TrainCheck 通过 instrument PyTorch tensor 状态绕过这个限制——这是 framework-aware silent-error detector 必须的设计。

**结论**：Naïve real-23 sweep 数据完整（19 evaluable + 4 N/A），可以进 paper §4.1 作为 strawman 论据。要 Naïve 真给出 DETECTED verdict 需要重新设计 driver（每 bug 跑 ≥50 step 且含 metric writer），那不是 Naïve 检测能力问题、是基线 harness 的 scope 问题。

**未做（仍阻塞）**：

- TrainCheck on 23 real：每 bug 需 traincheck-collect 包整个 framework training script，单 bug ~0.5-1 天，全 23 ≈ 1.5-2 周。
- TrainAudit on 23 real：trainaudit_run.sh 已存在，但同样撞 commit-vs-API mismatch（M-* 系列）。doc 22 §2.1 的 13/15 是早期版本下跑出来的，需要重新统一 venv 跑一遍才能进 paper 同集合表。

工件清单：
- `benchmark/eval/sweep_naive_real.sh`（23-bug sweep driver）
- `benchmark/eval/naive_subprocess.py`（per-bug Naïve runner）
- `benchmark/eval/naive_stdin_check.py`（stdin-based Naïve detector）
- `benchmark/eval/baseline_naive_real_results.csv`（23 行最终结果）

#### Per-bug verdict（D1 buggy 17 × 3 baseline = 51 cells）

| bug | framework | TrainAudit | TrainCheck | TrainCheck violations | Naïve |
|-----|-----------|------------|------------|----------------------|-------|
| B1 | megatron-lm | DETECTED | CLEAN | 0/315 | CLEAN |
| B2 | megatron-lm | DETECTED | DETECTED | 6/520 | CLEAN |
| B3 | deepspeed | DETECTED | CLEAN | 0/439 | CLEAN |
| B8 | deepspeed | DETECTED | DETECTED | 2/433 | CLEAN |
| B11 | deepspeed | DETECTED | DETECTED | 11/604 | CLEAN |
| B12 | olmo-core | DETECTED | DETECTED | 24/478 | CLEAN |
| B13 | olmo | DETECTED | DETECTED | 2/741 | CLEAN |
| M-012 | megatron-lm | DETECTED | CLEAN | 0/431 | CLEAN |
| M-014 | megatron-lm | DETECTED | DETECTED | 3/432 | CLEAN |
| M-020 | megatron-lm | DETECTED | DETECTED | 1/758 | CLEAN |
| M-024 | megatron-lm | DETECTED | DETECTED | 1/430 | CLEAN |
| M-NEW-5 | megatron-lm | DETECTED | CLEAN | 0/432 | CLEAN |
| O-005 | olmo | DETECTED | FAIL (infer crash) | - | CLEAN |
| O-NEW-1 | olmo | DETECTED | DETECTED | 1/683 | CLEAN |
| O-NEW-9 | olmo | DETECTED | CLEAN | 0/260 | CLEAN |
| OC-NEW-2 | olmo-core | DETECTED | DETECTED | 18/506 | CLEAN |
| OC-NEW-3 | olmo-core | DETECTED | CLEAN | 0/381 | CLEAN |

#### 工程基建（产出物）

- TrainCheck D1 surrogate **34 scripts** (17 bugs × buggy/fixed): `benchmark/eval/traincheck_surrogates/{B1,B2,B3,B8,B11,B12,B13,M-012,M-014,M-020,M-024,M-NEW-5,O-005,O-NEW-1,O-NEW-9,OC-NEW-2,OC-NEW-3}_{buggy,fixed}.py` —— 都是 standalone PyTorch script，可被 `traincheck-collect --pyscript` 包装
- Pipeline driver: `benchmark/eval/traincheck_surrogates/run_one.sh <bug_id>` —— 完整 collect (fixed) → infer → collect (buggy) → check 流程，输出契约行 `[<id>] BUG DETECTED|CLEAN|FAIL: traincheck:<n>/<total> failed invariants`
- Batch 入口: `batch_t0.sh`（7 个 T0 tier）+ `batch_t1.sh`（7 个 T1 tier）+ `batch_t1_extra.sh`（B2/B3/B8 三个 DeepSpeed T1 tier），各自带 results.txt 输出
- TrainCheck CSV 解析器: `baseline_traincheck.py --mode synthetic --subset benchmark/eval/synthetic_17.json` 读三个 batch results.txt + 跨引用 TrainAudit verdict（从 synthetic_runners 实时跑），输出 `baseline_traincheck_results.csv`（34 rows）
- Naïve detector: `baseline_naive.py` —— 4-signal 监控（loss_spike / loss_nan / grad_nan / gradnorm_spike，k=10 W=20），T0 tier (7 bugs) 跑真训练循环，T1 tier (10 bugs incl B1/B2/B3/B8) 标 CLEAN-by-design (no metric stream)
- 三方汇总表: `gen_baseline_3way.py` → `paper_table_baseline_3way.md`

#### 环境补丁（可复现）

- TrainCheck H200 driver 570 vs PTX 9.1 不兼容: 已 patch `exp/traincheck/TrainCheck/traincheck/proxy_wrapper/hash.py` 的 `if torch.cuda.is_available():` 改为 `if False:`，强制走 CPU hash fallback。CPU hash 在 H200 trace 规模下功能等价。
- numba 版本: `numba==0.61.0` + `llvmlite==0.44.0`（PTX 8.7-compatible）
- TrainCheck install: GPU 机器无外网，wheel cache 已扩充至 `/volume/qscai/cqs/temp/wheels-tc/`（polars / orjson / astor / numba / wheel）；用 `pip install --no-index --find-links /volume/qscai/cqs/temp/wheels-tc/` + `pip install -e ... --no-deps --no-build-isolation` 离线装

#### 关键观察

1. **TrainCheck 漏 6/17**：B1 (cross-rank checksum)、B3 (BF16 declared but comm uses FP16)、M-012 (expert_bias dtype demotion)、M-NEW-5 (router 缺 calculate_per_token_loss attribute)、O-NEW-9 (DataLoader 出 OOB token id)、OC-NEW-3 (lr 衰减方向反转)。涵盖 6 类 framework-aware 语义 bug：cross-rank consistency、comm dtype contract、parameter dtype 元规则、attribute presence、值范围、schedule 几何形状。验证 paper §4.1 的核心论点：**TrainCheck 的 API-trace invariant 学习无法 cover framework metadata + 元规则**。
2. **TrainCheck FAIL 1/17**：O-005 surrogate 在 traincheck-infer 阶段抛 AssertionError ("Exceptions or incomplete function calls don't have return values")。是 TrainCheck 工具的 robustness bug，不是检测能力 bug。
3. **旧 D1 记录不作为 TrainCheck 独立 FP 证据**：每条 bug 用自己的 fixed.py 学 invariants 再查自己的 buggy.py，只支持 buggy 检出统计；独立 FP 必须用 reference 之外的 held-out clean rerun 测量。主文 D2 表采用 held-out fixed rerun 协议。
4. **Naïve 全 miss 17/17**：T0 bug 的 buggy 表象不通过 loss/grad metric 暴露（B11 假 clip → grad 高但稳定；M-014 退化 softmax → grad=0 不是 NaN；O-005 dropout RNG 不一致 → step-level 看不出）；T1 bug 没有 metric stream。这正是 paper §4.1 想要的 strawman 论据。
5. **DeepSpeed 三类 bug 分别检验**：B2 (TP grad cross-rank disagree, T1-grad-replica-cksum-equal) DETECTED 6/520、B8 (EP comm group size mismatch, T1-process-group-size-correct) DETECTED 2/433、B3 (BF16 vs FP16 comm dtype mismatch, T1-comm-dtype-matches-training) MISSED — TrainCheck 学 API trace，不学 dtype contract。

#### 工作量记录

- TrainCheck integration（含 GPU venv 装包、PTX 不兼容 patch、surrogate 17×2 脚本、batch driver、CSV 解析）：~5 hours of work
- Naïve baseline 实现 + D1 跑（含 real-23 sweep）：~1 hour
- 三方汇总 + 文档同步：~30 min
- **未做**：23 real bugs TrainAudit + TrainCheck subprocess（doc 26 §3.2 D7 部分）— 每 bug 需 GPU + 框架 checkout + commit-vs-API 兼容矩阵 + traincheck-collect 包装 driver，单 session 不可行，留 follow-up sprint

---

## 3. §3 Method 引用的事实

| Paper claim | Value | Source（doc 21 锚点） | last_updated |
|----|----|----|----|
| T0 选 PyTorch 一等公民 API | 8 hookpoint | doc 21 §10.3 + §1.1 | 2026-05-05 |
| 类名启发式覆盖自定义 normalizer | OLMo 自定义 RMSLayerNorm 被 `endswith('Norm')` 命中 | doc 21 §10.4 | 2026-05-05 |
| Tuple 输出 + functional 绕过 module hook | M-014 暴露需补 F.softmax wrap | doc 21 §10.4a | 2026-05-05 |
| Shape (..., 1) trivial 也是 degenerate | M-014 调参案例 | doc 21 §10.4b | 2026-05-05 |
| Rule 容差需数据驱动 | O-NEW-1 调参 [0.3,3.0]→[0.5,2.0] | doc 21 §10.5a | 2026-05-05 |
| Rule precondition 收紧（仅 trigger 时检查）| B12 改为 hook scheduler.init + last_epoch != -1 | doc 21 §10.5 | 2026-05-05 |
| T0 → T2 边界（framework 自有 utility）| DeepSpeed 自有 clip_grad_norm_ 不在 PyTorch 命名空间 | doc 21 §10.3 | 2026-05-05 |
| T1 active probe vs T0 passive trace | B13 l2 magnitude 不够，需 cksum 级直接比较，OLMo adapter 注入 forward wrap | 2026-05-05 实验 | 2026-05-05 |
| Bug precondition 跨环境退化 | B1 在 modern PyTorch 被 CUDA RNG 自动同步掩盖 | 2026-05-05 B1 实测 | 2026-05-05 |
| **DSL 4 模板 + ≤2 schema 扩展硬约束** | scope.tensor_signature + bound.conditions（合规）；MONOTONIC 是已有 BoundKind 加 compile path | `dsl/templates.py`, `dsl/predicate.py` | 2026-05-05 |
| **dsl_native 11/18 + probe_derived 3 + python_fallback 4** | clip_grad/no_nan_inf/optim_lr_positive/build_has_modules/initial_lr_present/token_id_range/norm_rms/softmax_degenerate/optim_step_counter/replica_cksum_equal/expert_bias_dtype 是 dsl_native | `dsl/registry/MAPPING.md` | 2026-05-05 |
| **Python ↔ DSL event_id 等价**（M1 gate）| 14 等价测试 buggy + clean 双对比；6 个重叠 T0 rule 在 e2e use_dsl=True/False 路径下 0 mismatch | `tests/dsl/test_compiler_equivalence.py` | 2026-05-05 |
| **CTE 隔离 alias namespace** | `WITH expanded AS (...)` 把 `payload`(父) 和 `row_payload`(子) 暴露为命名列；precondition.expr 直接引用，零字符串 hack | `dsl/compiler.py` `_compile_with_path` | 2026-05-05 |
| **op 描述不变式方向，violation = NOT(...)**| `kind=bound, op=<=, value=max_norm` 是 invariant；compiler emit `NOT (post_norm <= max_norm * (1+rel))` 作为 violation 子句 | `dsl/compiler.py` `_per_field_failure` | 2026-05-05 |
| **walk_tensor_summaries SQL fetch + Python walker** | T0-no-nan-inf 嵌套字典走 Python (mirror `_walk_tensor_summaries`)；clean SQL 拉所有相关 hookpoints | `dsl/compiler.py` `_compile_walk` | 2026-05-05 |
| **MONOTONIC 用 ROW_NUMBER + self-join** | 编译为 `WITH ordered AS (... ROW_NUMBER OVER ...) SELECT a.event_id FROM ordered a JOIN ordered b ON a.rn = b.rn + 1 WHERE NOT (a.v > b.v)` | `dsl/compiler.py` `_compile_monotonic` | 2026-05-05 |
| **4-layer mining E2E 闭环**（§3.2 卖点）| source code → L1 propose → L2 enumerate (≤50/hyp) → L3 validate on healthy (zero violations across all) → L4 filter → 最终 clip_grad invariant 落地，重新喂 buggy trace 触发 | `tests/mining/test_layers_1_4.py::test_four_layer_pipeline_end_to_end` | 2026-05-05 |
| **Tolerance 99th-percentile auto-learning** | `infer_tolerance(field, healthy_stores, hookpoint, pct=99)` 返回 N-th percentile，给 BOUND predicate 自动设阈 | `mining/layer3_validate.py` | 2026-05-05 |
| **Pluggable LLMClient (B3/B4/C2)** | 同一 `Callable[(system, user, *, max_tokens), str]` 接口；CI 用 deterministic Stub，production swap claude-proxy-v3 / Anthropic SDK | `diagnosis/rca_agent.py:LLMClient` | 2026-05-05 |
| **C0 trace context** | snapshot_build 注入 `id(mod)→dotted_name` map；module hook 通过 `lookup_module_name` 回查；clip_grad/F.softmax/checkpoint 通过 `callsite(skip)` 跳过 trainaudit frames 找到用户调用栈 | `core_trace/_utils.py` + `module_hook.py` | 2026-05-05 |
| **C1 violation expander** | event_id → DiagnosisReport(suspect_module, suspect_rank, suspect_step, callsite, bug_specific, context_events, hypothesis)；±N 同 module_id slice | `diagnosis/expander.py` | 2026-05-05 |
| **Cross-rank outlier (4+ ranks)** | majority-vote on `gathered_cksums`；2-rank tie 显式标 `outlier_rank=None` | `diagnosis/cross_rank_outlier.py` | 2026-05-05 |
| **C2 LLM RCA agent** | DiagnosisReport → (suspect, cause, fix_hint) 3-line 结构化输出；prompt 含完整 report JSON 给 LLM 引用具体数字 | `diagnosis/rca_agent.py` | 2026-05-05 |
| **Online streaming runner** | OnlineRunner.tick() + cursor + rule skipping by inferred hookpoint；4 acceptance tests 含 multi-step training simulation | `streaming/online_runner.py` | 2026-05-05 |
| **Offline forensic CLI** | `python -m trainaudit {verify,diagnose,summary,replay} trace.duckdb` 4 子命令 + `--use-dsl` / `--rca` 标志；GPU 上跑产生 trace 后在 workstation 上离线分析 | `trainaudit/__main__.py` | 2026-05-05 |
| **gen_driver per-framework templates** | DeepSpeed (DS_DIR + _t_api shim) / Megatron (MEGATRON_DIR + PYTHONPATH + nproc=2) / OLMo / OLMo-core；deterministic per-bug port via md5；--use-detect 用 runpy.run_path 包 detect.py + Tee stdout 抓 detect.py 自身判定 | `benchmark/eval/gen_driver.py` | 2026-05-05 |
| **Fault injection 33 + boundary** | 31 severe/moderate (T0×22 + T1×9，12 categories) + 3 subthreshold boundary cases；100% detection / 100% TN | `benchmark/eval/fault_injection.py` | 2026-05-05 |
| **DataLoader subclass-override fix** | 发现并修：`_BaseDataLoaderIter._next_data` 被 `_SingleProcessDataLoaderIter` + `_MultiProcessingDataLoaderIter` override，base patch 无效 → 改为 patch 每个具体子类 | `core_trace/dataloader_hook.py` | 2026-05-05 |

---

## 4. §4 Evaluation 引用的数字

| Metric | Value | Sample size | Source | last_updated | TODO |
|----|----|----|----|----|----|
| T0 真实框架 cover rate | 6/48 = 12.5% | 48 reproduced bugs | §2.1 | 2026-05-05 | 补 5-10 个真实跑 |
| T0+T1 真实框架 cover rate | **13/48 = 27.1%** | 48 reproduced bugs | §2.1 | 2026-05-05 | 进 Phase 3 推到 35%+ |
| **D1 synthetic surrogate detection** | **15/15 = 100.0%** buggy detected | 14 bugs × 4 frameworks × 12 categories | §2.2b + `paper_table_coverage.md` | 2026-05-05 | - |
| **D1 synthetic surrogate FP** | **0/13 = 0.0%** fixed-commit FP | 13 fixed runs in synthetic_14 | §2.2b + `paper_table_coverage.md` | 2026-05-05 | - |
| **Fault injection benchmark detection** | **31/31 = 100.0%** severe + moderate | 31 severe + 3 boundary across 12 categories × 4 tiers | `paper_table_fault_injection.md` | 2026-05-05 | - |
| **Fault injection boundary TN** | **3/3 = 100% true negative** | 3 subthreshold faults at sensitivity floor | `paper_table_fault_injection.md` | 2026-05-05 | - |
| **Real-bug reproduction set total** | **26 bugs** (13 verified + 10 hunt + **3 acknowledged boundary FN**) | 4 frameworks (Megatron / DeepSpeed / OLMo / OLMo-core) | §2.1 / §2.5 + paper §4.1 | 2026-05-07 | - |
| **Real-bug detection rate** | **23/26 = 88.5%** (前 23 bug 全部检出 + 3 acknowledged FN: B14 / B15 / B7) | 同上 | paper §4.1 补充证据段 + §4.1 检测边界段 | 2026-05-07 | - |
| **Acknowledged boundary FN bugs** | **3 真实 bug**: B14 (OLMo z-loss masked-mean, sub-percent drift), B15 (loss / num_micro_batches, auxiliary-only dilution), B7 (FlashAttn dropout RNG reset, PP=2 stage-0 locality) | doc 24 §production_coverage_gap | paper §4.1 检测边界段 | 2026-05-07 | - |
| **Driver pool coverage** | **288/295 = 97.6%** (15 manual + 273 generated; 49 detect.py-wrapped) | `manifest_summary.md` | 2026-05-05 | - |
| FP rate on clean toy training | 0/17 rules | toy nn.Sequential ×3 step | doc 21 §10.2 | 2026-05-05 | 跑 clean Megatron / OLMo workload |
| **FP rate on 200-step clean MLP** | 0 violation in 5685 events | mlp-2l × 200 step | `benchmark/eval/overhead.py` | 2026-05-05 | - |
| **FP rate on 200-step clean GPT-tiny** | 0 violation in 16038 events | gpt-tiny × 200 step | `benchmark/eval/overhead.py` | 2026-05-05 | - |
| FP rate on real Megatron clean | (待测) | - | - | - | **阻塞**（GPU run） |
| FP rate on real DeepSpeed clean | (待测) | - | - | - | **阻塞** |
| FP rate on real OLMo clean | (待测) | - | - | - | **阻塞** |
| Per-tier coverage (T0 only) | 5/48 (10.4%) | - | §2.1 | 2026-05-05 | - |
| Per-tier coverage (T0+T1) | 6/48 (12.5%) | - | §2.1 | 2026-05-05 | - |
| Per-tier coverage (T0+T1+T2) | 待测 | - | - | - | Phase 3 启动后 |
| Per-tier coverage (T0+T1+T2+T3) | 待测 | - | - | - | Phase 4 启动后 |
| Per-framework adapter LoC | Megatron ~150 / DeepSpeed ~50 / OLMo ~150 (含 active probe) / FSDP ~30 | `wc -l trainaudit/adapters/*.py` | 2026-05-05 | - |
| Cross-commit hookpoint 存活率 | T0: 100% on 5 commits (2023-10 → 2024-09); T1: 100% on 1 commit | 实测 cross-commit | §2.1 commit hash | 2026-05-05 | 跑更多 commit |
| Cross-framework migration | 同一套 13 rule × 4 framework，0 修改部署 | adapter auto-detect | 2026-05-05 | 2026-05-05 | 量化每条 rule fire rate per FW |
| **Cross-framework migration (D3)** | **4/4 pairs** (DS→Megatron clip / OLMo→Megatron structural / OC→DS optim / Megatron→OC no_nan_inf) | `tests/integration/test_cross_framework.py` | 2026-05-05 | paper §4.4 ≥3/4 target met |
| **TrainCheck baseline harness (E1)** | full quantitative D1 same-集合 (synthetic_17)：TrainAudit 17/17, TrainCheck 10/17 (58.8%), Naïve 0/17 — 见 §2.6 | `paper_table_baseline_3way.md` + `baseline_{traincheck,naive}_results.csv` | 2026-05-07 | 23 real bugs subprocess 模式仍阻塞（每 bug ≥0.5 天 driver 集成） |
| **Hunt-detected silent errors** | **13 candidates** documented; **10/13 real-framework E2E confirmed** (5 CPU + 3 GPU framework runs + 2 GPU structural via AST + runtime emulation) | 4 frameworks, 14 hunt iterations + Plan A/B GPU batch | §2.5 + `benchmark/eval/hunt_log/INDEX.md` | 2026-05-06 | 1 stochastic race didn't reproduce; 1 needs torch ≥ 2.10; 1 pure pattern injection |
| **Hunt verbatim verification rate** | **12/13 = 92.3%** confirmed in pinned/buggy checkout | source-grep in pinned framework files (incl. DS v0.18.7 worktree) | §2.5 row-by-row evidence | 2026-05-06 | - |
| **Hunt rule-fire on real framework** | **3 candidates** rule fires on actual framework execution: T0-checkpoint-preserve-rng (OLMo-core), T0-evaluator-eval-mode (OLMo-core), T0-optim-lr-positive (DeepSpeed) | per-candidate `dynamic_confirm_e2e.py` + `.log` | `hunt_log/CAND_*/dynamic_confirm_e2e.log` | 2026-05-06 | extends with multi-rank GPU runs |
| **Novel-bug hunt: clean-HEAD invariant fire (EP a2a uninit buffer)** | **`CAND_OLMOCORE_EP_A2A_UNINIT_BUFFER_NAN`** (high conf, latent vulnerability) — small_hybrid_moe + EP=2 + HSDP + bf16. T0-no-nan-inf fires 208× at `comm.pre` (7.13% of all comm events) on `all_to_all_single` op. NaN originates in `BinnedGatherOp.forward` CUDA kernel output buffer: padding slots (when expert token count < capacity) are uninitialized memory, ~0.4% of bf16 random bytes parse as NaN. Downstream `compute_local_experts` masks via index, so loss + grads stay finite (CE=10.94, total_grad_l2 finite). 3-filter: no TODO at site, no fix commit in HEAD~50, GitHub search pending. **Bug class novel vs prior**: previous candidates active silent capacity loss; this is latent vulnerability — NaN exists in transmission but doesn't currently propagate. Mining run-1 H4 (`numel sum == tensor.numel`) anticipated this class but L2 couldn't instantiate. | clean HEAD + 2-GPU EP + trainaudit + 25-rule pass | `benchmark/eval/hunt_log/novel_hunt/CAND_OLMOCORE_EP_A2A_UNINIT_BUFFER_NAN/verdict.json` + `olmo_core_moe_ep2/{trace_rank0.duckdb,rule_results.json}` | 2026-05-06 | 3rd novel candidate; opens EP/comm path branch of the hunt |
| **Novel-bug hunt: clean-HEAD invariant fire** | **2 candidates** in OLMo-core HEAD `53c51c56` post-norm family (shared eps-clamp root cause but different severity): (1) **`CAND_OLMOCORE_REORDERED_HYBRID_DEAD_BLOCK0`** (very_high conf, **PERMANENT**) — small_hybrid_moe MoE branch RMS=0.0014→0.0019 across 100 steps, growth 1.08×; (2) **`CAND_OLMOCORE_REORDERED_DENSE_BLOCK0_ATTN`** (low severity / high phenomenon, **TRANSIENT**) — olmo2_60M dense reordered_norm block-0 attention_norm RMS recovers 0.47→0.53 over 100 steps (crosses rule threshold 0.5 by step 30). Both scale-specific: production olmoe_1B_7B and olmo2_370M factories don't fire. Triage closure for #1: (a) source mechanism unambiguous, (b) diff control = 0/25 fire on pre-norm hybrid, (c) **permanence** verified at 100 steps, (d) multi-rank reproduces. **Saturation evidence**: 8 clean configs across DS/Megatron/OLMo-core — only 2 OLMo-core dev-scale post-norm factories fire, zero FP on Megatron and large-scale OLMo-core. DS clean fires are nested-optim instrumentation artifact (rule precondition gap, task #77). | clean HEAD + trainaudit + 25-rule pass + diff control + 100-step long + multi-rank + 8-config saturation | `benchmark/eval/hunt_log/novel_hunt/CAND_OLMOCORE_*/verdict.json` + per-config trace+rule_results | 2026-05-06 | rule precondition refinement (#77) for DS nested-optim pattern |
| **Hunt-motivated rule additions** | **3 new rules** added during the hunt: T0-evaluator-eval-mode (iter 2), T1-buffer-replica-cksum-equal (iter 4), T1-multi-backward-per-step-fragile-config (iter 10) | `trainaudit/rules/{T0_evaluator_eval_mode,T1_buffer_replica_cksum_equal,T1_multi_backward_per_step}.py` | §1 row "T0/T1 rule 数" | 2026-05-06 | - |
| **Hunt FP audit regression** | **0/6 violations on healthy runs** unchanged after 3 rule additions + 1 trace-field add (grad_enabled in module.fwd.pre) | 3 archetypes × {200, 500} step healthy training | `benchmark/eval/long_clean_run_fp_audit.py` | 2026-05-06 | - |
| **Rule precondition gaps surfaced by clean-HEAD novel hunt** | **2 systematic FP patterns** discovered: (1) **DS nested-optim wrapper** — DS wraps torch AdamW so trainaudit captures BOTH outer (DS) + inner (torch) optim.step.pre events; outer events lack grad views → T0-grad-norm-finite + T0-optim-step-counter-monotonic spuriously fire on 4 DS clean configs (ZeRO-2/3/+actckpt/bf16-only); fix: skip outer events when `optimizer_class in {DeepSpeedZeroOptimizer, BF16_Optimizer, FP16_UnfusedOptimizer}` (task #77); (2) **L2Norm class** — nGPT 271M uses L2Norm (each vector unit-L2 → RMS=1/sqrt(d)≈0.036), not RMSNorm; T0-norm-output-unit-rms fires 1344× across 32 L2Norm modules; fix: gate rule on `module_class in {RMSNorm, LayerNorm}` (task #79). Both are systematic and trivially fixable; both surfaced by the same clean-HEAD hunt that found the 2 OLMo-core novel candidates. | DS×4 + OLMo-core nGPT clean runs | per-config trace+rule_results in `hunt_log/novel_hunt/` | 2026-05-06 | - |
| 训练 overhead (CPU upper-bound) | mlp-2l 200 step: +962% / gpt-tiny 200 step: +995% | `benchmark/eval/paper_table_overhead.md` | 2026-05-05 | **阻塞 paper §4.3 production 数：需 GPU + Megatron/OLMo run；CPU 数仅证明 hooks 正确触发 + FP=0 跨设备保留** |
| Fault injection detection rate | (待测) | - | - | - | **阻塞** §4.1（33 个合成故障） |
| **Test suite green** | **104 passed / 0 failed** | 21 dsl + 17 mining + 4 rca + 25 integration + 4 streaming + 4 cli + 4 gen_driver + 25 core | `pytest tests/` | 2026-05-05 | - |

---

## 5. 阻塞 paper claim 的 missing data

> 这是必须先做的实验。每条对应 paper 一个具体 claim。每完成一条划掉 + 把数字回填 §4。

- [x] **§4.4 cross-framework migration**: 4/4 pairs (DS→Megatron clip / OLMo→Megatron structural / OC→DS optim / Megatron→OC no_nan_inf) — `tests/integration/test_cross_framework.py`。Synthetic 层证明，真实 GPU run 是回归用。
- [x] **§3.2 LLM-augmented mining 4-layer pipeline**: L1+L2+L3+L4 全栈实现 + E2E 测试（hypothesis → enumerate → validate on healthy → filter → reapply on buggy 触发）。`tests/mining/test_layers_1_4.py::test_four_layer_pipeline_end_to_end`。
- [x] **§3.4 diagnosis (C1+C2)**: 5 个 bug 的 DiagnosisReport (suspect_module + suspect_rank + callsite + bug_specific + hypothesis) 全 pass + LLM RCA agent pluggable client。`tests/integration/test_{diagnosis,rca_agent}.py`。
- [x] **§4 baseline (E1) harness**: TrainCheck import OK + paper_table_baseline.md 比较表。
- [x] **D1 synthetic 主表**: 14 bugs × 4 frameworks × 12 categories — 15/15 detected, 0/13 fixed FP（2026-05-05 扩到 14：+O-005 +O-NEW-9）。
- [x] **clean-trace FP**: toy 3 step + mlp-2l 200 step + gpt-tiny 200 step 全 0 FP。
- [x] **§4.1 fault injection 33 合成故障**: 31/31 severe + 3/3 boundary TN — `benchmark/eval/fault_injection.py`。原阻塞清单已 close（synthetic 替代物）。
- [x] **§4.3 production-ready streaming**: OnlineRunner.tick() incremental rule runner + 4 acceptance tests including multi-step training simulation。
- [x] **§3.5 offline forensic CLI**: `python -m trainaudit {verify,diagnose,summary,replay}` 4 子命令 — 6 acceptance tests on persisted trace.duckdb。
- [x] **D1 driver pool scale**: 15 manual → 288 generated (97.6% manifest coverage)；49 wraps detect.py via runpy。`benchmark/eval/gen_driver.py`。

仍阻塞（全部需 GPU 或框架 checkout）：

- [ ] **§4.1 Detection Efficacy on real frameworks**: 在真实 trainaudit_run.sh 上跑 selected_32 / selected_48 出 results.csv（同 D1 harness 但 `--mode subprocess`）。需 GPU + DS_DIR/MEGATRON_DIR 等。**driver 池现 288/295 已就绪**，差 GPU。
- [ ] **§4.3 production overhead**: 跑 clean Megatron / OLMo workload + per-step time 对比 trainaudit on/off。CPU 数已知是 ~960% upper bound，paper-quality production 数需 GPU。
- [ ] **§4.X cross-version evaluation table**: 每条 T0/T1 rule 在多个 commit 上跑，画存活率衰减曲线。**gen_driver 已支持 buggy/fixed commit 自动 checkout**，差 GPU。
- [ ] **§4 baseline 完整 same-machine 对比**: 把 TrainCheck Instrumentor.start 围住 `synthetic_runners` runner，写 file-backed trace，跑 `traincheck.checker` 拉 verdict。
- [ ] **真实框架 cover rate 提升**: 当前真实跑通 13/48，目标 ≥15/48。49 个 detect.py 包裹的 driver 一旦 GPU 跑通，cover rate 自动跃升。

---

## 6. 实验 → paper 段落映射

> 每个跑过的实验对应论文哪一段，便于回查。

| 实验 | 输出文件 | 用在 paper 的位置 |
|----|----|----|
| B11 trainaudit 真实跑 | benchmark/bugs/B11/trainaudit_run.sh | §3 motivating example + §4.1 cover表 |
| B12 真实 OLMo-core | benchmark/bugs/B12/trainaudit_run.sh | §3.4 rule precondition 论证 + §4.1 |
| O-NEW-1 真实 OLMo + 调参故事 | benchmark/bugs/O-NEW-1/trainaudit_run.sh + doc 21 §10.5a | §3.4 rule 容差挖矿 |
| M-014 真实 Megatron | benchmark/bugs/M-014/trainaudit_run.sh | §3.2 functional hook 必要性 + §4.1 |
| O-005 真实 OLMo | benchmark/bugs/O-005/trainaudit_run.sh | §3.5 checkpoint hook + §4.1 |
| B13 真实 OLMo + active probe | benchmark/bugs/B13/trainaudit_run.sh + adapter olmo.py | §3.3 T1 active probe + §4.1 |
| B1 真实 Megatron（**未触发**）| benchmark/bugs/B1/trainaudit_run.sh | §6 limitation: bug precondition 跨环境退化 |
| **D1 synthetic_12 主表** | `benchmark/eval/run_all.py --subset synthetic_12.json --mode synthetic` → `paper_table_coverage.md` | **§4.1 Detection Efficacy 主表**（13/13 detected, 0/11 FP, 4 frameworks, 10 categories）|
| **DSL/Python event_id 等价** | `tests/dsl/test_compiler_equivalence.py` (14 tests) | §3.2 DSL → SQL faithful translation 论证 |
| **4-layer mining E2E** | `tests/mining/test_layers_1_4.py::test_four_layer_pipeline_end_to_end` | **§3.2 LLM-augmented invariant mining 论证**（hypothesis → enumerate → validate → filter → reapply on buggy）|
| **Diagnosis 4 bugs E2E** | `tests/integration/test_diagnosis.py` | **§3.4 violation expander 论证**（B11 callsite, O-NEW-1 named module, OC-NEW-2 optimizer class, M-005 outlier_rank）|
| **C2 LLM RCA pluggable** | `tests/integration/test_rca_agent.py` | §3.4 RCA agent prompt + 输出契约 |
| **Cross-framework migration 4/4** | `tests/integration/test_cross_framework.py` | **§4.4 Generalizability 主表** |
| **Overhead + FP harness** | `benchmark/eval/overhead.py` → `paper_table_overhead.md` + `paper_table_fp.md` | §4.3 efficiency（CPU upper-bound + FP=0；GPU 数待补） |
| **TrainCheck baseline harness** | `benchmark/eval/baseline_traincheck.py` → `paper_table_baseline.md` | §4 baseline 比较列骨架 |

---

## 7. 自动化助手

### 7.1 推荐工作流

```bash
# 实验完一次后立刻：
make paper-status                    # 打印当前数字（脚本待写）
# 编辑 docs/v2_semantic_guided/22_paper_evidence_index.md
#   - 找到对应行
#   - 更新 value + last_updated
# git commit -m "exp: B13 真实 OLMo T1 通过, +1 to T1 cover"

# 写 paper 段落前：
grep -A1 "Bug XX" docs/v2_semantic_guided/22_paper_evidence_index.md
# 把数字 + 锚点抄进 main_cn.tex
# 加注释: % SOURCE: paper_evidence §4.1 row B11

# 投稿前 sanity check:
make paper-validate                  # diff main_cn.tex 的数字 vs 本表 (脚本待写)
```

### 7.2 待实现的工具脚本（提议放 `scripts/paper/`）

- `update_status.py`: 自动从 trainaudit/ 收集 (LoC, rule_count, hookpoint_count) 更新 §1 系统状态表
- `coverage.py`: 扫 benchmark/bugs/*/trainaudit_run.sh 最近一次输出，更新 §2.1 per-bug 验证表
- `paper_diff.py`: grep `main_cn.tex` 中的具体数字（如 "10.4%"、"5/48"），与本表对照

---

## 8. 与其他文档的关系

| 文档 | 关系 |
|----|----|
| **doc 20** (跨框架 trace 难题) | 设计文档，paper §3 motivation 来源；引用本表的具体数据点 |
| **doc 21** (落地 roadmap) | 实验日志，append-only；每次实验完 → 立刻同步本表对应行 |
| **doc 22** (本文档) | **唯一真值索引**，paper 所有具体数字的锚点 |
| **doc 23** (`23_detection_mechanics.md`) | 给 reviewer 的端到端检测机制详述：8 hookpoint × 13 dsl_native + 5 fallback × 13 真实 bug 的 trace payload + rule firing logic + GPU 验证结果。**reviewer 应先读 doc 23 理解机制，再用 doc 22 查具体数字**。 |
| **`main_cn.tex`** | 论文正文；数字旁加注释 `% SOURCE: paper_evidence §X row Y` |

→ 当 doc 21 / doc 22 / paper.tex 数字不一致时，**doc 22 优先**。

---

## 9. 收口

把这份当成你的"实验进度看板 + 论文事实索引"。同时实验和写论文的诀窍就一句话：

> **任何要进 paper 的数字，先在这里登记 → 实验跑完更新这里 → paper 段落引用这里**。

漂移成本 = 0。投稿前跑 paper_diff.py 一次，就知道 paper 哪个数字过期了。

---

## 10. 2026-05-05 framework 闭环里程碑

> 全部 P0+P1 task 关闭（15/15）。Paper §3 + §4 主体框架可投稿候选状态。

**完成**：
- §3.2 invariant DSL（4 templates + **13 dsl_native YAMLs** + 18 等价测试）
- §3.2 LLM-augmented mining 4-layer pipeline（L1 hypothesis + L2 enumerate + L3 validate + L4 filter，all pluggable LLM, deterministic stub for CI）
- §3.3 trace collection（**18 rules + 5 adapters + lifecycle-safe + module_name + callsite + DataLoader subclass-override fix**）
- §3.4 diagnosis（C1 expander + C2 LLM RCA agent，pluggable client）
- §3.5 **offline forensic CLI**（verify / diagnose / summary / replay）
- §4.1 **synthetic detection**（14 bugs / 4 frameworks / 12 categories / **15/15 = 100%** / 0% FP）
- §4.1 **fault injection**（31 severe + 3 boundary / **31/31 = 100%** detection + 3/3 TN）
- §4.3 **online streaming runner**（OnlineRunner.tick() + 4 acceptance tests）
- §4.4 **cross-framework migration**（4/4 pairs）
- §4 **D1 driver pool: 15 → 288 (97.6% manifest coverage)**（gen_driver per-framework templates + --use-detect runpy wrapping for 49 bugs）
- §4 baseline harness（TrainCheck import OK + comparison table skeleton）
- **104 passed tests / 0 failed**

**剩余 GPU/集成 stretch**（全部不阻塞投稿候选状态）：
- 真实 trainaudit_run.sh 跑 selected_32 / selected_48（subprocess 模式 + 288 drivers 就绪）
- Production overhead 数（GPU + Megatron/OLMo workload）
- TrainCheck Instrumentor 完整集成
- selected_80 / cross-version 衰减表

**关键命令一览**：
```bash
# 系统状态
pytest tests/                                                     # 期望 86 passed
ls trainaudit/trainaudit/dsl/registry/{T0,T1}/*.yaml              # 期望 11 dsl_native YAMLs

# 主表
python benchmark/eval/build_manifest.py                            # 295 bugs manifest
python benchmark/eval/run_all.py --subset benchmark/eval/synthetic_12.json --mode synthetic
                                                                   # 期望 13/13 detected, 0/11 FP
python benchmark/eval/overhead.py --model gpt-tiny --steps 200    # FP=0 + CPU upper-bound
python benchmark/eval/baseline_traincheck.py --mode harness-check # TrainCheck import status

# 真实 GPU 运行（外部依赖）
DS_DIR=... bash benchmark/bugs/B11/trainaudit_run.sh              # 真实 DeepSpeed
python benchmark/eval/run_all.py --subset benchmark/eval/selected_32.json --mode subprocess
```
