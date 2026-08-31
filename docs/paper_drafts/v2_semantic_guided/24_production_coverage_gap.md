# 真实训练部署 + 高覆盖率：还差什么

> 给定当前 framework（doc 23 描述的 18 rule + 13 dsl_native + 4-layer mining + GPU 13/14 验证），如果要**实际部署到一次真实 production training run** 并且**显著提高 silent error 检测覆盖率**，还需要做什么。
>
> 本文是 **honest gap 分析 + 优先级 action plan**，不是宣传 — 每条限制都明确标"还没解决/不充分"。
>
> last_updated: 2026-05-05

---

## 1. 一句话现状评估

**今天**：framework 在 4 个真实框架的 14 个真实 bug commit 上 GPU 端到端验证通过（13/14 detect, 0/12 FP），但**真实部署还差 3 类工作 — coverage / production-readiness / multi-node validation**。

**真实覆盖率口径**：

| 口径 | 数值 | 说明 |
|---|---|---|
| 已 reproduced bug × 已检测 | **13/48 = 27.1%** | doc 22 §2.1 真实 GPU 跑通 |
| 已 reproduced bug × manual driver 已写 | **15/48 = 31.3%** | 多 2 个驱动写完但 GPU 上 B1 mask + O-NEW-8 mup 阻塞 |
| 全 manifest（含未 reproduced）× 已检测 | **13/295 = 4.4%** | 大部分 bug 在 manifest 中只有 config，没复现，不知道 trainaudit 跑得过去吗 |
| 14 categories（doc 23 §6.4）× 已检测 | **12/14** | gradient_clipping / lr_schedule / residual / numerical / dtype / normalization / control_flow / optimizer / loss_scaling / moe_router_init / configuration_validation / data_loading 已覆盖；**numerical_precision_drift（B14/B15）和 backward_communication（B2）等仍 0 覆盖** |

**结论**：要把 coverage 推到能撑 paper 主表 + 生产部署：
- 短期（1-2 周）能做到 **20+/48 reproduced detection**
- 中期（1-2 月）能做到 **30+/48 + 生产实测 0 FP / <5% overhead**
- 长期（持续）才到 **80%+ 全 manifest** —— 需要源码挖矿 + framework 厂商合作

---

## 2. Coverage 还差什么：按 bug 类别拆开

### 2.1 当前 14 个 category 的覆盖盘点

✅ **已覆盖**（GPU 验证 + 真实 bug commit）：

| Category | Rule | Bug 实例 |
|---|---|---|
| gradient_clipping | T0-clip-grad-bounded | B11 |
| lr_schedule | T0-initial-lr-present, T1-sqrt-decay-front-loaded | B12, OC-NEW-3 |
| residual_connection | T1-residual-stream-preserved | B13, O-002 |
| numerical (degenerate softmax) | T0-softmax-degenerate | M-014 |
| dtype | T1-expert-bias-fp32, T1-jitter-preserves-dtype | M-012, M-024 |
| normalization | T0-norm-output-unit-rms | O-NEW-1 |
| control_flow (checkpoint rng) | T0-checkpoint-preserve-rng | O-005 |
| optimizer state | T0-optim-step-counter-monotonic | OC-NEW-2 |
| loss_scaling (router attr) | T1-router-has-calculate-per-token-loss | M-NEW-5 |
| configuration_validation | T1-layer-count-strict | M-020 |
| data_loading | T0-token-id-in-vocab | O-NEW-9（合成） |
| sanity (NaN/Inf) | T0-no-nan-inf | universal |

❌ **完全没覆盖（重要类别）**：

| Category | Bug 实例 | 为什么没覆盖 | 需要的工具 |
|---|---|---|---|
| **numerical_precision_drift** | B14 (z_loss masked-mean), B15 (loss / num_micro_batches) | 偏差 < 0.5%/step，落在所有 bound 容差内；需要 **reference implementation 比较** | T4 instance detector：构造 oracle 计算 + 用差值阈值 |
| **backward_communication** | B2 (frozen-weight all-reduce 缺失) | 没有 module.bwd 之后的 grad cross-rank check；现 cross_rank_cksums 只在 build.snapshot 跑一次 | 加 `bwd.cross_rank_grad_cksum` active probe，每个 step backward 后采样 |
| **pipeline parallelism specific** | B6 (interleaved microbatch counter), B7 (FlashAttn dropout reset) | PP 的 stage-local vs global 状态分裂，trainaudit 没建模 stage 概念 | 加 `pp_stage` 维度到事件 metadata + PP-aware rule |
| **MoE expert grad scaling** | B5 (缺 /num_gpus 归一化) | 没有 expert-aware grad rule | T2 rule：MoELayer 后 grad_norm 跨 expert 应均衡 |
| **Process group init** | B8 (DeepSpeed EP 用 num_experts 而非 ep_size) | 没扫 dist process_group 元数据 | 加 `dist.init_process_group` hook + group-shape rule |
| **Sequence parallelism** | B9 (uneven attn head 触发错误 all2all) | TP/SP shard 量化检查不在 | T2: SP-aware tensor shape rule |
| **EP communication group** | B8, M-NEW-* | 同上 | adapter `list_groups()` 的接口位已留，没实现 |
| **Sampler / data offset** | (M-026 类) | DataLoader.batch hook 只看 input_ids，没看 sampler state | 加 `sampler.state` hook |
| **Checkpoint serialization** | (D-NEW-* 类) | checkpoint.call 只看 preserve_rng；不读 saved tensor | 加 `torch.save/load` 包装 + 字节级 cksum 比较 |
| **Mixed precision autocast** | (跨 framework) | 没 autocast context 维度 | 加 `autocast.enter/exit` hook + dtype-stack rule |

**估算**：以上 10 个 category 每个加 1-2 条 rule + 配套 hook/probe → 能覆盖 **manifest 中另外 ~25-30 个 reproduced bug**。

### 2.2 Tier 维度：T2/T3/T4 还没真实落地

trainaudit 设计了 5 tier（doc 23 §1），但目前 rule 只在 T0/T1：

| Tier | 含义 | 已落地 | 缺什么 |
|---|---|---|---|
| **T0_PYTORCH** | PyTorch 一等公民 API | ✅ 11 rule | — |
| **T1_FW_METADATA** | adapter 注入语义 | ✅ 7 rule | — |
| **T2_FW_PRIMITIVE** | 框架特定 primitive | ❌ 0 rule | 需要包装 framework-specific function（如 DeepSpeed `clip_fp32_gradients`、Megatron `_p2p_communicate`） |
| **T3_FW_SPECIFIC** | 特定方法/算法 | ❌ 0 rule | 例如 ZeRO-3 sharded grad reduce 数学不变量 |
| **T4_INSTANCE** | 单一 bug instance detector | ❌ 0 rule | 例如 B14 z_loss 必须等于 reference impl + 容差 |

**T2 是关键缺口**：B2 / B5 / B6 / B8 / B9 这些跨框架 PP/EP/SP 的 silent error 都需要 T2 来抓 framework 自身的 primitive function 行为。

### 2.3 Framework 维度

**已有 5 adapter，但 4 个有真实功能**：
- Megatron ✅（label_param + label_module + build_invariants + jitter probe）
- DeepSpeed ✅（label_param ds_id 检测）
- OLMo ✅（residual probe）
- OLMo-core ✅（decay probe）
- FSDP ⚠️ 仅 label_param，没 active 功能

**完全没覆盖的高价值 framework**：

| Framework | 真实使用率 | 已知 silent error 类 | 工作量 |
|---|---|---|---|
| **HuggingFace Trainer + accelerate** | 极高（教育/研究） | sampler state, mixed precision context | 1-2 周（adapter + 5-10 rule） |
| **PyTorch Lightning** | 高 | optimizer hooks 多个 | 1-2 周 |
| **ColossalAI** | 中 | TP/PP 混合的 group 初始化 | 2-3 周 |
| **Ray Train + DeepSpeed/Megatron 包装** | 中 | 多进程 shard 一致性 | 2-3 周 |
| **NeMo / Megatron-LM-NeMo** | 中（Nvidia 内部多） | 同 Megatron 但有额外 wrapper | 1 周 |
| **TRL (RLHF)** | 增长快 | reward/policy 分支独立采样 | 2 周 |
| **JAX (Pax / MaxText)** | 中（Google 系） | 完全不同 framework，需要新 trace 设计 | 4-6 周 |

---

## 3. Production-readiness gap

要部署到一次**真实多卡多节点 long-run training**（数千 GPU、N 周持续），目前 framework 还有以下硬限制：

### 3.1 性能与开销

| 问题 | 当前状态 | Production 要求 | 怎么补 |
|---|---|---|---|
| **每步 hook 开销** | CPU toy 测得 +962% 上限；真实 GPU 大模型未测 | <5% per step | (1) 异步 emit (queue + bg thread) (2) 采样 (sample_rate per hookpoint) (3) 大 tensor stat 用 GPU-side reduction（torch._foreach_norm 已有；当前 summarize_tensor 是 per-tensor 6 kernel） |
| **events 表无界增长** | 单进程跑 100k step 估计 ~10GB JSON | 跑 7×24 必须有滚动窗口 | OnlineRunner 已是 cursor-based；需在 store 层加 retention 策略（保留最近 N step 或 N 小时） |
| **DuckDB JSON 解析慢** | 每条 rule 跑 SQL 时反复解析 payload | 高频 rule 需缓存 | (1) 把热门字段 promote 到 column（grad_norm / cksum / has_nan） (2) DSL 编译器已支持，扩到所有 rule |
| **Cross-rank cksum 同步** | build.snapshot 一次性，跑训练完整步骤会 hang | 训练阶段不能阻塞 | 改成 ring-async：rank-0 周期性 gather 而不是每个 rank 都 all_gather |

### 3.2 稳定性与容错

| 问题 | 当前 | Production |
|---|---|---|
| **Hook 内部 raise 是否会拖崩训练** | 多数 hook 有 try/except 但不一致；module hook 直接抛 | 永不影响训练 — 必须全 wrap try/except，failure 计数告警 |
| **Trace store 写盘失败** | 抛 IOError | 降级为 in-memory ring + 告警 |
| **CUDA OOM 时 trainaudit 是否还工作** | 不确定（hook 也分配 tensor 算 stat） | 需要 OOM-safe path：`with_stats=False` 自动降级 |
| **Multi-node 多 rank 写同一 db_path** | 每个 rank 各写各的（不冲突） | 但**没有 rank-0 聚合**，rule 跑在 single rank 看不到全图 → 跨 rank rule 失效。需要 rank-0 aggregator daemon |
| **训练 hang/crash 后 trace 完整性** | DuckDB append-only，崩溃后可读 | OK，但**没有定期 flush**；最近 256 events 可能丢失（buffer_limit）|
| **进程 fork (DataLoader workers)** | DataLoader hook 已修，但 fork 后子进程可能 import trainaudit 重 enable | 需要 `os.register_at_fork` 显式 disable in workers |

### 3.3 集成

| 集成点 | 当前 | Production |
|---|---|---|
| **告警** | violation 只 print 到 stdout / log | 需要 alerting hook（Slack / PagerDuty / OnCall webhook）|
| **Dashboard** | `paper_table_*.md` 静态产出 | Grafana / Weights & Biases panel 实时显示 violation count + rule_id breakdown |
| **结构化日志** | trainaudit print，混在框架 log 中 | 接入 Python logging + structured JSON output |
| **CI/CD 集成** | 手跑 pytest | 跑训练前的 pre-flight smoke check（在 train 命令前跑 `trainaudit verify` 一遍 build.snapshot） |
| **Trace 长期归档** | 本地 .duckdb | S3/GCS 上传 + 按 run_id 索引 |

### 3.4 配置

当前没有正式的配置文件，启用是 hardcoded：
```python
trainaudit.enable(tier=Tier.T1_FW_METADATA, db_path="./trace.duckdb")
trainaudit.snapshot_build(model, optimizer)
```

Production 需要：
```yaml
# trainaudit.yaml
tier: T1_FW_METADATA
db_path: /shared/runs/${RUN_ID}/trace.duckdb
sample_rates:
  module.fwd.post: 0.1     # 高频 hook 抽样
  module.bwd: 0.05
  utils.clip_grad.post: 1.0
disabled_rules:
  - T0-token-id-in-vocab   # 例如 vocab 校验已在数据 pipeline 做了
async: true
async_queue_size: 1024
multi_rank:
  aggregator_rank: 0
alerting:
  violations_threshold: 1
  webhook: ${SLACK_WEBHOOK}
retention:
  keep_last_steps: 1000
  flush_interval_s: 30
```

`trainaudit/config.py` 不存在，需要写。

### 3.5 多节点验证

**当前测试规模**（doc 23 §6）：
- 单机最多 2 GPU（torchrun --nproc_per_node=2）
- 步数最多 200 step
- 模型最大 ~1.3B 参数

**Production 规模**（典型 7B+ 训练）：
- 8 节点 × 8 GPU = 64 ranks
- 100k+ step
- TP=4, PP=2, DP=8 层级混合
- 长跑 7-30 天

**完全未测试**的场景：
- 跨节点 NCCL collective + trainaudit dist_hook 是否正确捕捉
- 64-rank cross_rank_cksums all_gather 性能
- TP=4 + PP=2 下 framework_invariants 抓取完整性
- ZeRO-3 sharded param 在 hook 看到的是 full 还是 shard

**这是最大的未知量** — 在小规模工作不代表在大规模工作。

---

## 4. 优先级 Action Plan（投入 → 效果排序）

### P0 — 部署前必做（≤2 周，单人）

> 没有这些，不能在生产 run 上启用。

1. **Async hook executor**（~3 天）
   - 把 store.emit 改成往内存 queue push，bg 线程批量序列化写 DuckDB
   - 训练步内开销降到 us 量级
   - 预期：CPU toy 上从 +962% 降到 <50%，GPU production 推到 <5%

2. **Sampling 实现**（~1 天）
   - OnlineRunner 已留 sample_rates 接口；填实现：高频 hookpoint 按 hash(event_id)%K==0 抽样
   - 配合 #1 可达 paper §4.3 目标

3. **配置文件 + CLI flag**（~2 天）
   - `trainaudit/config.py` + `from_yaml(path)`
   - `trainaudit.enable(config=...)` 入口
   - `python -m trainaudit verify --config trainaudit.yaml trace.duckdb`

4. **Multi-rank aggregator**（~3 天）
   - rank-0 daemon 周期性从其他 rank 拉 trace.duckdb 增量段
   - 跨 rank rule 在聚合后 store 上跑
   - 关键：解决 cross_rank_cksums 在长跑中的同步开销

5. **Hook 失败容错统一**（~1 天）
   - 每个 hook 顶层 try/except，failure 计数器
   - 故障次数 > N 时自动 disable 该 hookpoint 而不影响训练
   - 加 `trainaudit.health()` 返回 hook OK/FAIL 计数

6. **真实大模型 baseline FP/overhead 验证**（~3 天 GPU）
   - 跑 1-2 个真实 framework workload（OLMo 1.3B / Megatron-7B 各 200 step）
   - 量化 overhead + 0 FP confirmation
   - 写进 doc 22 §4 row "训练 overhead"

**预期成果**：framework 可以挂在 production training 上跑 24h+ 不阻塞、不漂、不漏；overhead < 5%；可视化的 health metrics。

### P1 — 短期 coverage 提升（≤1 月）

7. **新增 5-8 条 T2/T3 rule 覆盖空白 category**（~1 周/条）
   - **T2-tp-grad-allreduce-presence**：抓 backward 后 TP-shard param 应该有 cross-rank grad sync（B2）
   - **T2-microbatch-counter-globally-consistent**：跨 PP rank microbatch_id 全局对齐（B6）
   - **T2-moe-expert-grad-balance**：同步 expert 的 grad_norm 应在容差内（B5）
   - **T2-zero-shard-allgather-pre-step**：ZeRO-3 step 前应有 all_gather 把 shard 拼齐（D-* 类）
   - **T3-loss-reference-comparison**：T4 reference oracle 比较，催 B14/B15
   - **T3-numerical-drift-window**：滑动窗口检测 loss 偏离 baseline > 0.5% 累积（B14）

8. **L1 LLM hypothesis 接真 LLM**（~3 天）
   - 现在用 stub；切成 claude-proxy-v3 或 Anthropic SDK
   - 在 Megatron-LM/megatron/core/transformer/moe/router.py 等 5 个核心文件上跑 mining
   - 期望：mine 出 5-10 个 hypothesis → L2/L3 验证 → 至少 2 条转为 dsl_native YAML

9. **gen_driver --use-detect 在 GPU 上跑通 selected_48**（~3 天 GPU）
   - 当前 49 个 detect.py-wrapped scaffold 没在 GPU 上跑过
   - 跑通后 cover rate 跃升到 30+/48

10. **HF Trainer + accelerate adapter**（~1 周）
    - 这是研究/教育最常用的 framework，加这个能让 trainaudit 立即对接 HF 生态

11. **Dashboard / alerting 集成**（~1 周）
    - W&B / TensorBoard log writer
    - Slack webhook on violation
    - `trainaudit.metrics` 导出 Prometheus 格式

**预期成果**：reproduced bug coverage 从 13/48 → **25-30/48**；HF Trainer / Lightning 等生态可以直接挂。

### P2 — 中期 coverage 推到 paper claim（≤3 月）

12. **批量 mining 真实 framework 源码**（~2 周）
    - 写 `mining/run_pipeline.py --src megatron/ --frameworks megatron,deepspeed`
    - 在每个 framework 全部源码上跑 4-layer pipeline
    - 期望：mine 出 30-50 条候选；L4 filter 后 10-20 留下；L3 healthy validate 后 5-10 通过 → 落到 dsl_native YAML

13. **Cross-version 衰减表**（~1 周 GPU）
    - 每条 rule × 5 个 commit (横跨 1 年) 跑通
    - 画出 hookpoint 存活率衰减曲线 — paper §4.X 的核心证据

14. **Fault injection 扩到 100+** （~3 天）
    - 自动化 bit-flip / NaN / param-corruption 注入 healthy trace
    - 批量验证 detection rate

15. **TrainCheck 完整 Instrumentor 集成**（~1 周）
    - 当前只 import OK；需把 Instrumentor.start 围 synthetic_runners + checker pipeline 串通
    - 拿到真同机比较数

16. **selected_80 / 80 bug stretch**（~2 周 GPU）
    - 把 manifest 中所有 reproduced + driverable 跑一遍
    - 真实 cover rate 推到 30-40+

**预期成果**：reproduced bug coverage 30-40/48；全 manifest cover 率 15-20%；paper §4 主表完整。

### P3 — Long-tail（持续）

17. **JAX/Pax 适配**（~1.5 月，需团队）
18. **生产用户接入** — 1-2 个外部团队使用 + 反馈 bug
19. **持续追踪 framework upstream** — 每季度同步新 commit 的 hook 兼容性
20. **新 silent error 类型挖掘** — paper 之外的真实 production incident 复盘加新 rule

---

## 5. 用 4-layer mining 自动扩 coverage（关键路径详述）

P2 的 #12 是最重要的 coverage scaling 路径，单独详述。

### 5.1 当前 mining pipeline 状态

```
trainaudit/mining/
├── layer1_hypothesis.py    ← LLM 接口已搭，stub 实现
├── layer2_enumerate.py     ← deterministic, 6 relation type, 已通过 5 unit test
├── layer3_validate.py      ← 多 healthy trace 验证 + tolerance auto-learn 已实现
└── layer4_filter.py        ← LLM 接口已搭，stub 实现
```

`tests/mining/test_layers_1_4.py::test_four_layer_pipeline_end_to_end` 验证：source → L1 → L2 → L3 → L4 → 重新喂 buggy → 触发，全链路通。

### 5.2 接真 LLM 的最小配置

```python
# 替换 layer1_hypothesis.propose_hypotheses() 的默认 stub:
import os, anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def my_llm(system: str, user: str, *, max_tokens: int = 1024) -> str:
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text

# 然后:
hypotheses = propose_hypotheses(open("megatron/router.py").read(),
                                 framework="Megatron-LM",
                                 llm_client=my_llm)
```

### 5.3 批量 mining 入口（需写）

```python
# trainaudit/mining/run_pipeline.py（新建）
def mine_framework(framework_dir: Path,
                    healthy_traces: List[TraceStore],
                    buggy_traces: Optional[List[TraceStore]] = None,
                    *,
                    llm_client: LLMClient,
                    file_filter: List[str] = None) -> List[Predicate]:
    """对一个 framework 源码目录做 4-layer mining，返回 verified invariants。

    1. 用 file_filter 选高价值文件（router.py, optimizer.py, attention.py 等）
    2. 对每个文件批量 propose_hypotheses() → 候选 Hypothesis 池
    3. 对池中每个 Hypothesis × healthy trace schema 做 enumerate
    4. 对每个候选 predicate × healthy traces 做 validate
    5. 对接受的 predicate × buggy traces 做 reapply check（可选 — 验证 sensitivity）
    6. L4 filter 拒绝 spurious
    7. 输出 dsl/registry/<framework>/*.yaml 草稿
    """
```

预期产出：每个 framework 跑一次 mining 输出 10-20 条候选 → 人工审核 5-10 条加进 registry。

### 5.4 mining 频率与维护

不是一次性挖完。框架每个 release 都会引入新 silent error 风险，需要 quarterly mining：
- 每季度对每个 framework 主分支跑一次 mining
- diff 新挖出的 vs 既有 registry → 增量 PR
- 配 CI：upstream commit 触发 mining sweep

---

## 6. 风险与限制（reviewer 应该追问的点）

### 6.1 Coverage 上限是真的 80%+ 吗

**不是**。Paper §3.4.5 doc 09 §高频静默错误分类显示，约 26% 的 silent error 需要 reference implementation 比较（B14 z_loss 类）或 checkpoint 字节级比较 — 这些**结构上不在 trainaudit T0-T3 范围**。T4 instance detector 可以做但每条都需要单独写 oracle，**摊不动**。

诚实的真实 cover rate 上限估计：**60-70%** 通过自动化能达到，剩下需要单实例 detector。

### 6.2 GPU 验证只跑了 2 ranks，真大规模可能炸

真。doc 23 §6.4 的 13/14 detection 都是 2-rank 设置。64-rank+ 必须实测，否则我无法保证：
- cross_rank_cksums all_gather 在 64 rank 不会 hang
- DataLoader hook 在 multi-process worker 不会重复 emit
- DuckDB 单文件并发写入不会 corrupt

**P0 #6 必须先跑过**才能 production deploy。

### 6.3 Async hook 改造可能引入新 bug

是。同步 → 异步是非小改：
- 事件序列化时机变了，依赖事件 ordering 的 rule（如 monotonic）需要保证 queue FIFO
- bg 线程崩溃 != 训练崩溃 — 需要 watchdog
- 改动需要全部 104 测试 + GPU 14 bug 重跑回归

### 6.4 LLM mining 在新 framework 上未必产生有用 hypothesis

是。现在用 stub LLM，stub 是 pattern match 加分类好的 fingerprint。换真 LLM 跑一次未知 framework，hypothesis 质量是 P1 #8 的关键风险点。最坏情况：mining 只产出已有 rule 的同义复述，新 coverage 为零。降低风险方案：先用 5 个高确定性 framework 文件验证 mining 流程，再扩。

### 6.5 部署 ergonomics 不充分

`trainaudit.enable()` API 当前对**框架开发者友好**（一行 import），但对**SRE/Infra 团队**不够：
- 没有 health check endpoint
- 没有 prometheus metrics export
- 没有 runbook（"violation X 怎么处理"）
- 没有 quickstart guide
- pip 安装包没发布（用户得 git clone）

这都是 P1/P2 的工作。

---

## 7. 立即开始的 1 个动作

如果只能选一件事 today 做，做这个：

**P0 #1：异步 hook executor**
- 输入：`core_trace/_utils.py` 的 emit 路径同步阻塞
- 输出：内存 queue + bg 线程异步刷盘，emit 内仅做 enqueue
- 效果：CPU toy 上 +962% → 预计 +20-50%；GPU 上预计落入 paper §4.3 目标 <5%
- 风险：事件 ordering、关闭时 flush、bg 线程崩溃
- 工作量：~3 天 + 完整回归

做完这一件，framework 才有资格"可以挂在生产 run 上不影响训练"。其他都是 nice-to-have。

---

## 8. 总结：3 个 question 的 reviewer-friendly 回答

**Q1: framework 现在能用在真实 production training 吗？**
A: 可以做 forensic（事后拿 trace 跑 verify），可以做 small-scale validation（≤2 rank toy/tiny），**不能**直接挂到 7B+ multi-node 长跑 — 缺 async + multi-rank aggregator + 大规模 FP 实测。最快 P0 完成（~2 周）就能挂。

**Q2: coverage 离 "高" 还差多少？**
A: 当前 13/48 reproduced = 27%。P1 完成后 25-30/48 = 50-60%；P2 完成后 30-40/48 = 60-80%。**>80% 需要 T4 instance detector 或 reference-impl 比较**，那是结构性限制不是工作量问题。

**Q3: 最大风险是什么？**
A: GPU 大规模未测（64+ ranks / 100k+ steps）+ async 改造引入新 bug + LLM mining 在新 framework 上质量未知。P0 #6 + 完整回归是必须的 mitigations。
