# Paper §4.3 — Production overhead optimization (P0 #1+#2)

> 2026-05-05 update — async hook executor + per-hookpoint sampling shipped.

Workload: `gpt-tiny` × 200 steps, CPU, batch=4, single rank.

| 配置 | mean_step (ms) | overhead | events | 说明 |
|---|---:|---:|---:|---|
| baseline (no trainaudit) | 19.5 | +0.0% | — | 训练裸跑 |
| sync, no sampling (旧基线) | 190.5 | +876.6% | 16038 | doc 22 §4 之前记录的 +962% 实测同量级 |
| async, no sampling | 247.0 | +1166.0% | 16038 | **CPU 上 thread contention 反而拖慢** — 期望见 GPU |
| **sync + sample fwd@0.1** | **51.1** | **+162.0%** | 2337 | **sampling 是 CPU 上真正的 overhead 杀手 (5.4× 加速)** |
| async + sample fwd@0.1 | 88.1 | +351.6% | 2337 | async + sampling 在 CPU 上不互补（queue 上下文切换 dominates） |
| async + sample fwd@0.05 | 64.1 | +228.6% | 1575 | rate 更低也没救 — 单 thread 模型 |

## 关键发现

1. **CPU 上 sampling 是唯一真正起作用的优化**。把 module.fwd.{pre,post} + module.bwd 三个高频 hookpoint 抽样到 10%，overhead 从 876% 降到 162%（**5.4× 加速**）。
2. **CPU 上 async hook executor 反而拖慢**。原因：toy model 单步只有 19ms，emit() 不在 critical path 上；async 的 queue.put_nowait + 后台 thread context switch 反而成 bottleneck。
3. **GPU 上 async 才会有效益**。原因：GPU 单步 200-2000ms，emit 路径里的 json.dumps + INSERT 是少量纯 CPU 工作可以隐藏；sampling 又能进一步 cut hook 计算到原来的 10%。预期 GPU 上 `async_mode=True + sample_rates={"module.fwd.post": 0.05, ...}` 落入 paper §4.3 < 5% overhead 目标。

## 推荐生产配置

```python
trainaudit.enable(
    tier=trainaudit.Tier.T1_FW_METADATA,
    db_path="/shared/runs/${RUN_ID}/trace.duckdb",
    async_mode=True,                              # 隐藏 emit 时的 json+INSERT
    async_queue_size=8192,
    sample_rates={
        "module.fwd.pre":  0.05,                  # 10× → 20× 降事件密度
        "module.fwd.post": 0.05,
        "module.bwd":      0.05,
        # 其他低频 hookpoint 不抽样：
        "utils.clip_grad.post": 1.0,
        "optim.step.post":      1.0,
        "build.snapshot":       1.0,
        "scheduler.init":       1.0,
    },
)
```

## 重要注意事项

**Sampling 影响哪些 rule**：
- ✅ 不受影响：T0-clip-grad-bounded（每次 clip 都看）/ T0-optim-lr-positive（build.snapshot 一次性）/ T1-replica-cksum-equal / T0-initial-lr-present 等基于低频 hookpoint 的 rule
- ⚠️ 部分受影响：T0-no-nan-inf 在 module.fwd.post 上扫，sampling 0.05 下只看 5% 的 forward → **真有 NaN 但落在未采样事件上时会漏**。生产 mitigation：`sample_rates["module.fwd.post"] = 1.0`，只抽样 module.fwd.pre 和 module.bwd（这两个事件量大但相对冗余 — fwd.post 已经能告诉你输出 stats）
- ⚠️ T0-norm-output-unit-rms / T0-softmax-degenerate：同样在 module.fwd.post 上检查，sampling 会成比例漏检

**生产级方案**：
- T0-no-nan-inf 类 critical-correctness rule 必须 sample_rate=1.0
- module.fwd.pre 和 module.bwd 可以激进抽样（0.01-0.05）
- module.fwd.post 保持 1.0 但 summarize_tensor 用 GPU-side `torch._foreach_norm` 批量化（P0 #2 的下一步优化方向）
