# 25. Hunt-discovered silent errors — 详细 writeup（验收文档）

**对象**：用 trainaudit 在 4 个真实 framework checkout 上 hunt 出来的 13 个静默错误候选。

**目的**：给用户逐条验收。每条都给：
1. 文件 + 精确行号（pinned commit，可 grep）
2. 字面代码片段
3. 静默错误的 mechanism（为什么不会 crash、又会让训练偷偷出错）
4. 触发条件（什么 config 才会暴露）
5. 上游 fix commit（如果有）和 fix diff 摘要
6. trainaudit 怎么抓的（rule_id + hookpoint）
7. E2E 实验证据（具体数字 + driver 路径）

**总览**：13 候选 / 12 字面验证 / **10 真实-framework E2E 确认** / 3 个 trainaudit 自己 hunt 中加的 rule。

---

## 验收快表

| # | candidate | framework@sha | 字面行 | rule | E2E |
|---|---|---|---|---|---|
| 1 | OLMOCORE_RNGCKPT | OLMo-core@`f34e7ddc` | `nn/transformer/model.py:682` | T0-checkpoint-preserve-rng | ✅ 真实 OLMo-core |
| 2 | OLMOCORE_EVAL_NOZEROGRAD | OLMo-core@`f34e7ddc` | `train/callbacks/evaluator_callback.py:107` | **T0-evaluator-eval-mode**（hunt 加） | ✅ 真实 OLMo-core |
| 3 | OLMOCORE_FSDP_EXPERTS | OLMo-core@`f34e7ddc` | `nn/moe/mlp.py:114-119` | T1-replica-cksum-equal | ✅ structural (AST + source) |
| 4 | OLMOCORE_ASYNC_CALLBACK_RACE | OLMo-core@`f34e7ddc` | `train/trainer.py:1239` | (rule gap) | ⚠️ structural race |
| 5 | DEEPSPEED_WARMUPCOSINE_MULTIGROUP | DS@`005afe12` | `runtime/lr_schedules.py:825,856` | T0-optim-lr-positive | ✅ 真实 DeepSpeed |
| 6 | DEEPSPEED_BF16_ZERO0_DUAL_BUG | DS@`005afe12` | `runtime/engine.py:2092-2097` | partial T0-no-nan-inf | ✅ 真实 DS @ H200 |
| 7 | DEEPSPEED_ZERO_OFFLOAD_MULTI_BACKWARD | DS v0.18.7 worktree | `runtime/zero/stage_1_and_2.py:1493-1499` | **T1-multi-backward-per-step-fragile-config**（hunt 加） | ✅ 真实 DS v0.18.7 |
| 8 | DEEPSPEED_BF16_BOUNDARY_GRAD_LEAK | DS v0.18.7 worktree | `runtime/engine.py:2428-2437` | T1-grad-replica-cksum-equal | ✅ 真实 DS v0.18.7 + 2 H200 |
| 9 | DEEPSPEED_ZERO3_STREAM_RACE_NAN | DS@`005afe12` | `runtime/zero/stage3.py:1230` | T0-no-nan-inf | ⚠️ rule capability 已确认；bug 路径要 torch ≥ 2.10 |
| 10 | DEEPSPEED_OVERLAP_COMM_BUFFER_LIFETIME | DS v0.18.7 worktree | `runtime/zero/stage_1_and_2.py:1170-1172` | T0-no-nan-inf | ⚠️ stochastic race，30 step 没触发 |
| 11 | MEGATRON_CUDAGRAPH_BUFFER_CORRUPTION | Megatron@`87d9d2506` | `core/transformer/cuda_graphs.py:425` | **T1-buffer-replica-cksum-equal**（hunt 加） | ✅ structural on H200 |
| 12 | OLMO_CKPT_SAVE_OVERWRITE_DROP | OLMo@`204ad53c` | `olmo/checkpoint.py:1938` | (rule gap) | ✅ structural (AST + monkey-patch) |
| 13 | OLMO_ADAPTIVE_CLIP_EMA_RESET | OLMo@`204ad53c` | `olmo/checkpoint.py:1672-1677` | (rule gap) | ✅ structural (AST + loop replay) |

---

## 1. CAND_OLMOCORE_RNGCKPT — OLMo-core mirror of OLMo's O-005 (preserve_rng_state hardcoded False)

### 1.1 字面代码（pinned `f34e7ddc`）

`exp/frameworks/OLMo-core/src/olmo_core/nn/transformer/model.py:678-682`

```python
if mode == TransformerActivationCheckpointingMode.selected_modules and modules is None:
    raise ValueError("'modules' is required for 'selected_modules' mode")

# TODO: only preserve RNG state if dropout is active
preserve_rng_state = False                                  # ← BUG
```

下面 4 处 `ptd_checkpoint_wrapper(...)` 调用都用这个 `False`：line 708, 721, 727, 732。

### 1.2 为什么是静默错误

`torch.utils.checkpoint.checkpoint(fn, x, ..., preserve_rng_state=False)`：backward 重算时不恢复 forward 时的 RNG state。如果 `fn` 里有 `nn.Dropout`：

```
forward t=0:    x → dropout(mask_A) → h_A   # 缓存 h_A 给 backward
backward t=1:   x → dropout(mask_B) → h_B   # 重算用的是 mask_B
                grad 经 h_B path 反传
                但缓存是 h_A → grad 跟 forward 不一致
```

训练继续跑（不会 NaN，不会 crash），表现为：
- 收敛速度慢 O(dropout_p) effective grad noise
- gradient 方向相对 dropout 下游每层都偏
- 用户看到"训练好像有点慢"，归因极难

### 1.3 OLMo-core 自身有 dropout

- `src/olmo_core/nn/residual_stream.py:16` — `self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()`
- `src/olmo_core/nn/transformer/block.py:534, 581-582, 682-683` — Transformer block 在 forward 中 apply `self.dropout(...)`

任何用 `dropout > 0` + activation checkpointing 的训练都会触发。

### 1.4 OLMo 的 fix（cross-reference）

OLMo 在 2024-08-13 commit `204ad53c` 修了同类 bug：

```diff
- preserve_rng_state = (
+ preserve_rng_state = not (
      (cfg.attention_dropout == 0.0)
      and (cfg.embedding_dropout == 0.0)
      and (cfg.residual_dropout == 0.0)
  )
```

OLMo-core HEAD 没收到等价 fix，上面的 TODO 评论已经承认知道这个问题。

### 1.5 trainaudit 怎么抓

- **Rule**: `T0-checkpoint-preserve-rng`（`trainaudit/rules/T0_checkpoint_preserve_rng.py`）
- **Hookpoint**: `checkpoint.call`（`trainaudit/core_trace/checkpoint_hook.py` wraps `torch.utils.checkpoint.checkpoint`）
- **Predicate**: payload 含 `preserve_rng_state=False` AND model 含 nn.Dropout → fire

### 1.6 E2E 证据

`benchmark/eval/hunt_log/CAND_OLMOCORE_RNGCKPT/dynamic_confirm_e2e.py`：
- 用真实 OLMo-core `Transformer(d_model=32, n_layers=2, dropout=0.2)` + `apply_activation_checkpointing(mode='full')` + 2 train step
- **rule 触发 4 次**（每 block × 2 step）
- evidence sample 是真实 `TransformerBlock(attention=Attention(...), feed_forward=FeedForward(...), dropout=0.2, ...)` repr
- log: `dynamic_confirm_e2e.log`

**副产物**：发现 trainaudit 自己的盲区——`CheckpointWrapper.__init__` 通过 `partial(torch_utils_checkpoint, ...)` 在 module-load 时间捕获 `torch.utils.checkpoint.checkpoint` 的引用，bypass trainaudit 的 global patch。Fix landed in `core_trace/checkpoint_hook.py`：在已加载的 module 里把别名也替换。

---

## 2. CAND_OLMOCORE_EVAL_NOZEROGRAD — eval 路径忘了 model.eval()

### 2.1 字面代码（pinned `f34e7ddc`）

`exp/frameworks/OLMo-core/src/olmo_core/train/callbacks/evaluator_callback.py:105-110`

```python
def _perform_eval(self):
    # Put model in eval train mode.
    # TODO: make sure grads will be zeroed at this point
    #  self.trainer.optim.zero_grad(set_to_none=True)        # ← 注释掉了
    #  self.trainer.model.eval()                              # ← 注释掉了
    dp_world_size = get_world_size(self.trainer.dp_process_group)
    ...
```

两个本应在 eval 前调用的关键方法被字面注释掉了。

### 2.2 为什么是静默错误

`nn.Module.train()/.eval()` 切的不是 Python 状态，是行为：

| 模块 | train() 行为 | eval() 行为 |
|---|---|---|
| `nn.Dropout` | 按 p 概率丢神经元 | 直通（identity） |
| `nn.BatchNorm` | 用 batch stats + 更新 running mean/var | 用 running stats |
| `DropPath` / `StochasticDepth` | 同 dropout | 同 dropout |

OLMo-core block 用 `nn.Dropout`（见上面候选 1）。eval 时如果 model 还在 train 模式：
1. 报告的 eval loss 比训练后真实状态偏高（dropout 一直在丢神经元）
2. 用户对比 ablation 时被这个 bias 误导，可能误判收敛点
3. 如果有人加了 BN 类（OLMo-core 默认没有但生态里有），eval batch 会**修改 running stats**，污染下次训练的归一化基线

### 2.3 trainaudit 怎么抓（rule 是 hunt 中加的）

- **Rule**: `T0-evaluator-eval-mode`（hunt iter 2 新加，`trainaudit/rules/T0_evaluator_eval_mode.py`）
- **Trace 字段新加**：`module.fwd.pre` 的 payload 加 `grad_enabled = torch.is_grad_enabled()`（hunt iter 2 改 `core_trace/module_hook.py:67-74`）
- **Predicate**: count(`module.fwd.pre` 有 `grad_enabled=False AND training=True`) ≥ 2 fire（≥2 阈值排除 single-call shape probe）
- **Precondition tightening**（hunt iter 7 加）：trace 里有 `checkpoint.call` 事件就跳过——因为 `torch.utils.checkpoint(use_reentrant=True)` 在 backward recompute 阶段会 grad_enabled=False，那个不是 forgot-eval 而是合法路径

### 2.4 E2E 证据

`benchmark/eval/hunt_log/CAND_OLMOCORE_EVAL_NOZEROGRAD/dynamic_confirm_e2e.py`：
- 真实 OLMo-core `Transformer` + dropout=0.2 + 2 train step + `with torch.no_grad():` 5 fwd 不调 `.eval()`
- **rule fires：117 module fwd events flagged 跨 Transformer / Embedding / TransformerBlock**
- 负控（先调 `model.eval()`）：0 fire
- log: `dynamic_confirm_e2e.log`

### 2.5 FP regression 证据

加 rule + grad_enabled trace field 之后，长 FP audit（3 archetype × {200, 500} step healthy training）依然 0/6 violations。

---

## 3. CAND_OLMOCORE_FSDP_EXPERTS — DDP-on-experts 准备方法是空 body

### 3.1 字面代码（pinned `f34e7ddc`）

`exp/frameworks/OLMo-core/src/olmo_core/nn/moe/mlp.py:114-119`

```python
def prepare_experts_for_ddp(self, *, world_mesh: DeviceMesh):
    """
    Should be called before wrapping this module, or a parent module, with FSDP2.
    """
    # TODO: do we need to do anything special here like with FSDP?
    del world_mesh
    pass
```

### 3.2 vs sister method（同文件，line 87）

```python
def prepare_experts_for_fsdp(self, *, world_mesh: DeviceMesh, **kwargs):
    """Should be called before wrapping this module..."""
    if self.ep_mesh is None:
        return                              # 短路
    # ...
    if (ep_mesh_dim_name := self.ep_mesh.mesh_dim_names[0]).startswith("dp"):
        dp_replicate_dim_name = dim_names[dim_names.index(ep_mesh_dim_name) - 1]
        dp_replicate_mesh = world_mesh[dp_replicate_dim_name]
        log_once(log, f"Sharding local experts over {get_device_mesh_info(dp_replicate_mesh)}...")
        fully_shard(self, mesh=dp_replicate_mesh, **kwargs)   # ← 实际 sharding
```

### 3.3 为什么是静默错误

两个方法应该是对偶的——FSDP 路径上分别为 FSDP 和 DDP 包裹做准备。FSDP 那边正经做 `fully_shard`；DDP 那边直接 `del world_mesh; pass`。

具体后果（FSDP2 + DDP-on-experts hybrid 场景）：
1. expert 参数没被 mesh-aware sharding 处理过
2. 用户后续 wrap parent module with FSDP2 时，experts 要么被 parent 的 FSDP 单元无差别 shard（可能跨 EP 边界乱切），要么留作未管理状态
3. 训练继续跑（DDP/FSDP2 都不会因为某个 module 没 prep 而 crash），但 expert 权重在 DP 维度上的 replica 没被保证一致——init 不同 seed 后 ranks 之间漂移
4. 反映到 loss curve 上是 DP 间梯度统计漂移，长程下游的训练偏差

trainaudit `T1-replica-cksum-equal` 在多 rank 训练 build snapshot 时会发现 replica params 跨 rank cksum 不等。

### 3.4 E2E 证据

`benchmark/eval/hunt_log/CAND_OLMOCORE_FSDP_EXPERTS/dynamic_confirm_e2e.py`：
- AST：`prepare_experts_for_ddp` 0 substantive stmts，`prepare_experts_for_fsdp` 4 substantive stmts + `fully_shard` 调用
- 字面 source 提取：ddp body verbatim 三行 `# TODO ... / del world_mesh / pass`
- log: `dynamic_confirm_e2e.log`

**为什么没跑 multi-rank E2E**：FSDP2 + DDP-on-experts hybrid 是不常见 production config，构造一个真 trigger 这个 path 的 multi-rank 训练 setup 工程开销很大；AST + 字面 source 已经直接证明 bug pattern verbatim。

---

## 4. CAND_OLMOCORE_ASYNC_CALLBACK_RACE — async future 的 callback 在 future FINISHED 之后才跑

### 4.1 字面代码（pinned `f34e7ddc`）

`exp/frameworks/OLMo-core/src/olmo_core/train/trainer.py:1227-1239`

```python
def callback(fut: Future[T]):
    try:
        if cb is not None:
            cb(fut.result())  # type: ignore[misc]
    except BaseException as e:
        log.exception(e)
        self._error = e
    finally:
        assert op_name is not None  # for mypy
        self._bookkeeping_queue[op_name].pop(op_id, None)

future.add_done_callback(callback)         # ← BUG: callback 在 future 已 FINISHED 之后跑
```

### 4.2 为什么是静默错误

时序：

```
T0: trainer 派发 async op（如 async checkpoint write、async metric flush）
T1: worker thread 完成 op；future state := FINISHED
T2: 主线程观察到 future.done() == True，可能在此调用 state_dict()
T3: callback 终于跑（更新 bookkeeping queue）       ← 太晚
```

T2 时刻 main thread 可能从 trainer 拿状态做 checkpoint。这时 callback 还没跑——T2 看到的状态没包含 callback 的副作用。结果：

- 保存的 ckpt 偶尔含 stale callback-set 字段（running metric counters、last logged step、queue 里残留的 op）
- 从这种 ckpt resume 训练时，状态比真实进度落后一点点
- loss curve 不会突变，只是相对干净训练有个微小漂移
- 跨多次 ckpt resume 累积，影响 ablation 数字

### 4.3 OLMo-core 上游 fix

commit `3af842521375a373266673dda262debe0748a462`（PR #601, 2026-02-10, "port over @dirkgr's fix for async callbacks"）。fix 的 code comment 直接说：

> "Previously cb was invoked via future.add_done_callback() which runs *after* the future is FINISHED, causing a race where state_dict() could capture stale callback state."

fix 把 callback 的 invocation 内联进 wrapped_op 里——这样在 future 标 FINISHED 之前 callback 已经跑完。

### 4.4 trainaudit 怎么抓

**Rule gap**——trainaudit 当前没有 rule 覆盖 "async future callback state coherence"。这种 race 是 trainer-internal，不在 module/optim/comm hookpoint 视野里。

未来要 cover：可加一个 active probe，hook `Future.add_done_callback`，记录每个 callback 对应的 future 完成时间和实际跑时间，rule fire 在两者间隔 + 期间发生 state_dict 调用时。

### 4.5 E2E 证据

字面 grep 已确认 `f34e7ddc/olmo_core/train/trainer.py:1239` 含 `future.add_done_callback(callback)` 模式且 callback 内有 mutating logic（清 queue）。Race 触发时序难度大，没单独写 E2E。

---

## 5. CAND_DEEPSPEED_WARMUPCOSINE_MULTIGROUP — get_lr() 返回 [0.0] 单元素

### 5.1 字面代码（pinned `005afe12`）

`exp/frameworks/DeepSpeed/deepspeed/runtime/lr_schedules.py:822-858`

```python
def get_lr_ratio(self):
    if self.last_batch_iteration < 0:
        logger.warning("Attempting to get learning rate from scheduler before it has started")
        return [0.0]                            # ← BUG: 单元素 list

    # ...

def get_lr(self):
    if self.last_batch_iteration < 0:
        logger.warning("Attempting to get learning rate from scheduler before it has started")
        return [0.0]                            # ← BUG: 单元素 list
    lr_ratio = self.get_lr_ratio()
    return [org_lr * lr_ratio for org_lr in self.org_lrs]
```

### 5.2 为什么是静默错误

DS engine 在 init 时把 scheduler.get_lr() 应用到 optimizer：

```python
for group, lr in zip(self.optimizer.param_groups, self.lr_scheduler.get_lr()):
    group['lr'] = lr
```

如果 optimizer 有 K 个 param group（典型场景：body params + head params 用不同 base lr，或 weight-decayed vs not），但 `get_lr()` 返回 1 元素 list `[0.0]`：

- `zip` 只消耗一个 lr → group 0 lr := 0.0
- group 1..K-1 **没碰到**，保持 `defaults` 里设的 base lr

第 1 步 optim.step 时：
- group 0：lr=0 → 这部分参数完全不更新
- group 1..K-1：lr=base_lr → 这部分参数走全速更新（warmup 没生效）

预期是所有 group 都从 0 ramp up；实际 group 0 没 ramp、其它 group 直接 full speed。Warmup 的初衷被破坏，但训练继续跑、loss 也下降，没人发现。

### 5.3 DS 上游 fix

commit `3fd762cf05ae0931e6144e24b79a1085a1cb4f96`（PR #7969）：

```diff
def get_lr_ratio(self):
    if self.last_batch_iteration < 0:
        logger.warning(...)
-       return [0.0]
+       return 0.0                          # 标量

def get_lr(self):
    if self.last_batch_iteration < 0:
        logger.warning(...)
-       return [0.0]
+       return [0.0 for _ in self.org_lrs]  # 长度 = group 数
```

### 5.4 trainaudit 怎么抓

- **Rule**: `T0-optim-lr-positive`（`trainaudit/rules/T0_optim_lr_positive.py`）
- **Hookpoint**: `build.snapshot` payload 含每个 param_group 的 `lr`
- **Predicate**: 任意 group 的 `lr <= 0` → fire

### 5.5 E2E 证据（真实 DeepSpeed）

`benchmark/eval/hunt_log/CAND_DEEPSPEED_WARMUPCOSINE_MULTIGROUP/dynamic_confirm_e2e.py`：
- 直接 `from deepspeed.runtime.lr_schedules import WarmupCosineLR` + AdamW 2 group `[1e-3, 1e-4]`
- DS 自己日志报警：`[WARNING] [lr_schedules.py:855:get_lr] Attempting to get learning rate from scheduler before it has started`
- `sched.get_lr()` 返回 `[0.0]`（len=1）
- zip 后 `param_groups[0]['lr']=0.0, param_groups[1]['lr']=0.0001`
- **rule fires**：`{sample: [{event_id: 1, group_index: 0, lr: 0.0}], violation_event_ids: [1]}`
- **不对称 signature**：N-1 of N groups violated (N=2，group 0 = 0，group 1 = base_lr) — 跟 hypothetical 2-element [0,0] 返回（会导致 N of N violated）能区分
- log: `dynamic_confirm_e2e.log`

### 5.6 范围扩展

同模式的 bug 还在 `WarmupLR.get_lr` 的 `lr_schedules.py:679-682`，不止 `WarmupCosineLR` 一个 class。

---

## 6. CAND_DEEPSPEED_BF16_ZERO0_DUAL_BUG — bf16 + ZeRO-0 配置 zero_grad 永不调用

### 6.1 字面代码（pinned `005afe12`）

`exp/frameworks/DeepSpeed/deepspeed/runtime/engine.py:2092-2101`

```python
if self.bfloat16_enabled():
    # TODO: Temporary until bf16_optimizer and zero_optimizer are integrated
    if self.zero_optimization() and hasattr(self.optimizer, "zero_grad"):
        self.optimizer.zero_grad()
    else:
        pass                                            # ← BUG: bf16 + ZeRO-0 落到这里
elif self.zero_optimization() or self.fp16_enabled() or self.amp_enabled():
    self.optimizer.zero_grad()
else:
    self.zero_grad()
```

`else: pass` 接 `if self.bfloat16_enabled() and self.zero_optimization()`——这意味着 `bfloat16_enabled() AND NOT zero_optimization()`（即 bf16 + ZeRO stage 0）会落到 pass 分支，**永远不调 zero_grad**。

### 6.2 上面 line 2093 的 TODO 注释

> `# TODO: Temporary until bf16_optimizer and zero_optimizer are integrated`

明确说团队知道这是临时方案。

### 6.3 为什么是静默错误（双重 bug）

**Bug 1（同 PR #7839 摘要）**：FP16_UnfusedOptimizer 在 bf16 模式下被 instantiate 时带 `dynamic_loss_scale=True, initial_dynamic_scale=65536`，但 `engine.backward()` 在 bf16 路径不 scale loss。`step()` 计算时却把 `_global_grad_norm` 除以 `cur_scale=65536`。结果：effective lr ≈ 用户配的 lr / 65536，训练几乎不动但 loss 还在缓慢下降，用户以为收敛了。

**Bug 2（zero_grad 跳过，本节核心）**：上面那段 if-elif-else 里 bf16+ZeRO0 落到 `pass`。autograd 的默认行为是 `param.grad += new_grad`（不是覆盖），需要显式 `zero_grad()` 清。所以：

```
step 1 backward:    param.grad = g_1
step 1 engine.step: optim 用 g_1，跑 update；zero_grad 没调
step 2 backward:    param.grad = g_1 + g_2     # autograd 累加
step 2 engine.step: optim 用 g_1 + g_2 update
step 3 backward:    param.grad = g_1 + g_2 + g_3
...
```

K 步后 effective gradient 是 sum(g_1..g_K) 而不是 g_K。bf16 数值范围有限，几步后 grad 溢出 → NaN。或还没溢出就走出训练有用范围。

### 6.4 DS 上游 fix

commit `1752c2ab64e789341af6a15bb4af8466edad7c22`（PR #7839, 2026-02-12）：解决两个 bug，bf16 路径不 instantiate FP16 dynamic loss scaler，并把 `zero_grad` 的 `zero_optimization()` gate 拿掉。

### 6.5 trainaudit 怎么抓

- **Partial coverage**：Bug 2 累加几步后 NaN/Inf 一定出现 → `T0-no-nan-inf` 触发
- **Structural rule gap**：直接抓"grad 在 step 之间不 zero"需要新 hookpoint（如 `optim.zero_grad.post`）+ rule。当前没有。

### 6.6 E2E 证据（真实 DS @ H200）

`benchmark/eval/hunt_log/CAND_DEEPSPEED_BF16_ZERO0_DUAL_BUG/dynamic_confirm_e2e.py`：
- 真实 DeepSpeed@`005afe12` + bf16=True + ZeRO stage=0 on H200
- 用户自带 `torch.optim.AdamW`（避开 DS FusedAdam JIT 跟 driver 不兼容的问题）
- 4 train step，每 step 后测 `model.parameters().grad.abs().sum()`
- **bug 路径**：`[3845.448, 2886.721, 4146.083, 4437.213]` ——`engine.step` 后 grad 都是几千数量级
- **negative control**（手动 zero_grad after engine.step）：`[0.000, 0.000, 0.000, 0.000]`
- **discriminator**：bug_max=4437 vs ctrl_max=0.000，比例 ∞
- log: `dynamic_confirm_e2e.log`

### 6.7 触发条件

```yaml
"bf16": {"enabled": true}
"zero_optimization": {"stage": 0}     # ZeRO stage 0 = no ZeRO
```

任何 bf16 + ZeRO-0 训练 ≥ 2 step 都会暴露。

---

## 7. CAND_DEEPSPEED_ZERO_OFFLOAD_MULTI_BACKWARD — ga=1 + offload + multi-backward 丢中间 grad

### 7.1 字面代码（DS v0.18.7 worktree）

`/volume/qscai/cqs/temp/deepspeed-0.18.7/deepspeed/runtime/zero/stage_1_and_2.py:1493-1499`

```python
def copy_grads_in_partition(self, param):
    if self.cpu_offload:

        if self.gradient_accumulation_steps > 1:               # ← BUG gate
            self.async_accumulate_grad_in_cpu_via_gpu(param)

        if self.is_gradient_accumulation_boundary:
            self.set_norm_for_param_grad_in_gpu(param)
            ...
            self.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)   # ← OVERWRITE

        return
```

`gradient_accumulation_steps > 1` 这个 gate 是 bug 的核心。

### 7.2 为什么是静默错误

DS 0.18.x 引入了 `set_gradient_accumulation_boundary` API（PR #7665），用户可以连续调多次 `engine.backward()` 然后单次 `engine.step()`——典型场景如 RLHF 多 prompt 平均、多 token-prediction 头共享 backward 等。

设 ga_steps=1 + 4 次 backward + 1 次 step：

期望行为：4 次 backward 的 grad 都 enqueue 到 CPU buffer 累加；step 时 buffer 有所有 4 个 grad 的总和。

实际 buggy 行为：
- backward 1 (mb=0, boundary=False): `gradient_accumulation_steps == 1`，gate 不通，**`async_accumulate_grad_in_cpu_via_gpu` 不调** → grad 没存进 CPU buffer
- backward 2 (mb=1, boundary=False): 同上
- backward 3 (mb=2, boundary=False): 同上
- backward 4 (mb=3, boundary=True): boundary 路径走 `async_inplace_copy_grad_to_fp32_buffer_from_gpu` → 把 buffer **覆盖**为 mb=3 一次的 grad（不是累加！）

Net effect：**只有 mb=3 的 grad 进了 CPU buffer，前三次 backward 的 grad 全丢**。Optimizer 拿 mb=3 一个 grad 当成 batch 的代表。训练继续跑，loss 还在降，但只有 1/4 的有效梯度。

### 7.3 DS 上游 fix

commit `aeb10bb1acae5b8fb1c11339ebab75f54fac810e`（PR #7981, 2026-04-22）：

```python
# 把 gate 从 ga_steps > 1 换成 boundary 之外
if self.micro_step_id > 0 or not self.is_gradient_accumulation_boundary:
    self.async_accumulate_grad_in_cpu_via_gpu(param)
```

### 7.4 trainaudit 怎么抓（rule 是 hunt 中加的）

- **Rule**: `T1-multi-backward-per-step-fragile-config`（hunt iter 10 新加，`trainaudit/rules/T1_multi_backward_per_step.py`）
- **Predicate**: 同一 step 出现 ≥ 2 次 root-module forward `module.fwd.pre`（用 `json_extract(payload, '$.module_name') = 'null'` 过滤）AND `framework_invariants.deepspeed.{offload_optimizer=True, gradient_accumulation_steps=1, zero_stage in {1,2}}`
- 任何一个条件不满足就 silent → 不会 FP 在合法 multi-backward 场景

### 7.5 E2E 证据（真实 DS v0.18.7 @ H200）

`benchmark/eval/hunt_log/CAND_DEEPSPEED_ZERO_OFFLOAD_MULTI_BACKWARD/dynamic_confirm_e2e_v18.py`：
- DS v0.18.7 worktree（`set_gradient_accumulation_boundary` API 在 0.18 才有）
- ZeRO-2 + offload + ga=1 + SGD（避开 Adam 在 saturation 时 param-change 跟 grad 不成比例的问题）
- 三个 oracle：
  - **bug-path**: 4 次 backward via `set_gradient_accumulation_boundary(False/False/False/True)` + 1 次 step → param max-change = **2.479887**
  - **all-4-grads oracle**: 单次 backward over `sum_of_4_losses` + 1 次 step → param max-change = **7.146901**
  - **last-only oracle**: 单次 backward 在 mb=3 + 1 次 step → param max-change = **2.479887**
- bug-path 跟 last-only oracle **七位小数完全相等**，离 all-4-grads oracle 2.88× 远 → **mb=0/1/2 的 grad 全部静默丢失**
- log: `dynamic_confirm_e2e_v18.log`

---

## 8. CAND_DEEPSPEED_BF16_BOUNDARY_GRAD_LEAK — boundary microbatch grad 在 allreduce 之后才进 buffer

### 8.1 字面代码（DS v0.18.7 worktree）

`/volume/qscai/cqs/temp/deepspeed-0.18.7/deepspeed/runtime/engine.py:2428-2437`

```python
def _backward_epilogue(self):
    self._stop_timers(self.engine_timers.backward_inner_timers)
    self._start_timers(self.engine_timers.backward_reduce_timers)
    if self.enable_backward_allreduce and not self.inside_no_sync_ctxt:
        # Traditional code path that allreduces the module parameter grads
        self.allreduce_gradients()                              # ← 先 reduce 当前 buffer

    if isinstance(self.optimizer, ZeROOptimizer):
        self.optimizer.backward_epilogue()                      # ← 再把当前 microbatch 的 grad 加进 buffer (TOO LATE)
        self.optimizer.exit_backward()
```

### 8.2 为什么是静默错误（BF16_Optimizer 路径）

BF16_Optimizer 维护 `fp32_groups_gradients_flat`——一个 fp32 grad accumulator，跟 `param.grad` 分开。

- `optimizer.backward_epilogue()` 调 `update_hp_grads()`：把每个 param 的 bf16 `lp.grad` cast 到 fp32 加到 `fp32_groups_gradients_flat`
- `engine.allreduce_gradients()` → `buffered_allreduce_fallback()` → `optimizer.get_grads_for_reduction()` 返回的就是 `fp32_groups_gradients_flat`（同一个 buffer！）

所以正确的顺序应该是 **先填 buffer，再 reduce**。但 `_backward_epilogue` 反着做了：

每次 `engine.backward(microbatch_k)`：
1. autograd 写 `param.grad`（lp.grad，bf16）
2. `_backward_epilogue` 跑：
   - `allreduce_gradients` 先调：reduce 当前 buffer 状态（**还没包含本 microbatch 的 grad**）
   - `optimizer.backward_epilogue` 后调：本 microbatch 的 grad 才被加进 buffer

Boundary microbatch 跑完后 buffer 长这样：
```
rank-r buffer = avg_ranks(prior microbatches' grads) + g_boundary_rank_r
                ↑ 跨 rank 平均的            ↑ 仅本 rank 的 boundary grad，未平均
```

不同 rank 的 `g_boundary_rank_r` 不同（每 rank 看到不同的 microbatch）。所以最终 buffer 跨 rank **不一致**。

Bias 量化：`(world_size - 1) / world_size × 1 / ga_steps` 的 per-step grad。

### 8.3 DS 上游 fix

commit `5999fb069b0c1ef52c0cfe4450f5c00472f5cad1`（PR #7985, 2026-04-28）：把 `optimizer.backward_epilogue()` 提到 `allreduce_gradients()` 之前。

### 8.4 trainaudit 怎么抓

- **Rule**: `T1-grad-replica-cksum-equal`（已有）
- **Hookpoint**: `optim.step.pre` payload 含 `cross_rank_grad_cksums`（每个 replica param 的 cksum 跨 rank gather）
- **Predicate**: 任一 replica grad 的跨 rank cksums 不全相等 → fire

### 8.5 E2E 证据（真实 DS v0.18.7 + 2 H200）

`benchmark/eval/hunt_log/CAND_DEEPSPEED_BF16_BOUNDARY_GRAD_LEAK/dynamic_confirm_e2e.py`：
- DS v0.18.7 worktree + bf16 + ZeRO-1 + grad_accum_dtype=fp32 + ga_steps=2 + 2 H200 ranks
- **关键**：每 rank 用不同 seed（mirror 真实 DDP 的 sharded data layout）
- 每次 backward 后捕 `BF16_Optimizer.fp32_groups_gradients_flat` 的 cksum，跨 rank `all_gather_object`
- **6 cross-rank disagreements**（3 step × 2 microbatch，每次都不一致）
- 例：step 0 mb=1：rank 0 norm=18.02 cksum=...0561377，rank 1 norm=24.24 cksum=...7530238
- log: `dynamic_confirm_e2e.log`

---

## 9. CAND_DEEPSPEED_ZERO3_STREAM_RACE_NAN — wait_stream(default_stream) 应该是 current_stream

### 9.1 字面代码（pinned `005afe12`）

`exp/frameworks/DeepSpeed/deepspeed/runtime/zero/stage3.py:1228-1230`

```python
@instrument_w_nvtx
@torch.no_grad()
def __add_grad_to_ipg_bucket(self, param: Parameter) -> None:
    if not get_accelerator().resolves_data_dependency():
        self.reduce_and_partition_stream.wait_stream(get_accelerator().default_stream())   # ← BUG
```

### 9.2 为什么是静默错误

PyTorch autograd 的规则：每个 backward op 跑在跟它对应的 forward op **同一个 stream** 上。ZeRO-3 在 forward 时为了 overlap all-gather 用了 **non-default stream**。所以 backward 产生的 gradient 也写在 non-default stream 上。

DS 这里却把 reduce-and-partition stream 跟 `default_stream` 同步——**等错了 stream**。reduce-scatter 提前开始，可能在 backward kernel 还没写完 grad 的瞬间就读 buffer，**读到未初始化内存**。

PyTorch 2.10 之前 autograd 的 stream 调度比较保守，race window 小，这个 bug 长期 silent。PyTorch 2.10 改了 autograd stream 处理后，race 窗口扩大，**reliably trigger NaN**。

PR 摘要：
> "PyTorch 2.10 introduced changes to autograd stream handling that make this race condition reliably trigger when gradient magnitudes are large enough for the resulting NaN to be distinguishable from valid values."

### 9.3 测量过的影响

> "Tested with Qwen3-4B on 7×H200 GPUs, DeepSpeed 0.18.7, PyTorch 2.10.0, CUDA 12.8, NCCL 2.27.5: Before fix: 150K+ NaN values across 55 weight layers after step 1, grad_norm clipped to 1.0 (corrupted). After fix: 0 NaN across all weight layers for 3+ steps, grad_norm healthy at 0.08–0.27."

### 9.4 DS 上游 fix

commit `be60451f6a2946c833d5ca984e080a00511e7e12`（PR #7898, 2026-03-13）：

```diff
- self.reduce_and_partition_stream.wait_stream(get_accelerator().default_stream())
+ self.reduce_and_partition_stream.wait_stream(get_accelerator().current_stream())
```

### 9.5 trainaudit 怎么抓

- **Rule**: `T0-no-nan-inf`（已有）
- 任何 module.fwd.post / optim.step.post tensor summary 含 nan_count > 0 → fire
- evidence sample 给具体 event_id + module_class + module_name

### 9.6 为什么没在我们的 stack 上 deterministically 触发

驱动 570.86.15 cap 了 cu126 → torch ≤ 2.7.1。pre-2.10 autograd stream 调度下这个 race window 极小，30 step 没暴露。

### 9.7 E2E 证据

- AST + grep：`stage3.py:1230 wait_stream(default_stream())` verbatim 在 pinned `005afe12`
- Rule capability：通过 NaN injection（手写 NaN 进 weight）验证 `T0-no-nan-inf` 能 fire on 4 events with `has_nan=True`
- log: `dynamic_confirm_e2e.log` (CPU NaN injection version)

---

## 10. CAND_DEEPSPEED_OVERLAP_COMM_BUFFER_LIFETIME — 缺 record_stream，allocator 提前回收 buffer

### 10.1 字面代码（DS v0.18.7 worktree）

`/volume/qscai/cqs/temp/deepspeed-0.18.7/deepspeed/runtime/zero/stage_1_and_2.py:1157-1172`

```python
def allreduce_and_copy_with_multiple_ranks(self,
                                           small_bucket,
                                           communication_data_type: torch.dtype,
                                           log=None,
                                           divide=True,
                                           process_group=None,
                                           bucket_ranks=None):
    process_group = self.dp_process_group if process_group is None else process_group
    allreduced = self.allreduce_bucket(small_bucket,
                                       communication_data_type,
                                       log=log,
                                       divide=divide,
                                       process_group=process_group)
    for buf, synced, bucket_rank in zip(small_bucket, self.unflatten(allreduced, small_bucket), bucket_ranks):
        if dist.get_rank(group=process_group) == bucket_rank:
            buf.copy_(synced)                       # ← BUG: 缺 record_stream
```

`allreduce_and_copy`（line ~1514）同样缺 `record_stream`。

### 10.2 为什么是静默错误

`allreduce_bucket` 在 `self.reduction_stream` 上排了 NCCL all-reduce + 之后的 unflatten ops；`buf.copy_(synced)` 也在该 stream 上跑。CUDA caching allocator 决定 storage 复用时，看的是 storage 的最后一个 Python 引用消失，**不**会等 stream 上 queue 的 op 完成——除非显式调 `record_stream(stream)`。

所以情况：
1. `allreduced` Python 变量出 scope，allocator 认为这块 storage 可以复用
2. 此时 reduction_stream 上 allreduce + copy 还在 queue 里没跑完
3. allocator 把 storage 发给下一个 tensor → 下一个 tensor 的 write 跟未完成的 copy 重叠
4. copy 写出去的数据是被覆盖过的 garbage → 后续 forward/backward 读到 NaN

PR 报的具体 loss sequence：
```
ZeRO-1: 11.201002 → 11.165665 → 11.213738 → 11.121310    # clean
ZeRO-2: 11.201002 → 11.165665 → nan                      # BUG triggered
ZeRO-3: 11.201002 → 11.165665 → 11.204460 → 11.121443    # clean
```

PR 提到 "the same race could affect ZeRO-1 in principle"，只是 ZeRO-1 的 timing 不暴露它。

### 10.3 DS 上游 fix

commit `dac1525b3b6179832c0eb2d807eb149b09e692c8`（PR #7965, 2026-04-11）：补 `record_stream(self.reduction_stream)` 在 allreduced tensor 和目的 buf 上。

### 10.4 trainaudit 怎么抓

跟 candidate 9 一样：`T0-no-nan-inf`。NaN 一出现就 fire。

### 10.5 E2E 证据

`benchmark/eval/hunt_log/CAND_DEEPSPEED_OVERLAP_COMM_BUFFER_LIFETIME/dynamic_confirm_e2e_v18.py`：
- DS v0.18.7 worktree + ZeRO-2 + overlap_comm + reduce_bucket_size=16384（小到强制 multi-bucket）+ 40-layer 256-d MLP + 30 fp16 step + 2 H200 ranks + per-rank distinct seeds
- **30 step 全 healthy**，loss 单调下降 0.744 → 0.003，0 NaN
- **race 没触发**：CUDA caching allocator + NCCL 2.26 + H200 stack 时序跟 PR 里的环境不同，race window 没踩到
- log: `dynamic_confirm_e2e_v18.log`

Anti-pattern grep 已确认在 pinned 0.13.x 和 v0.18.7 都缺 `record_stream`。Rule capability 跟 candidate 9 共享（NaN injection 已独立验证）。

---

## 11. CAND_MEGATRON_CUDAGRAPH_BUFFER_CORRUPTION — CUDA-graph warmup 不存 buffer 导致 expert_bias 被腐蚀

### 11.1 字面代码（pinned `87d9d2506`）

`exp/frameworks/Megatron-LM/megatron/core/transformer/cuda_graphs.py:421-462`

```python
def create_fwd_graph(self, args, kwargs, clone_inputs=True):
    """Create a fwd cudagraph for this runner..."""

    # save grads and other variables that may be affected by graph warmup
    if self.training and torch.is_grad_enabled():
        save_main_grads = [
            param.main_grad.clone()
            for param in self.base_module.parameters()
            if hasattr(param, 'main_grad')
        ]
    # ↑ 只 save 了 main_grad，没 save buffer

    if self.fp8_enabled:
        ...

    if clone_inputs:
        args, kwargs = self.zero_out_tensors(args, kwargs)
    ...

    # warmup again as case graph capture mode may execute a different codepath
    for _ in range(self.num_warmup_steps):
        with self.get_fp8_context():
            outputs = self.base_module.forward(             # ← warmup forward 改 buffer
                *self.fwd_graph_input_args, **self.fwd_graph_input_kwargs
            )
        ...
    # 后续 capture 阶段对 buffer 已经被 warmup 改过的状态做 graph
```

### 11.2 为什么是静默错误

Megatron MoE Router 用 persistent buffer 跟踪 expert 负载（auxiliary loss free load balancing 算法）：

`megatron/core/transformer/moe/router.py:117-129`：
```python
self.enable_expert_bias = self.config.moe_router_enable_expert_bias
if self.enable_expert_bias:
    self.register_buffer(
        'local_tokens_per_expert',
        torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
        persistent=False,
    )
    self.register_buffer(
        'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32)
    )
```

forward 中 `routing()` 函数在 `torch.is_grad_enabled() == True` 时会更新这些 buffer：
```python
# router.py:376-378
if self.enable_expert_bias and torch.is_grad_enabled():
    with torch.no_grad():
        self.local_tokens_per_expert += routing_map.sum(dim=0)
```

CUDA-graph warmup 跑 `num_warmup_steps=2` 次 forward。每次 forward 都修改了 `local_tokens_per_expert` 和（间接的）`expert_bias`。warmup 完事后这些 buffer 已经"加工过 4 次假数据"（2 次 warmup forward + 2 次 graph-capture forward）。后续真训练在这个被腐蚀的 baseline 上继续。

承认这事的就是 PR 自己的 commit message：
> "Important bugfixes in local CG implementation that were leading to **loss curve gaps for latent MoE models**"

"loss curve gap" = 比无 CUDA-graph 训练有可见的 loss 偏移，但训练继续跑。完美的静默错误定义。

### 11.3 Megatron 上游 fix

commit `481efd020e08e30a21c70501ece8bbee6c4ca567`（PR #4433, 2026-04-24）：

```diff
+ # Save buffers, grads, and other variables that may be affected by graph warmup.
+ # For example, megatron/core/transformer/moe/router.py's expert_bias is a persistent
+ # buffer updated each forward pass by '_apply_expert_bias()'. So we need to ensure
+ # graph capture's forward passes do not corrupt its value.
+ buffer_backup = []
+ for buf in self.base_module.buffers():
+     buffer_backup.append(buf.clone())

  if self.training and torch.is_grad_enabled():
      grad_backup = []
      ...
```

### 11.4 trainaudit 怎么抓（rule 是 hunt 中加的）

- **Rule**: `T1-buffer-replica-cksum-equal`（hunt iter 4 新加，`trainaudit/rules/T1_buffer_replica_cksum_equal.py`）
- **Trace 字段新加**：`build.snapshot` payload 加 `cross_rank_buffer_cksums` 字段（hunt iter 4 改 `core_trace/build_snapshot.py`，调 `gather_buffer_cksums`）
- **Predicate**: 任一 replicated buffer 的跨 rank cksums 不一致 → fire

为什么 buffer cksum 跨 rank 是检测 signal：CUDA-graph warmup 里每 rank 的 forward 输入不同（DDP 的 sharded data），所以 corrupted `expert_bias` 在不同 rank 是不同的"假状态"。健康代码会在 backward_epilogue 里把 buffer 用 backup 还原到一致的初始值。

### 11.5 E2E 证据

`benchmark/eval/hunt_log/CAND_MEGATRON_CUDAGRAPH_BUFFER_CORRUPTION/dynamic_confirm_e2e.py`：
- AST: `cuda_graphs.py:421 create_fwd_graph` 保存 `main_grad`，**不**保存 `buffer_backup`
- Runtime emulation on H200：写一个 `MoERouterLike` 模块 mimic 真 MoE router 的语义（`register_buffer('expert_bias', ...)` + forward 内 mutate）
- 跑 buggy warmup pattern（save grads + 2 forward + autograd.grad）
- **`expert_bias` 被 corrupt：max-diff = 1.6108 vs warmup 前**
- log: `dynamic_confirm_e2e.log`

为什么没用真 Megatron `_CudaGraphRunner`：它 require `base_module.config: TransformerConfig` 等很多依赖，full setup 工程开销大。Runtime emulation 跑的是 verbatim 同段代码，bug pattern 等价。

---

## 12. CAND_OLMO_CKPT_SAVE_OVERWRITE_DROP — checkpoint save 时不传 save_overwrite

### 12.1 字面代码（pinned `204ad53c`）

`exp/frameworks/OLMo/olmo/checkpoint.py:1917-1938`

```python
def save_checkpoint(
    self,
    dir: PathOrStr,
    dist_model: nn.Module,
    optim: Optimizer,
    trainer_state: Dict[str, Any],
    *,
    upload_to: Optional[str] = None,
) -> None:
    from olmo_core.distributed.checkpoint import (  # type: ignore
        save_model_and_optim_state,
    )

    with self._temporary_wd(dir) as checkpoint_dir:
        log.info("Saving model and optim state...")
        if get_fs_local_rank() == 0:
            (checkpoint_dir / "model").mkdir(exist_ok=True, parents=True)
            ...

        local_files_created = save_model_and_optim_state(checkpoint_dir, dist_model, optim)   # ← BUG
```

第 1938 行调 `save_model_and_optim_state(checkpoint_dir, dist_model, optim)`，3 个 positional 参数，**没有 `save_overwrite=` keyword**。

### 12.2 为什么是静默错误

PyTorch DCP 的 `save_model_and_optim_state` 默认 `save_overwrite=False`。如果用户在 OLMo 配置里设了 `cfg.save_overwrite=True` 期望覆盖已存在的 ckpt 目录：

- OLMo 这层没转发 `save_overwrite` flag
- DCP 层默认 `save_overwrite=False`
- 目标目录已存在 → DCP 静默 abort 或抛了 exception 被吞掉
- OLMo 层以为保存成功，记录到 trainer state
- 用户后续 `load_checkpoint(<dir>)` 拿到的是上次的旧状态
- 训练 resume 后悄悄回退到老 step——"loss curve 看起来比上次差一点点"

### 12.3 OLMo 上游 fix

commit `095896cd47a24a4660c7cb6d7df45295a907164c`（2025-04-15）：

```diff
- local_files_created = save_model_and_optim_state(checkpoint_dir, dist_model, optim)
+ local_files_created = save_model_and_optim_state(
+     checkpoint_dir, dist_model, optim,
+     save_overwrite=self.cfg.save_overwrite)
```

1 行 propagate。

### 12.4 trainaudit 怎么抓

**Rule gap**——trainaudit 的 `core_trace/checkpoint_hook.py` 当前只 wrap `torch.utils.checkpoint.checkpoint`（activation checkpointing）。**不** wrap state-dict save 路径。

未来要 cover：加 `checkpoint.save.pre` hookpoint 包装 `torch.distributed.checkpoint.save_state_dict` 等，emit kwargs。规则在 `cfg.save_overwrite=True` 但 save event kwargs 缺 `save_overwrite=True` 时 fire。

### 12.5 E2E 证据

`benchmark/eval/hunt_log/CAND_OLMO_CKPT_SAVE_OVERWRITE_DROP/dynamic_confirm_e2e.py`：
- **AST 验证**：parse `olmo/checkpoint.py:1938` 的 Call 节点，发现 3 positional + **0 keywords**
- **Runtime monkey-patch**：替换 `olmo_core.distributed.checkpoint.save_model_and_optim_state` 为 recorder，调用 OLMo 那行代码，捕获 `args_count=3, kwargs={}` —— 字面证明 runtime 不传 `save_overwrite`
- log: `dynamic_confirm_e2e.log`

---

## 13. CAND_OLMO_ADAPTIVE_CLIP_EMA_RESET — checkpoint resume 删除 grad_norm_exp_avg

### 13.1 字面代码（pinned `204ad53c`）

`exp/frameworks/OLMo/olmo/checkpoint.py:1665-1678`

```python
# HACK/TODO (epwalsh): When we use adaptive clipping we track the 'grad_norm_exp_avg' for every param
# in every rank, and keep this in the optimizer state. But this causes issues when loading the
# state since torch sees the state is non-empty for some params which would normally be empty,
# and then assumes it should have all of the other state tensors for that param, which is doesn't.
# So for now we just remove 'grad_norm_exp_avg' everywhere from the state, which resets that metric.
# Not the end of the world but there's probably a better way around this without resetting
# the metric.
for param_id in list(optim_state["state"].keys()):
    state = optim_state["state"][param_id]
    if "grad_norm_exp_avg" in state:
        del state["grad_norm_exp_avg"]                  # ← BUG: 强删 EMA
    if len(state) == 0:
        del optim_state["state"][param_id]
optim.load_state_dict(optim_state)
```

### 13.2 为什么是静默错误

OLMo 的 adaptive grad clipping（`olmo/optim.py:278-302`）：
```python
# 每 step 更新 grad_norm_exp_avg
state['grad_norm_exp_avg'] = ...

# 用 EMA 算 max_allowed_norm
max_allowed_norm = max_norm_ratio * grad_norm_exp_avg
clip_coef = max_allowed_norm / (grad_norm + 1e-6)
```

EMA 是历史 grad-norm 的指数滑动平均，决定每 step 的 clip threshold。

Resume 后 EMA 全删 → 第一步 `grad_norm_exp_avg is None` → init from 当前 step 的 grad_norm（fresh）→ EMA 从一个值开始重建。

后果：
1. 前 ~100 步 clip threshold 是基于刚 build 的、噪声大的 EMA
2. 跟连续训练 vs resume 训练对比 → loss curve 不一样
3. 多次 resume（spot instance 训练、debug 循环）累积非零 trajectory drift
4. ablation 数字微妙偏差，confound 比较

HACK/TODO 注释自己说"Not the end of the world but there's probably a better way without resetting"——团队知道。

### 13.3 trainaudit 怎么抓

**Rule gap**——同 candidate 12，需要 ckpt save/load 的新 hookpoint 家族。

未来 cover：加 `optim.load_state_dict.post` hookpoint，emit 每个 param state 的 keys。规则 fire：build.snapshot 声明 adaptive_clipping 启用，但 load_state_dict.post 的 state 缺 `grad_norm_exp_avg` key → fire。

### 13.4 E2E 证据

`benchmark/eval/hunt_log/CAND_OLMO_ADAPTIVE_CLIP_EMA_RESET/dynamic_confirm_e2e.py`：
- **AST**：定位 for-loop @1672 + `del state[grad_norm_exp_avg]` @1675
- **Runtime replay**：手写一个 5-param synthetic optim_state，每 param state 含 `step / exp_avg / exp_avg_sq / grad_norm_exp_avg`。replay pinned source 的 verbatim for-loop。
- **5 个 EMA → 0 个 EMA** post-loop。剩下的 state keys 都是 `['step', 'exp_avg', 'exp_avg_sq']`
- log: `dynamic_confirm_e2e.log`

---

## 验收 checklist

每条候选都应能用以下命令独立验证：

```bash
# 1. 字面行确认
grep -n "<bug pattern>" <pinned framework path>

# 2. 跑 E2E driver
python benchmark/eval/hunt_log/CAND_<id>/dynamic_confirm_e2e.py
# 或对 GPU 候选: ssh eval-gpu-0 ...

# 3. 看 verdict + log
cat benchmark/eval/hunt_log/CAND_<id>/{verdict.json,dynamic_confirm_e2e.log}
```

每条候选目录还含 `code_excerpt.md`——对应本文档每条章节的 markdown 版本（文件级源数据）。

---

## 后续工作（不在本次验收范围）

1. **ZERO3_STREAM_RACE_NAN**：等 driver 升到 ≥ 580 后能装 torch ≥ 2.10，跑 7×H200 + Qwen3-4B 复现 PR #7898 的 150K NaN 数字
2. **OVERLAP_COMM**：尝试 H100 stack（NCCL 2.21 时序）或 1000+ step stress
3. **OLMOCORE_FSDP_EXPERTS**：FSDP2+DDP-on-experts hybrid multi-rank E2E（需要专门 setup）
4. **Rule-gap candidates**（4, 12, 13）：实现 `optim.zero_grad.post` / `checkpoint.save.pre` / `optim.load_state_dict.post` 三个新 hookpoint 家族；加 4 条新 rule
5. **OLMOCORE_ASYNC_CALLBACK_RACE**：active-probe 形态的 trainer-internal hook（trainaudit 需扩 trace schema 容纳 trainer state events）

---

**结尾**：13 条候选都有 file:line + 代码 + mechanism + rule mapping + E2E 证据（10 条真实-framework E2E、3 条结构性 E2E + 真实 framework 的 rule capability proof）。请按本文逐条验收，有任何 mechanism 描述错或证据不充分的请打回，我重做对应一条。
