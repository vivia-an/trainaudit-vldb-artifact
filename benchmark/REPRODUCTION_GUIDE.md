# Bug 复现完整指南

> 目标：从 Megatron-LM、DeepSpeed、OLMo/OLMo-core 三个框架中端到端复现 100 个静默错误

## 一、用户核心要求（绝对不可违反）

1. **必须端到端复现**：运行真实框架代码（模型构建 → forward → backward → optimizer.step → 检查运行时值）
2. **必须在 GPU 上跑**：所有 bug 复现都要走 `torchrun` + CUDA，禁止 CPU-only 或仅 forward / 仅构造 optimizer 的"伪复现"
3. **绝对禁止 `inspect.getsource()`**：不能通过读源码来做检测，必须观察运行时行为
4. **不能 mock 实现**：不能自己重写 bug 逻辑然后检测，必须调用框架真实函数
5. **可开源格式**：每个 bug 必须有 `config.json` + `detect.py` + `run.sh` + `reproduce.sh`
6. **检测信号必须来自运行时值**：tensor.dtype, tensor.shape, 函数返回值, 参数值, 属性存在性等

## 二、基础设施

### 机器配置

| 机器（远端 hostname） | 用途 | SSH 别名 | HostName : Port |
|------|------|---------|-----------------|
| beijing-dev-0（当前容器） | 开发、搜索、CPU 测试 | 本地（172.16.63.170） | — |
| eval-gpu-0 | Megatron / DeepSpeed GPU 测试 (8x H200) | `eval-gpu-0` 或 `beijing-dev-gpu` | `183.242.150.6:32054` |
| olmo-gpu-0 | OLMo / OLMo-core GPU 测试 (8x H200) | `beijing-olmo-gpu` | `183.242.150.6:32462` |

> 验证（2026-05-05）：两台 GPU 机器均通，`torch.cuda.is_available()=True`，各 8 张 H200。

#### SSH 凭证（在 beijing-dev-0 上的配置）

- 私钥：`/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025/tmp/ubi`（对应远端 root 账号）
- 已写入 `/root/.ssh/config`，可直接 `ssh eval-gpu-0` / `ssh beijing-olmo-gpu`
- `beijing-dev-gpu` 别名带 `RemoteForward 7881 127.0.0.1:7882`，远端 7881 已被占用会报 *"remote port forwarding failed"*，**无害**，不需要这个 forward 时直接用 `eval-gpu-0` 别名即可
- 公钥指纹：`AAAAC3NzaC1lZDI1NTE5...FxeY1k 1454491905@qq.com`（Mac 上对应 `~/.ssh/ubi`）

### 关键路径

```
项目根目录: /volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025/
仓库:
  Megatron-LM: exp/frameworks/Megatron-LM/
  DeepSpeed:   exp/frameworks/DeepSpeed/
  OLMo:        exp/frameworks/OLMo/
  OLMo-core:   exp/frameworks/OLMo-core/
Benchmark:
  bugs/:       benchmark/bugs/
  tools/:      benchmark/tools/
  status:      benchmark/status.json
```

### eval-gpu-0 SSH 注意事项

- **必须用 `bash -l`**：`ssh eval-gpu-0 "bash -l -c 'command'"`（否则 CUDA 不可用）
- **必须 cd 到仓库目录**：默认目录是 `/root`，不是 git 仓库
- **共享文件系统**：`/volume/qscai/` 在 beijing-dev-0 / eval-gpu-0 / olmo-gpu-0 三台之间共享（GPFS 挂载点 `09b40f91`），所以临时脚本可以直接写到 `/volume/qscai/cqs/temp/` 给远端读
- **依赖安装**：GPU 机器无网络，需先在 beijing-dev-0 `pip download` 到 `/volume/qscai/cqs/temp/` 再离线安装
- **快速测试连通**：`ssh eval-gpu-0 "nvidia-smi -L"` 应列出 8 张 H200

### GPU 机器已安装的额外依赖

- py-cpuinfo, hjson, mup（用于特定 commit 的兼容性）

## 三、已验证的高效复现模式

### 模式 A：Megatron MoE Hook（成功率 ~90%）

```python
# detect.py 模板
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe import moe_utils

# 1. Hook 目标函数
_orig = TopKRouter.method
_results = []
def _hooked(self, *args, **kwargs):
    if not _checked[0]:
        _results.append({...})  # 捕获运行时值
    return _orig(self, *args, **kwargs)
TopKRouter.method = _hooked

# 2. Hook train_step 来报告结果
_ts_mod = __import__("megatron.training.training", fromlist=["train_step"])
_orig_ts = _ts_mod.train_step
def _patched_ts(*args, **kwargs):
    result = _orig_ts(*args, **kwargs)
    if not _reported[0] and _results:
        # 输出检测结果
        pass
    return result
_ts_mod.train_step = _patched_ts

# 3. 启动训练
exec(open("pretrain_gpt.py").read())
```

```bash
# run.sh 模板
MEGATRON_DIR="${MEGATRON_DIR:?}"
cd "$MEGATRON_DIR"
torchrun --nproc_per_node=2 --master_port=${MASTER_PORT:-29500} \
    "$SCRIPT_DIR/detect.py" \
    --num-layers 2 --hidden-size 128 --num-attention-heads 4 \
    --seq-length 64 --max-position-embeddings 64 \
    --micro-batch-size 2 --global-batch-size 4 \
    --train-iters 3 --lr 1e-4 --min-lr 1e-5 \
    --bf16 --no-save-optim --no-save-rng \
    --tokenizer-type NullTokenizer --vocab-size 50304 --mock-data \
    --swiglu --disable-bias-linear \
    --num-experts 2 --moe-aux-loss-coeff 0.1 \
    [额外 MoE 参数]
```

**典型检测点**：
- `save_to_aux_losses_tracker` 是否传了 `avg_group` / `reduce_group`
- `TopKRouter.__init__` 是否设了某属性
- 函数入参的 dtype/shape
- 返回值的数值范围

### 模式 B：DeepSpeed Engine Hook（成功率 ~70%）

```python
# detect.py 模板
import deepspeed
from deepspeed_train_wrapper import SimpleGPT  # 在 tools/ 下

# Hook 目标
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
_orig = DeepSpeedZeroOptimizer.method
def _hooked(self, *args, **kwargs):
    # 捕获运行时行为
    return _orig(self, *args, **kwargs)
DeepSpeedZeroOptimizer.method = _hooked

def main():
    deepspeed.init_distributed()
    model = SimpleGPT(vocab_size=1024, d_model=64, n_heads=4, n_layers=1)
    ds_config = {
        "train_batch_size": 2 * world_size,
        "train_micro_batch_size_per_gpu": 2,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
        "zero_optimization": {"stage": 1},
        "bf16": {"enabled": True},
    }
    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    for step in range(3):
        loss, _ = engine(input_ids, labels=input_ids.clone())
        engine.backward(loss)
        engine.step()
    # 报告结果
```

**注意**：
- `train_batch_size` 必须 = `micro_batch * ga_steps * world_size`
- 旧 commit (< 2025-03) 可能缺 `cpuinfo`/`hjson` 依赖
- BF16_Optimizer 需要 `"data_types": {"grad_accum_dtype": "fp32"}` 且 model 要 `.to(torch.bfloat16)`

### 模式 C：OLMo 端到端 GPU 训练 Hook

```python
# detect.py 模板（OLMo 真训练，跑在 olmo-gpu-0）
import os, torch, torch.distributed as dist
try:
    from olmo.model import OLMo as Olmo
except ImportError:
    from olmo.model import Olmo
from olmo.config import ModelConfig

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# 1. 在模型构造前 monkey-patch 目标函数
import olmo.model as _om
_orig = _om.TargetClass.method
_results = []
def _hooked(self, *args, **kwargs):
    out = _orig(self, *args, **kwargs)
    _results.append({...})  # 捕获运行时值
    return out
_om.TargetClass.method = _hooked

# 2. 构造小模型 + 真实 train step（必须 forward + backward + optimizer.step）
config = ModelConfig(d_model=128, n_heads=4, n_layers=1,
                    vocab_size=1024, embedding_size=1024)
model = Olmo(config).to(device)
model.train()
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

input_ids = torch.randint(0, 1024, (2, 16), device=device)
for step in range(3):
    optim.zero_grad()
    output = model(input_ids)
    loss = output.logits.float().sum()
    loss.backward()
    optim.step()

# 3. rank 0 报告检测结果
if dist.get_rank() == 0:
    print("DETECTED" if _results and <condition> else "NOT_DETECTED")
dist.destroy_process_group()
```

```bash
# run.sh 模板
OLMO_DIR="${OLMO_DIR:?}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$OLMO_DIR"
export PYTHONPATH="$OLMO_DIR:${PYTHONPATH:-}"
torchrun --nproc_per_node=1 --master_port=${MASTER_PORT:-29500} \
    "$SCRIPT_DIR/detect.py"
```

**注意**：
- 走 `ssh beijing-olmo-gpu` 执行，不要在 beijing-dev-0 上跑 CPU 版本
- OLMo 类名在不同版本可能是 `Olmo` 或 `OLMo`，用 try/except
- 需要 `pip install mup`（某些 commit）
- `block_type="sequential"` 用于测试 OLMoSequentialBlock
- backward 必须真实跑通；如果目标 bug 只在 forward 路径出现，仍要保留 backward 来证明端到端不爆炸

### 模式 D：OLMo-core 端到端 GPU 训练步

```python
# detect.py 模板（OLMo-core 真训练，跑在 olmo-gpu-0）
import os, torch, torch.distributed as dist
import torch.nn as nn
from olmo_core.optim.adamw import SkipStepAdamWConfig

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# Hook 目标
from olmo_core.optim.adamw import SkipStepAdamW
_orig_step = SkipStepAdamW.step
_states = []
def _hooked_step(self, *args, **kwargs):
    out = _orig_step(self, *args, **kwargs)
    _states.append({id(p): dict(self.state[p]) for g in self.param_groups for p in g["params"]})
    return out
SkipStepAdamW.step = _hooked_step

# 真实小模型（必须有可训练参数 + 真 backward）
model = nn.Sequential(nn.Embedding(1024, 128), nn.Linear(128, 1024)).to(device)
optim = SkipStepAdamWConfig(lr=1e-3).build([{"params": list(model.parameters())}])

input_ids = torch.randint(0, 1024, (2, 16), device=device)
for step in range(5):
    optim.zero_grad()
    logits = model(input_ids)
    loss = logits.float().sum()
    loss.backward()
    optim.step()

# 检查 optimizer.state[p]['step'] / scheduler 输出 / hook 捕获值
if dist.get_rank() == 0:
    print("DETECTED" if <condition over _states> else "NOT_DETECTED")
dist.destroy_process_group()
```

```bash
# run.sh 模板
OLMO_CORE_DIR="${OLMO_CORE_DIR:?}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$OLMO_CORE_DIR"
export PYTHONPATH="$OLMO_CORE_DIR/src:${PYTHONPATH:-}"
torchrun --nproc_per_node=1 --master_port=${MASTER_PORT:-29500} \
    "$SCRIPT_DIR/detect.py"
```

**注意**：
- 同样走 `ssh beijing-olmo-gpu`
- 不再允许"只构造 optimizer/scheduler 跑空 step"——必须挂在真模型上
- 如检测目标在 scheduler 公式上，可在真训练循环里取 `lr_scheduler.get_last_lr()` / 调度内部值，不要脱离训练单独调用

## 四、常见陷阱和教训

### 4.1 必须避免的错误

| 错误 | 说明 |
|------|------|
| `inspect.getsource()` | 绝对禁止，用户会拒绝 |
| 模拟 bug 逻辑 | 不能自己实现 buggy 公式然后检测，必须调用框架函数 |
| 只在一个 commit 上测试 | 必须 buggy 和 fixed 都测，确认差异 |
| `set -euo pipefail` + `&&` | reproduce.sh 中用 `set +eo pipefail` 包裹 run 部分 |
| 忘记 `tools/__init__.py` | Megatron 旧 commit 的 `git checkout` 会因为这个文件冲突，需要先 `rm -f` |

### 4.2 检测逻辑常见错误

| 问题 | 正确做法 |
|------|---------|
| Hook 了 JIT-compiled 函数 | 需要 hook 底层的 `_bias_dropout_add_func` 而非 fused 版本 |
| 函数返回 tuple 但按 tensor 处理 | 先检查 `isinstance(result, tuple)` |
| 旧 commit 缺少新参数 | 用 `hasattr` 检查或 `getattr(..., default)` |
| 两个 commit 检测结果相同 | 说明检测逻辑错误——可能 hook 了不受 fix 影响的路径 |

### 4.3 旧 commit 兼容性问题

| 时间范围 | 问题 | 解决方案 |
|----------|------|---------|
| < 2024-06 | 缺 `cpuinfo`/`hjson` | 离线安装到 GPU 机器 |
| < 2024-03 | DeepSpeed API 大变动 | 跳过，找更新的 commit |
| Megatron TE 依赖 | `--fp32-residual-connection` 与 TE 冲突 | 避免这个 flag 或关掉 TE |
| `--mock-data` 不存在 | 用 `gen_fake_data.py` 生成 | 仅针对非常旧的 commit |
| `--softmax-scale` 等新参数 | 旧 commit 不支持 | 跳过或换检测方式 |

## 五、搜索新 Bug 的高效策略

### 5.1 最高效搜索命令

```bash
# 搜索特定文件的小改动 fix commit
git log --all --oneline -5000 -- <file> | while read hash msg; do
  stats=$(git diff --shortstat "$hash~1..$hash" -- <dir> 2>/dev/null)
  total=$((${ins:-0} + ${del:-0}))
  if [ "$total" -gt 1 ] && [ "$total" -le 15 ]; then
    echo "($total) $hash $msg"
  fi
done | grep -iE "fix|bug"
```

### 5.2 最值得搜索的文件

| 框架 | 高价值文件 |
|------|-----------|
| Megatron | `megatron/core/transformer/moe/router.py`, `moe_utils.py`, `moe_layer.py` |
| DeepSpeed | `deepspeed/runtime/zero/stage_1_and_2.py`, `engine.py`, `lr_schedules.py`, `bf16_optimizer.py` |
| OLMo | `olmo/model.py`, `olmo/train.py`, `olmo/optim.py` |
| OLMo-core | `src/olmo_core/optim/scheduler.py`, `adamw.py`, `nn/attention/`, `data/composable/` |

### 5.3 已用完的 commit（不要重复搜索）

Megatron: `5f668c10e, 90fcb538a, e6d56d6828, a58768725f, 5153efea0b, 99f999a46, 20b395424d, 87d9d2506, a0177b681, c125e98c7, 5c8eb08f6`

DeepSpeed: `5999fb06, d56e847b, 853c9389, 3fd762cf, 116dbe28, 22cf1a44, 1f706621, b4513f63, 1752c2ab, 0bf92ccd, aeb10bb1, 09885efe`

OLMo: `c482df74, 204ad53c, 8472d0b4, 6c4b8e15, cebdbe53, 2f706197, f81904f3, 167a7ac8, 67c9e315, a57f3803, c205912b`

OLMo-core: `2b6cf996, f34e7ddc, 2504cc28, b9d161bf, 6ce62ccd`

## 六、已复现 37 个 Bug 完整列表

### Megatron-LM (10 个)

| ID | Commit | 检测方式 | Bug 描述 |
|----|--------|---------|---------|
| M-010 | e6d56d6828 | hook save_to_aux_losses_tracker 调用次数 | MoE aux_loss 被 activation checkpointing 累计 2x |
| M-012 | a58768725f | hook TopKRouter.forward 检查 dtype | expert_bias 是 bf16 而非 fp32 |
| M-014 | 5153efea0b | hook routing 检查 probs 值 | topk=1 + post-softmax → probs 全为 1.0 |
| M-020 | 99f999a46 | hook all_gather 检查 layer 数 | PP 静默丢层（配置 5 层实际 4 层）|
| M-024 | 20b395424d | hook apply_input_jitter 检查 dtype | input_jitter 在 fp32 而非 bf16 计算 |
| M-033 | 90fcb538a | hook MoEAuxLossAutoScaler 比较值 | aux_loss gradient 少除了 dp_size |
| M-NEW-1 | 5f668c10e | hook sigmoid 调用检查 dtype | sigmoid 在 bf16 计算（应 fp32）|
| M-NEW-4 | c125e98c7 | 拦截 .clone() 调用检查是否存在 | fp32 logits 缺 clone → inplace 破坏 |
| M-NEW-5 | 87d9d2506 | 检查 Router 属性是否存在 | 缺 calculate_per_token_loss 属性 |
| M-NEW-7 | a0177b681 | hook save_to_aux_losses_tracker 检查 kwargs | z_loss 缺 avg_group 参数 |

### DeepSpeed (12 个)

| ID | Commit | 检测方式 | Bug 描述 |
|----|--------|---------|---------|
| D-011 | 5999fb06 | hook 函数调用顺序 | BF16 grad leak: allreduce→epilogue 顺序错 |
| D-015 | d56e847b | 检查参数是否更新 | FP16 overflow return 位置错误 → 参数不更新 |
| D-027 | b4513f63 | 检查 softmax 输出 | softmax 用了错误 dim → 专家分数不为 1 |
| D-038 | 1f706621 | 注入 inf + hook grad norm | logical_or 产生 NaN grad norm |
| D-NEW-1 | 853c9389 | 检查 gate 值范围 | topk 在 logits 而非 softmax 上 |
| D-NEW-2 | 3fd762cf | 检查 get_lr() 返回长度 | 多 group 只返回 1 个 LR |
| D-NEW-3 | 116dbe28 | 检查 frozen params 数 | requires_grad=False 的参数被加入 optimizer |
| D-NEW-8 | 22cf1a44 | 比较 LR 值 | WarmupLR hardcode 0.001 忽略 optimizer LR |
| D-NEW-15 | 1752c2ab | hook zero_grad 调用次数 | BF16+ZeRO-0 梯度不清零 |
| D-NEW-17 | 0bf92ccd | hook scale_if_loss 输入值 | loss 被双重缩放（gas_scaled） |
| D-NEW-18 | aeb10bb1 | 比较参数与 reference 的偏差 | CPU-offload 多 backward 梯度丢失 |
| D-NEW-20 | 09885efe | 检查 optimizer 类型 | BF16_Optimizer 被错误选择 |

### OLMo (9 个)

| ID | Commit | 检测方式 | Bug 描述 |
|----|--------|---------|---------|
| O-002 | c482df74 | 比较残差与 normed 距离 | attention norm 修改了 residual stream |
| O-005 | 204ad53c | 检查 preserve_rng_state 值 | dropout 时 rng_state 未保存 |
| O-009 | 8472d0b4 | 构建模型检查 shape | k_norm 计算错误导致 shape mismatch |
| O-NEW-1 | 67c9e315 | 检查 RMSNorm 输出 rms | L2 norm 而非 RMS → 输出缩小 16x |
| O-NEW-2 | 6c4b8e15 | mock attention 检查方向 | causal mask 反转 → 未来 token 可见 |
| O-NEW-4 | cebdbe53 | hook RoPE 检查 dtype | pos_sin=fp32 vs q=bf16 不匹配 |
| O-NEW-6 | 2f706197 | mock flash_attn 检查 shape | (B,H,S,D) 而非 (B,S,H,D) layout |
| O-NEW-8 | f81904f3 | 零化权重后检查输出值 | attn_norm 覆盖了 residual input |
| O-NEW-9 | 167a7ac8 | hook memmap 加载检查值 | token ID 被 %2^16 截断 |

### OLMo-core (6 个)

| ID | Commit | 检测方式 | Bug 描述 |
|----|--------|---------|---------|
| O-010 | (olmo-core) | 运行 optimizer step 检查 step 值 | optimizer step counter 不递增 |
| O-016 | (olmo-core) | 检查 param_group 属性 | initial_lr 缺失 |
| O-023 | (olmo-core) | 调用 F.pad 检查输出 dtype | int tensor pad 用了 float 值 |
| OC-NEW-1 | b9d161bf | 调用 as_tensor 检查输出 | float probs 被截断为 int |
| OC-NEW-2 | 2b6cf996 | 运行 SkipStepAdamW 检查 state | step counter 被注释掉 |
| OC-NEW-3 | f34e7ddc | hook _sqrt_decay 比较值 | sqrt decay 公式方向错误 |

## 七、待测试的准备好的脚本

| ID | 说明 | 需要 |
|----|------|------|
| M-NEW-9 | aux_loss metric TP scaling 错误 | GPU (eval-gpu-0) |
| D-NEW-19 | extra_large_param 用错 dtype key | GPU, 需要小 bucket_size |

## 八、下一步优先级

1. **等 GPU 恢复后**：批量测试 M-NEW-9 和其他 Megatron MoE commit
2. **搜索更多 DeepSpeed 2025+ 的 fix**（兼容当前 torch）
3. **OLMo-core 的 train_module/ 和 nn/ 目录**中的 fix
4. **升级 13 个 Tier 3 bug** 为真正的运行时检测
