# Benchmark Schema

每个 bug 目录 `bugs/<BUG_ID>/` 包含以下文件：

## 必须文件

### `config.json` — Bug 元数据

```json
{
  "bug_id": "M-010",
  "framework": "megatron-lm | olmo | olmo-core | deepspeed",
  "repo": "NVIDIA/Megatron-LM",
  "title": "一行描述",
  "issue_url": "https://github.com/...",
  "buggy_commit": "full SHA",
  "fixed_commit": "full SHA",
  "fix_date": "YYYY-MM-DD",

  "category": "control_flow | dtype | numerical | sharding | config_validation | optimizer_state | communication | data_loading | loss_computation | checkpoint",
  "severity": "high | medium | low",
  "gpu_needed": 2,
  "parallel_dimensions": ["TP", "DP", "PP", "CP", "EP"],

  "trigger_conditions": ["condition1", "condition2"],
  "root_cause": "一段话描述根因",

  "detection_method": "function_call_counting | dtype_invariant | value_invariant | structural_invariant | residual_stream_invariance | config_invariant | cross_rank_equality | source_analysis | value_scaling_check",
  "invariant": "用自然语言描述的不变量",

  "expected_output": {
    "buggy": "BUG DETECTED: ...",
    "fixed": "CLEAN: ... | AssertionError: ... | ValueError: ..."
  },

  "reproduction_status": "reproduced | failed | not_attempted | commit_mismatch | env_missing",
  "failure_reason": "仅 status != reproduced 时填写",
  "notes": "其他备注"
}
```

### `detect.py` — 检测脚本

自包含的 Python 脚本，通过 monkey-patching 检测 bug。

**Megatron bugs**:
```python
# 1. 可选: patch mock-data / 数据加载兼容性
# 2. Hook 目标函数 (train_step, router.forward, etc.)
# 3. exec(open("pretrain_gpt.py").read())
# 4. 输出: "[BUG_ID] BUG DETECTED: ..." 或 "[BUG_ID] CLEAN: ..."
```

**OLMo bugs**:
```python
# 1. sys.path.insert(0, OLMO_DIR)
# 2. 构建小模型，运行检测逻辑
# 3. 输出: "[BUG_ID] BUG DETECTED: ..." 或 "[BUG_ID] CLEAN: ..."
```

### `run.sh` — 训练启动器（Megatron only）

```bash
#!/bin/bash
set -euo pipefail
MEGATRON_DIR="${MEGATRON_DIR:?}"
# torchrun ... detect.py --training-args
```

### `reproduce.sh` — 一键复现

```bash
#!/bin/bash
# 1. checkout buggy → run detect → 验证 BUG DETECTED
# 2. checkout fixed → run detect → 验证 CLEAN
# 3. cleanup
```

## 状态跟踪

`benchmark/status.json` — 全局进度追踪：

```json
{
  "total": 128,
  "reproduced": 10,
  "failed": 3,
  "not_attempted": 115,
  "bugs": {
    "M-010": "reproduced",
    "M-005": "failed",
    ...
  }
}
```
