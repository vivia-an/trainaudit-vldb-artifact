# DP参数一致性错误注入使用指南

## 概述

本指南说明如何使用环境变量控制的错误注入机制，在Megatron-LM训练过程中产生可控的DP参数不一致错误，用于测试和验证SDCCheck约束检测系统。

## 架构设计

### 血缘流程

```
训练脚本 (pretrain_inject_error.sh)
    ↓ 设置环境变量
    ├─ SDCCHECK_INJECT_DP_ERROR (启用/禁用错误注入)
    ├─ SDCCHECK_ERROR_DP_RANK (目标DP rank)
    ├─ SDCCHECK_ERROR_PARAM_PATTERN (参数名匹配模式)
    ├─ SDCCHECK_ERROR_SCALE (扰动幅度)
    └─ SDCCHECK_ERROR_START_STEP (开始注入步数)
    ↓
启动训练 (torchrun + pretrain_gpt.py)
    ↓
Megatron-LM初始化
    ├─ training.py: 训练循环
    │   └─ 设置 optimizer._current_step = iteration
    ↓
每个训练步骤
    ├─ 前向传播
    ├─ 反向传播
    ├─ 梯度同步 (AllReduce in DP group)
    ├─ optimizer.step()
    │   ├─ 更新参数
    │   ├─ _inject_dp_inconsistency_error()  ← 错误注入点
    │   │   ├─ 检查环境变量
    │   │   ├─ 匹配DP rank
    │   │   ├─ 匹配参数名模式
    │   │   └─ 添加随机扰动
    │   └─ 复制main_params到model_params
    ↓
MegatronCollector收集数据
    └─ dump_main_param("model-after-optimizer-step")
    ↓
保存到数据库
    └─ coredump_{dp}_{tp}_{pp}.db
```

### 关键修改点

#### 1. optimizer.py (Megatron-LM)

**位置**: `megatron/core/optimizer/optimizer.py`

**修改内容**:
- 添加 `_inject_dp_inconsistency_error()` 方法
- 在 `step()` 方法末尾调用错误注入

**代码片段**:
```python
def step(self):
    # ... 原有逻辑 ...
    success = self.step_with_ready_grads()
    
    # [SDCCheck] 错误注入
    self._inject_dp_inconsistency_error()
    
    return success, grad_norm, num_zeros_in_grad
```

#### 2. training.py (Megatron-LM)

**位置**: `megatron/training/training.py`

**修改内容**:
- 在训练循环中设置 `optimizer._current_step`

**代码片段**:
```python
# Run training step.
args.curr_iteration = iteration
# [SDCCheck] Set current iteration in optimizer for error injection
if hasattr(optimizer, '_current_step'):
    optimizer._current_step = iteration
```

## 环境变量配置

### 核心环境变量

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `SDCCHECK_INJECT_DP_ERROR` | "0" | 是否启用错误注入（"1"启用，"0"禁用） |
| `SDCCHECK_ERROR_DP_RANK` | "0" | 目标DP rank（在哪个rank注入错误） |
| `SDCCHECK_ERROR_PARAM_PATTERN` | "layers.0" | 参数名匹配模式（只对包含此模式的参数注入） |
| `SDCCHECK_ERROR_SCALE` | "1e-5" | 扰动比例（控制错误大小） |
| `SDCCHECK_ERROR_START_STEP` | "2" | 开始注入的训练步数 |

### VTimeline数据收集变量

| 环境变量 | 说明 |
|---------|------|
| `VTIMELINE_LOGGER_DIR` | 数据库存储目录 |
| `VTIMELINE_DUMP_STEP` | 收集数据的最大步数 |

## 使用步骤

### 1. 准备环境

确保已完成以下准备工作：

```bash
# 1. 安装依赖
pip install duckdb torch

# 2. 确认Megatron-LM路径
cd /path/to/Megatron-LM

# 3. 确认VTimeline Collector已集成
# 检查是否存在 megatron_collector.py
ls -la src/vtimeline/megatron_collector.py
```

### 2. 修改训练脚本

编辑 `config/pretrain_inject_error.sh`，修改以下内容：

```bash
# 修改Megatron-LM路径
cd /path/to/Megatron-LM  # 改为实际路径

# 根据GPU数量调整并行配置
GPUS_PER_NODE=8          # 每个节点的GPU数
TENSOR_MODEL_PARALLEL_SIZE=2   # TP大小
PIPELINE_MODEL_PARALLEL_SIZE=2  # PP大小
# DP大小 = GPUS_PER_NODE / (TP * PP) = 8 / (2*2) = 2

# 调整错误注入参数（可选）
export SDCCHECK_ERROR_SCALE=1e-4  # 增大扰动
export SDCCHECK_ERROR_PARAM_PATTERN="weight"  # 修改匹配模式
```

### 3. 运行错误注入训练

```bash
# 方式1: 直接运行脚本
bash config/pretrain_inject_error.sh

# 方式2: 自定义环境变量
export SDCCHECK_INJECT_DP_ERROR=1
export SDCCHECK_ERROR_DP_RANK=0
export SDCCHECK_ERROR_SCALE=1e-3  # 更大的扰动
bash config/pretrain_inject_error.sh
```

### 4. 验证错误注入

训练完成后，运行验证脚本：

```bash
# 快速验证
python test_dp_consistency_violation.py

# 详细输出
python test_dp_consistency_violation.py --verbose

# 指定数据库路径
python test_dp_consistency_violation.py --db-path /custom/path
```

**预期输出**:
```
==================================================
检查DP一致性约束违反
==================================================
数据库: error_injection_db/Collector/coredump_2_2_2.db

📊 数据库记录总数: 850
📊 优化器更新后记录数: 120

==================================================
✅ 检测到 15 个参数存在DP不一致
==================================================

[1] 参数: decoder.layers.0.mlp.linear_fc1.weight
    步骤: 2
    不同cksum数量: 2/2
    涉及DP ranks: [0, 1]

[2] 参数: decoder.layers.0.mlp.linear_fc2.weight
    步骤: 2
    不同cksum数量: 2/2
    涉及DP ranks: [0, 1]

...

==================================================
✓ 错误注入成功
✓ DP一致性约束违反已被正确记录
==================================================
```

### 5. 运行完整约束检查

使用SDCCheck系统验证约束：

```bash
cd /path/to/sdccheck

python -m sdccheck \
    --db error_injection_db/Collector/coredump_2_2_2.db \
    --constraints config/predefined_constraints.json \
    --check "DP参数跨rank一致性检查"
```

## 高级配置

### 调整错误强度

根据测试需求调整错误幅度：

```bash
# 微小扰动（接近浮点精度）
export SDCCHECK_ERROR_SCALE=1e-6

# 小扰动（默认，适合大多数测试）
export SDCCHECK_ERROR_SCALE=1e-5

# 中等扰动（明显但不会导致训练崩溃）
export SDCCHECK_ERROR_SCALE=1e-4

# 大扰动（可能影响训练稳定性）
export SDCCHECK_ERROR_SCALE=1e-3
```

### 针对特定参数注入

修改参数匹配模式：

```bash
# 只注入第0层
export SDCCHECK_ERROR_PARAM_PATTERN="layers.0"

# 注入所有权重参数
export SDCCHECK_ERROR_PARAM_PATTERN="weight"

# 注入所有MLP层
export SDCCHECK_ERROR_PARAM_PATTERN="mlp"

# 注入所有注意力层
export SDCCHECK_ERROR_PARAM_PATTERN="attention"

# 注入特定参数
export SDCCHECK_ERROR_PARAM_PATTERN="linear_fc1.weight"
```

### 控制注入时机

```bash
# 从第一步开始注入
export SDCCHECK_ERROR_START_STEP=0

# 从第5步开始注入
export SDCCHECK_ERROR_START_STEP=5

# 注意：SDCCHECK_ERROR_START_STEP 必须 <= VTIMELINE_DUMP_STEP
# 否则不会收集到注入错误后的数据
```

### 针对特定DP Rank

```bash
# 在DP rank 0注入（默认）
export SDCCHECK_ERROR_DP_RANK=0

# 在DP rank 1注入
export SDCCHECK_ERROR_DP_RANK=1

# 注意：确保目标rank存在
# 如果DP=2，有效值为0或1
```

## 故障排查

### 问题1: 未检测到DP不一致

**症状**: `test_dp_consistency_violation.py` 显示"未检测到DP不一致"

**可能原因和解决方法**:

1. **错误注入未启用**
   ```bash
   # 检查环境变量
   echo $SDCCHECK_INJECT_DP_ERROR  # 应该是 "1"
   
   # 在训练脚本中明确设置
   export SDCCHECK_INJECT_DP_ERROR=1
   ```

2. **参数名模式不匹配**
   ```bash
   # 查看实际参数名
   python -c "
   import duckdb
   conn = duckdb.connect('error_injection_db/Collector/coredump_2_2_2.db')
   result = conn.execute('''
       SELECT DISTINCT json_extract_string(data, '$.name') as name
       FROM coredump WHERE stage = 'model-after-optimizer-step'
       LIMIT 10
   ''').fetchall()
   for r in result: print(r[0])
   "
   
   # 调整匹配模式
   export SDCCHECK_ERROR_PARAM_PATTERN="实际存在的参数名"
   ```

3. **开始步数设置过晚**
   ```bash
   # 确保开始步数在收集范围内
   export SDCCHECK_ERROR_START_STEP=2
   export VTIMELINE_DUMP_STEP=5  # 必须 >= START_STEP
   ```

4. **代码修改未生效**
   ```bash
   # 检查optimizer.py中是否有错误注入代码
   grep -n "inject_dp_inconsistency_error" \
       /path/to/Megatron-LM/megatron/core/optimizer/optimizer.py
   
   # 应该看到两处：
   # 1. 方法定义（约534行）
   # 2. 方法调用（约565行）
   ```

### 问题2: 数据库为空

**症状**: 数据库记录总数为0

**解决方法**:

1. **检查VTimeline Collector集成**
   ```bash
   # 确认Collector已初始化
   grep -r "MegatronCollector" /path/to/Megatron-LM/
   ```

2. **检查环境变量**
   ```bash
   echo $VTIMELINE_LOGGER_DIR  # 应该有值
   echo $VTIMELINE_DUMP_STEP   # 应该 > 0
   ```

3. **检查训练是否成功运行**
   ```bash
   # 查看训练日志
   # 应该看到"[SDCCheck] Injected DP inconsistency errors"
   ```

### 问题3: 训练崩溃

**症状**: 训练过程中出现NaN或crash

**解决方法**:

1. **减小扰动幅度**
   ```bash
   export SDCCHECK_ERROR_SCALE=1e-6  # 使用更小的值
   ```

2. **延后注入时机**
   ```bash
   export SDCCHECK_ERROR_START_STEP=5  # 等训练稳定后再注入
   ```

3. **限制注入范围**
   ```bash
   # 只注入非关键参数
   export SDCCHECK_ERROR_PARAM_PATTERN="layers.0.mlp"
   ```

## 测试用例

### 测试用例1: 基本错误注入

**目标**: 验证基本的错误注入和检测功能

**配置**:
```bash
export SDCCHECK_INJECT_DP_ERROR=1
export SDCCHECK_ERROR_DP_RANK=0
export SDCCHECK_ERROR_PARAM_PATTERN="layers.0"
export SDCCHECK_ERROR_SCALE=1e-5
export SDCCHECK_ERROR_START_STEP=2
```

**预期结果**: 检测到多个包含"layers.0"的参数不一致

### 测试用例2: 不同错误强度

**目标**: 测试不同错误强度对检测的影响

**配置**:
```bash
# 测试1: 微小扰动
export SDCCHECK_ERROR_SCALE=1e-7
bash config/pretrain_inject_error.sh

# 测试2: 大扰动
export SDCCHECK_ERROR_SCALE=1e-3
bash config/pretrain_inject_error.sh
```

**预期结果**: 两种情况都应该被检测到

### 测试用例3: 多个DP Rank

**目标**: 验证在不同DP rank注入错误

**配置**:
```bash
# 在DP rank 0注入
export SDCCHECK_ERROR_DP_RANK=0
bash config/pretrain_inject_error.sh
mv error_injection_db error_injection_db_rank0

# 在DP rank 1注入
export SDCCHECK_ERROR_DP_RANK=1
bash config/pretrain_inject_error.sh
mv error_injection_db error_injection_db_rank1

# 分别验证
python test_dp_consistency_violation.py --db-path error_injection_db_rank0
python test_dp_consistency_violation.py --db-path error_injection_db_rank1
```

**预期结果**: 两种情况都应该检测到不一致

## 参考资料

### 相关文件

- `megatron/core/optimizer/optimizer.py` - 错误注入实现
- `megatron/training/training.py` - 步数追踪
- `config/pretrain_inject_error.sh` - 训练脚本
- `test_dp_consistency_violation.py` - 验证脚本
- `config/predefined_constraints.json` - 约束定义

### 约束定义

```json
{
  "DP参数跨rank一致性检查": {
    "name": "优化器更新后DP参数跨rank一致性检查",
    "description": "优化器更新后,同一参数矩阵在不同DP rank间的权重和梯度校验是一致的",
    "type": "consistency",
    "applicable_conditions": {
      "dp": "> 1",
      "stage": "= 'model-after-optimizer-step'"
    }
  }
}
```

### SQL查询示例

检查参数不一致：
```sql
WITH param_stats AS (
    SELECT 
        json_extract_string(data, '$.name') as param_name,
        json_extract_string(data, '$.dp') as dp_rank,
        json_extract_string(data, '$.cksum') as cksum,
        step
    FROM coredump
    WHERE stage = 'model-after-optimizer-step'
)
SELECT 
    param_name,
    step,
    COUNT(DISTINCT cksum) as distinct_cksums,
    array_agg(DISTINCT dp_rank) as dp_ranks
FROM param_stats
GROUP BY param_name, step
HAVING COUNT(DISTINCT cksum) > 1;
```

## 总结

通过环境变量控制的错误注入机制，我们可以：

1. ✅ 在不修改训练逻辑的情况下注入可控错误
2. ✅ 灵活调整错误类型、强度、位置和时机
3. ✅ 验证约束检测系统的有效性
4. ✅ 支持多种测试场景和用例

这种设计遵循了 Megatron-LM 的环境变量参数覆盖模式，实现了配置与代码的解耦，便于测试和调试。




















