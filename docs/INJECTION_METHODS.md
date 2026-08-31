# Megatron-LM 错误注入方法详解

## 📍 注入位置

**文件**: `Megatron-LM/megatron/core/pipeline_parallel/schedules.py`  
**函数**: `backward_step()`  
**行号**: 第 436-598 行  
**时机**: Backward Pass 完成后，MegatronCollector.dump_model() 之前

## 🔄 代码血缘流程

```
训练脚本启动
    ↓
设置环境变量
    ├─ MEGATRON_INJECT_PARAM_CORRUPTION=1  (启用注入)
    ├─ MEGATRON_CORRUPT_DP_RANK=0          (目标rank)
    ├─ MEGATRON_CORRUPT_PARAM_SUBSTR="..."  (参数匹配)
    ├─ MEGATRON_CORRUPT_DELTA=1e-5         (扰动幅度)
    └─ MEGATRON_CORRUPT_OP="add"           (操作类型)
    ↓
torchrun pretrain_gpt.py
    ↓
训练循环开始
    ├─ Step N
    ├─ Forward Pass
    │   └─ MegatronCollector.dump_model("after-forward")
    ├─ Backward Pass
    │   ├─ 计算梯度
    │   ├─ 梯度同步 (AllReduce in DP group)
    │   └─ backward_step() 返回
    ↓
🎯 注入点：schedules.py::backward_step() line 436
    ├─ [📍 INJECT] 系统启动检查
    ├─ [⚙️  CONFIG] 读取环境变量配置
    ├─ [🏷️  RANK] 获取当前进程的DP/TP/PP rank
    ├─ [✅ MATCH] 检查是否匹配目标rank
    ├─ [🔍 SEARCH] 搜索模型参数，找到第一个匹配的参数
    ├─ [🎯 FOUND] 找到目标参数
    ├─ [📊 BEFORE] 记录注入前参数统计（均值、标准差、最小值、最大值）
    ├─ [💉 INJECT] 执行注入操作
    │   └─ 根据 MEGATRON_CORRUPT_OP 选择操作类型
    ├─ [📊 AFTER] 记录注入后参数统计
    ├─ [📈 DELTA] 计算参数变化量
    └─ [✅ SUCCESS] 注入完成
    ↓
数据收集
    ├─ MegatronCollector.dump_model("model-after-backward")  ← 捕获注入后的状态
    └─ MegatronCollector.dump_main_param("main-param-after-backward")
    ↓
保存到数据库
    └─ coredump_dp{dp}_tp{tp}_pp{pp}_cp{cp}.db
```

## 🛠️ 支持的注入方式

### 1. **add** - 加法扰动（默认）

**操作**: `param += delta`

**用途**: 向参数添加一个小的偏移量，模拟累积误差或精度损失

**环境变量**:
```bash
export MEGATRON_CORRUPT_OP="add"
export MEGATRON_CORRUPT_DELTA=1e-5
```

**日志输出**:
```
[💉 INJECT]   操作: param += 1e-05
[📊 BEFORE]   - 均值: 1.234567e-03
[📊 AFTER]    - 均值: 1.235567e-03
[📈 DELTA]    - 均值变化: 1.000000e-05 (0.8115%)
```

**适用场景**:
- ✅ 测试小幅度参数偏移
- ✅ 模拟数值累积误差
- ✅ 验证DP一致性检测

---

### 2. **scale** - 缩放扰动

**操作**: `param *= (1 + delta)`

**用途**: 按比例缩放参数值，模拟缩放因子错误或单位转换错误

**环境变量**:
```bash
export MEGATRON_CORRUPT_OP="scale"
export MEGATRON_CORRUPT_DELTA=0.01  # 1% 缩放
```

**日志输出**:
```
[💉 INJECT]   操作: param *= 1.01
[📊 BEFORE]   - 均值: 1.000000e-02
[📊 AFTER]    - 均值: 1.010000e-02
[📈 DELTA]    - 均值变化: 1.000000e-04 (1.0000%)
```

**适用场景**:
- ✅ 测试比例性错误
- ✅ 模拟学习率或权重衰减错误
- ✅ 验证相对误差检测

---

### 3. **zero** - 清零操作

**操作**: `param = 0`

**用途**: 将参数完全清零，模拟参数丢失或初始化错误

**环境变量**:
```bash
export MEGATRON_CORRUPT_OP="zero"
# delta 参数在此模式下被忽略
```

**日志输出**:
```
[💉 INJECT]   操作: param = 0 (清零)
[📊 BEFORE]   - 均值: 1.234567e-03
[📊 AFTER]    - 均值: 0.000000e+00
```

**适用场景**:
- ✅ 测试极端错误情况
- ⚠️ 可能导致训练崩溃或 NaN
- ✅ 验证约束检测的鲁棒性

---

### 4. **noise** - 随机噪声（新增）

**操作**: `param += uniform(-delta, delta)`

**用途**: 向参数添加均匀分布的随机噪声，模拟随机硬件错误或通信噪声

**环境变量**:
```bash
export MEGATRON_CORRUPT_OP="noise"
export MEGATRON_CORRUPT_DELTA=1e-5  # 噪声范围: [-1e-5, 1e-5]
```

**日志输出**:
```
[💉 INJECT]   操作: param += uniform(-1e-05, 1e-05)
[📊 BEFORE]   - 均值: 1.234567e-03
[📊 AFTER]    - 均值: 1.234789e-03  (随机变化)
```

**适用场景**:
- ✅ 模拟硬件错误（位翻转、ECC错误）
- ✅ 模拟通信噪声
- ✅ 测试对随机扰动的鲁棒性

---

### 5. **flip** - 符号翻转（新增）

**操作**: `param *= -1`

**用途**: 翻转参数的符号，模拟符号位错误或梯度方向错误

**环境变量**:
```bash
export MEGATRON_CORRUPT_OP="flip"
# delta 参数在此模式下被忽略
```

**日志输出**:
```
[💉 INJECT]   操作: param *= -1 (符号翻转)
[📊 BEFORE]   - 均值: 1.234567e-03
[📊 AFTER]    - 均值: -1.234567e-03
```

**适用场景**:
- ✅ 测试符号位错误
- ✅ 模拟梯度计算错误
- ⚠️ 可能导致训练发散

---

### 6. **nan** - 设置NaN（新增）

**操作**: `param[:10] = NaN`（仅前10个元素）

**用途**: 在参数中注入 NaN 值，模拟数值计算溢出或除零错误

**环境变量**:
```bash
export MEGATRON_CORRUPT_OP="nan"
# delta 参数在此模式下被忽略
```

**日志输出**:
```
[💉 INJECT]   操作: param[:10] = NaN
[📊 BEFORE]   - 均值: 1.234567e-03
[📊 AFTER]    - 均值: nan
```

**适用场景**:
- ✅ 测试 NaN 检测机制
- ✅ 模拟数值计算错误
- ⚠️ 通常会导致训练立即失败

---

## 📊 日志输出示例

运行时会看到详细的血缘日志：

```
======================================================================
[📍 INJECT] 错误注入系统已启动
[📍 INJECT] 时间: 2025-10-16 12:34:56.789
[📍 INJECT] 位置: schedules.py::backward_step() @ line 436
[📍 INJECT] 阶段: Backward Pass 完成后, Dump 之前
======================================================================
[⚙️  CONFIG] 注入配置:
[⚙️  CONFIG]   - 目标DP Rank: 0
[⚙️  CONFIG]   - 参数匹配模式: layers.0
[⚙️  CONFIG]   - 扰动幅度: 1e-05
[⚙️  CONFIG]   - 操作类型: add
[⚙️  CONFIG]   - 当前训练步: Step 1
[🏷️  RANK] 当前进程信息:
[🏷️  RANK]   - DP Rank: 0
[🏷️  RANK]   - TP Rank: 0
[🏷️  RANK]   - PP Rank: 0
[✅ MATCH] 当前DP rank (0) 匹配目标rank (0)
[✅ MODEL] 模型已加载，共 1 个模型
[🔍 SEARCH] 正在搜索匹配的参数...
[🎯 FOUND] 找到匹配参数: module.decoder.layers.0.mlp.linear_fc1.weight
[🎯 FOUND]   - 形状: [128, 320]
[🎯 FOUND]   - 数据类型: torch.bfloat16
[🎯 FOUND]   - 设备: cuda:0
[🎯 FOUND]   - 是否需要梯度: True
[📊 BEFORE] 注入前参数统计:
[📊 BEFORE]   - 均值: 1.234567e-03
[📊 BEFORE]   - 标准差: 8.765432e-03
[📊 BEFORE]   - 最小值: -2.345678e-02
[📊 BEFORE]   - 最大值: 2.456789e-02
[💉 INJECT] 正在执行注入操作: add
[💉 INJECT]   操作: param += 1e-05
[📊 AFTER] 注入后参数统计:
[📊 AFTER]   - 均值: 1.235567e-03
[📊 AFTER]   - 标准差: 8.765432e-03
[📊 AFTER]   - 最小值: -2.344678e-02
[📊 AFTER]   - 最大值: 2.457789e-02
[📈 DELTA] 参数变化:
[📈 DELTA]   - 均值变化: 1.000000e-05 (0.8115%)
[✅ SUCCESS] 参数注入完成!
[✅ SUCCESS]   - 参数名: module.decoder.layers.0.mlp.linear_fc1.weight
[✅ SUCCESS]   - DP Rank: 0
[✅ SUCCESS]   - 操作: add
[✅ SUCCESS]   - 扰动幅度: 1e-05
[✅ SUCCESS]   - Step: 1
======================================================================
```

## 🎯 使用示例

### 示例1：基本加法扰动

```bash
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_PARAM_SUBSTR="layers.0"
export MEGATRON_CORRUPT_DELTA=1e-5
export MEGATRON_CORRUPT_OP="add"

bash pretrain_inject.sh
```

### 示例2：随机噪声注入

```bash
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_PARAM_SUBSTR="attention"
export MEGATRON_CORRUPT_DELTA=5e-6
export MEGATRON_CORRUPT_OP="noise"

bash pretrain_inject.sh
```

### 示例3：缩放扰动

```bash
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_DP_RANK=1
export MEGATRON_CORRUPT_PARAM_SUBSTR="mlp.linear_fc1.weight"
export MEGATRON_CORRUPT_DELTA=0.001  # 0.1% 缩放
export MEGATRON_CORRUPT_OP="scale"

bash pretrain_inject.sh
```

### 示例4：极端测试 - 清零

```bash
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_PARAM_SUBSTR="layers.11.mlp.linear_fc2.weight"
export MEGATRON_CORRUPT_OP="zero"

bash pretrain_inject.sh
```

## 🔧 高级配置

### 针对不同参数类型的建议

| 参数类型 | 推荐操作 | 推荐 delta | 原因 |
|---------|---------|-----------|------|
| **权重矩阵** | add, noise | 1e-5 ~ 1e-4 | 较稳定，不易导致崩溃 |
| **偏置项** | add, scale | 1e-6 ~ 1e-5 | 偏置通常较小，需要更小扰动 |
| **LayerNorm** | scale | 1e-4 ~ 1e-3 | 归一化对缩放较敏感 |
| **输出层** | add | 1e-6 ~ 1e-5 | 输出层影响大，谨慎注入 |
| **嵌入层** | noise | 1e-5 ~ 1e-4 | 嵌入层可以承受一定噪声 |

### 错误强度等级

```bash
# 🟢 轻微扰动（推荐用于生产测试）
export MEGATRON_CORRUPT_DELTA=1e-6

# 🟡 中等扰动（推荐用于开发测试）
export MEGATRON_CORRUPT_DELTA=1e-5

# 🟠 明显扰动（可能影响训练但不会崩溃）
export MEGATRON_CORRUPT_DELTA=1e-4

# 🔴 强扰动（可能导致训练发散）
export MEGATRON_CORRUPT_DELTA=1e-3
```

## ⚠️ 注意事项

1. **训练稳定性**：
   - `zero`, `flip`, `nan` 操作可能导致训练崩溃
   - 建议先用 `add` 或 `noise` 测试

2. **参数选择**：
   - 避免注入关键参数（如输出层、第一层）
   - 优先选择中间层参数进行测试

3. **DP Rank 选择**：
   - 确保目标 rank 存在（如 DP=2，只能选择 0 或 1）
   - 建议在 rank 0 注入，便于调试

4. **日志大小**：
   - 详细日志会占用较大空间
   - 可以重定向到文件：`bash pretrain_inject.sh > injection.log 2>&1`

5. **性能影响**：
   - 日志输出会略微降低训练速度
   - 仅在需要时启用注入

## 🐛 故障排查

### 问题1：未看到注入日志

**检查**:
```bash
echo $MEGATRON_INJECT_PARAM_CORRUPTION  # 应该是 "1"
```

**解决**:
```bash
export MEGATRON_INJECT_PARAM_CORRUPTION=1  # 确保启用
```

### 问题2：显示 "未找到匹配参数"

**检查参数名**:
```python
# 在 Python 中查看实际参数名
import torch
model = ...  # 加载你的模型
for name, param in model.named_parameters():
    print(name)
```

**调整匹配模式**:
```bash
export MEGATRON_CORRUPT_PARAM_SUBSTR="decoder.layers.0"  # 使用正确的前缀
```

### 问题3：训练立即崩溃

**可能原因**:
- 使用了 `zero`, `flip`, 或 `nan` 操作
- delta 值过大

**解决**:
```bash
export MEGATRON_CORRUPT_OP="add"  # 改用温和的操作
export MEGATRON_CORRUPT_DELTA=1e-6  # 减小扰动幅度
```

## 📚 参考资料

- **代码位置**: `Megatron-LM/megatron/core/pipeline_parallel/schedules.py:436-598`
- **验证脚本**: `sdccheck/test_dp_consistency_violation.py`
- **SQL查询**: `sdccheck/INJECTION_VERIFICATION.sql`
- **训练脚本**: `sdccheck/config/pretrain_inject.sh`

---

**最后更新**: 2025-10-16  
**作者**: SDCCheck Team  
**版本**: 2.0 (增强版，支持6种注入方式)



