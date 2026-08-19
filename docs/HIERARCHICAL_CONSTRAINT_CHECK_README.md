# 层级约束检查与根因分析 - 实现文档

## 概述

本实现提供了基于约束树的层级检查机制，支持并发SQL查询和智能根因分析。

## 核心特性

### 1. **AI驱动的约束层级构建**
- 使用LLM分析约束语义，自动构建3-4级层级结构
- 按照"粗粒度->细粒度"、"通用->具体"的原则组织约束

### 2. **并发约束检查**
- 支持10个约束并发查询
- 失败节点的子树会被继续检查
- 线程安全的数据库访问

### 3. **智能根因分析**
- 识别失败链路（从根到叶子）
- 定位根本原因（最上层失败节点）
- 分析传播影响范围
- 生成JSON格式的详细报告

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                  LLMOrchestrator                            │
│  run_with_hierarchy() - 主入口                              │
└────────────────────┬────────────────────────────────────────┘
                     │
     ┌───────────────┼──────────────┐
     │               │              │
     ▼               ▼              ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐
│ Hierarchy│  │  Constraint  │  │  Concurrent  │
│ Builder  │  │  TreeManager │  │  Executor    │
└──────────┘  └──────────────┘  └──────────────┘
     │               │              │
     │      构建      │              │
     └──────────────>│              │
                     │   逐层并发   │
                     └─────────────>│
                                    │
                     ┌──────────────┘
                     ▼
              ┌──────────────┐
              │ Root Cause   │
              │  Analyzer    │
              └──────────────┘
```

## 核心模块

### 1. `models.py` - 数据模型
- `ConstraintNode`: 约束树节点
- `ConstraintTree`: 约束树结构
- `FailureChain`: 失败链路
- `RootCauseAnalysis`: 根因分析结果
- `BatchExecutionResult`: 批量执行结果

### 2. `constraint_tree.py` - 约束树管理
- `ConstraintTreeManager`: 构建和管理约束树
- 提供节点查询、层级遍历、路径查找等功能

### 3. `constraint_hierarchy_builder.py` - 层级构建
- `ConstraintHierarchyBuilder`: 使用AI构建约束层级
- 自动分析语义相似性
- 生成3-4级树结构
- 验证层级合理性

### 4. `concurrent_executor.py` - 并发执行器
- `ConcurrentConstraintExecutor`: 并发执行约束检查
- 支持批量执行（10个/批）
- 逐层执行策略：失败节点的子树继续检查

### 5. `root_cause_analyzer.py` - 根因分析器
- `RootCauseAnalyzer`: 分析失败原因
- 识别失败链路和根本原因
- 生成修复建议
- 导出JSON报告

### 6. `database_executor.py` - 数据库执行器（已增强）
- 支持线程安全的并发查询
- 批量执行支持

### 7. `llm_orchestrator.py` - 编排器（已扩展）
- 新增`run_with_hierarchy()`方法
- 集成完整的层级检查流程

## 使用方法

### 基本用法

```python
from sdccheck.models import TrainingConfig
from sdccheck.llm_orchestrator import LLMOrchestrator
from sdccheck.llm_providers import LLMProviderFactory
from sdccheck.config import LLMConfig
from sdccheck.generators import ConstraintGenerationMode

# 配置
config = TrainingConfig(dp=2, tp=2, pp=1)
llm_config = LLMConfig(model_name="gpt-4", temperature=0.1, max_tokens=4000)

# 创建编排器
provider = LLMProviderFactory.create_provider("openai", llm_config)
orchestrator = LLMOrchestrator(
    llm_provider=provider,
    llm_config=llm_config,
    constraint_generation_mode=ConstraintGenerationMode.PREDEFINED
)

# 执行层级检查
analysis = orchestrator.run_with_hierarchy(
    config=config,
    db_path="path/to/database.db",
    output_dir="path/to/output",
    max_workers=10,  # 并发数
    target_depth=4   # 树深度
)

# 查看结果
print(f"总约束: {analysis.total_constraints}")
print(f"失败: {analysis.failed_constraints}")
print(f"根本原因: {len(analysis.root_causes)}")
```

### 运行示例

```bash
cd sdccheck
python examples/hierarchical_check_example.py
```

## 执行流程

### 阶段1: 准备
1. 分析数据库模式
2. 生成适用的约束
3. 使用AI构建3-4级层级结构
4. 验证层级关系

### 阶段2: 并发执行
1. 从Level 0（根节点）开始
2. 并发执行该层级的所有约束（10个/批）
3. 记录失败的节点
4. 继续执行失败节点的子树
5. 重复直到所有相关节点都被检查

### 阶段3: 根因分析
1. 收集所有失败节点
2. 识别失败链路
3. 定位根本原因（最上层失败）
4. 分析影响范围
5. 生成修复建议
6. 导出JSON报告

## 输出格式

### JSON报告结构

```json
{
  "analysis_id": "RCA_20250124_143022",
  "timestamp": "2025-01-24T14:30:22",
  "total_constraints": 45,
  "passed_constraints": 38,
  "failed_constraints": 7,
  "failure_chains": [
    {
      "path": ["node_1", "node_2", "node_3"],
      "root_cause": "backward后DP参数cksum一致性检查",
      "affected_nodes": ["node_2", "node_3"],
      "severity": "high"
    }
  ],
  "root_causes": [
    {
      "node_id": "DP_param_consistency_abc123",
      "constraint_name": "backward后DP参数cksum一致性检查",
      "level": 1,
      "affected_count": 3,
      "suggestion": "检查backward阶段的AllReduce同步"
    }
  ],
  "recommendations": [
    "检查 backward 阶段的梯度同步",
    "建议检查 DP 通信组配置"
  ],
  "constraint_tree": {
    "max_depth": 4,
    "total_nodes": 45,
    "roots": [...]
  }
}
```

## 性能特点

- **并发度**: 10个约束同时执行
- **智能剪枝**: 只检查失败节点的子树
- **线程安全**: 支持多线程并发数据库查询
- **可扩展**: 支持自定义约束和层级结构

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_workers` | 最大并发数 | 10 |
| `target_depth` | 目标树深度 | 4 |
| `output_dir` | 输出目录 | None |
| `stop_on_all_pass` | 全部通过时是否停止 | False |

## 注意事项

1. **AI依赖**: 层级构建需要LLM支持，确保配置正确的LLM提供商
2. **数据库连接**: DuckDB支持并发读取，但请确保连接稳定
3. **内存使用**: 大规模约束树可能占用较多内存
4. **执行时间**: 并发执行可显著减少总时间，但仍取决于约束数量

## 扩展开发

### 添加自定义约束层级

```python
# 手动指定层级关系
constraint.parent = "parent_constraint_name"
constraint.children = ["child1", "child2"]
constraint.level = 2
```

### 自定义根因分析逻辑

```python
class CustomRootCauseAnalyzer(RootCauseAnalyzer):
    def _calculate_severity(self, root_cause, affected_nodes):
        # 自定义严重程度计算
        return "custom_severity"
```

## 故障排查

### 常见问题

**Q: 层级构建失败**
A: 检查LLM配置和API密钥，确保模型支持足够的tokens

**Q: 并发执行出错**
A: 检查数据库连接和SQL语法，确保约束logic字段正确

**Q: 根因分析为空**
A: 检查是否有约束失败，空结果说明所有约束都通过

## 版本信息

- **实现版本**: 1.0
- **实现日期**: 2025-01-24
- **Python要求**: >= 3.8
- **主要依赖**: duckdb, pandas, pydantic, concurrent.futures

## 贡献者

本实现基于用户需求开发，集成了AI驱动的语义分析和高性能并发执行机制。








