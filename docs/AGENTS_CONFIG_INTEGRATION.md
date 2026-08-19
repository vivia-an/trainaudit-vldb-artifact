# Agents 配置集成指南

## 概述

现在 SDCCheck 已经集成了 agents 文件夹中的配置系统，可以直接使用 `agents/llm_config.yaml` 中定义的模型配置。

## 配置文件位置

```
sdccheck/
├── agents/
│   └── llm_config.yaml          # 主配置文件
└── sdccheck/
    └── config_loader.py         # 配置加载器
```

## 配置文件格式

`agents/llm_config.yaml` 格式：

```yaml
specializations:
  test_mini: gpt-4o-mini-direct
  test: gpt-4o-direct
  data_extraction: gpt-4.1-mini-direct
  structured_output: gpt-4.1-mini-direct
  conversation: gpt-4.1-direct
  default: gpt-4.1-direct

models:
  gpt-4.1-direct:
    name: gpt-4.1-direct
    model: gpt-4.1
    type: openai
    api_key: your-api-key-here
    timeout: 3600
  
  gpt-4o-mini-direct:
    name: gpt-4o-mini-direct
    model: gpt-4o-mini
    type: openai
    api_key: your-api-key-here
    timeout: 3600
```

## 使用方式

### 方式1：使用 from_specialization（推荐）

```python
from sdccheck.llm_orchestrator import LLMOrchestrator
from sdccheck.models import TrainingConfig
from sdccheck.generators import ConstraintGenerationMode

# 直接使用 specialization 创建编排器
orchestrator = LLMOrchestrator.from_specialization(
    specialization="default",  # 或 'test', 'conversation', 'data_extraction' 等
    constraint_generation_mode=ConstraintGenerationMode.PREDEFINED
)

# 执行检查
config = TrainingConfig(dp=2, tp=2)
analysis = orchestrator.run_with_hierarchy(
    config=config,
    db_path="path/to/db.db",
    output_dir="path/to/output"
)
```

### 方式2：直接使用 LLMProviderFactory

```python
from sdccheck.llm_providers import LLMProviderFactory

# 从 specialization 创建 provider
provider = LLMProviderFactory.create_provider_from_specialization("default")

# 然后可以直接使用 provider
response = provider.chat_completion("your prompt here")
```

### 方式3：查看可用配置

```python
from sdccheck.config_loader import (
    list_available_specializations,
    list_available_models,
    get_model_config
)

# 列出所有可用的 specializations
specs = list_available_specializations()
print(f"可用的 specializations: {specs}")

# 列出所有模型
models = list_available_models()
print(f"可用的模型: {models}")

# 获取特定配置
config = get_model_config("default")
print(f"模型: {config.model}")
print(f"类型: {config.type}")
```

## Specializations 说明

| Specialization | 模型 | 用途 |
|---------------|------|------|
| `default` | gpt-4.1 | 默认任务 |
| `test` | gpt-4o | 测试任务 |
| `test_mini` | gpt-4o-mini | 快速测试 |
| `conversation` | gpt-4.1 | 对话任务 |
| `data_extraction` | gpt-4.1-mini | 数据提取 |
| `structured_output` | gpt-4.1-mini | 结构化输出 |

## 完整示例

```python
"""完整的层级约束检查示例"""
from sdccheck.llm_orchestrator import LLMOrchestrator
from sdccheck.models import TrainingConfig
from sdccheck.generators import ConstraintGenerationMode

# 1. 创建编排器（使用 agents 配置）
orchestrator = LLMOrchestrator.from_specialization(
    specialization="default",  # 使用默认配置
    constraint_generation_mode=ConstraintGenerationMode.PREDEFINED
)

# 2. 配置训练参数
config = TrainingConfig(dp=2, tp=2, pp=1)

# 3. 执行层级检查
analysis = orchestrator.run_with_hierarchy(
    config=config,
    db_path="data/merged_coredump.db",
    output_dir="logs/analysis",
    max_workers=10,
    target_depth=4
)

# 4. 查看结果
print(f"总约束: {analysis.total_constraints}")
print(f"失败: {analysis.failed_constraints}")
print(f"根本原因: {len(analysis.root_causes)}")
```

## 运行测试示例

```bash
# 方式1：运行带配置测试的示例
python examples/hierarchical_check_with_agents_config.py

# 方式2：仅测试配置加载
python -c "from sdccheck.config_loader import load_config; print('✅ 配置加载成功')"
```

## 环境变量支持

配置文件支持环境变量引用：

```yaml
models:
  gpt-4-env:
    name: gpt-4-env
    model: gpt-4
    type: openai
    api_key: ${OPENAI_API_KEY}  # 从环境变量读取
    timeout: 3600
```

设置环境变量：

```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key"

# Windows
set OPENAI_API_KEY=your-api-key
```

## 不同任务使用不同模型

```python
# 层级构建使用更强大的模型
hierarchy_orchestrator = LLMOrchestrator.from_specialization(
    specialization="conversation"  # gpt-4.1
)

# 数据提取使用更快速的模型
extraction_orchestrator = LLMOrchestrator.from_specialization(
    specialization="data_extraction"  # gpt-4.1-mini
)

# 测试使用更便宜的模型
test_orchestrator = LLMOrchestrator.from_specialization(
    specialization="test_mini"  # gpt-4o-mini
)
```

## 优点

✅ **集中配置** - 所有模型配置在一个文件中  
✅ **灵活切换** - 通过 specialization 轻松切换模型  
✅ **安全管理** - 支持环境变量管理 API key  
✅ **多模型支持** - 为不同任务配置不同模型  
✅ **配置缓存** - 自动缓存配置，避免重复加载  

## 故障排查

### 问题1：找不到配置文件

```
FileNotFoundError: 未找到 llm_config.yaml 配置文件
```

**解决方案**：
- 确保 `agents/llm_config.yaml` 文件存在
- 检查文件路径是否正确
- 尝试使用绝对路径

### 问题2：Specialization 不存在

```
ValueError: 未找到 specialization 'xxx' 的配置
```

**解决方案**：
- 运行 `list_available_specializations()` 查看可用配置
- 检查 yaml 文件中的 specializations 部分
- 使用 'default' 作为回退选项

### 问题3：API Key 无效

```
OpenAI API调用失败: Invalid API Key
```

**解决方案**：
- 检查 yaml 文件中的 api_key 配置
- 如果使用环境变量，确保已正确设置
- 验证 API key 是否有效且有足够的配额

## 迁移指南

### 从旧方式迁移

**旧方式**（需要手动创建 provider）：
```python
llm_config = LLMConfig(model_name="gpt-4", api_key="xxx")
provider = LLMProviderFactory.create_provider("openai", llm_config)
orchestrator = LLMOrchestrator(provider, llm_config)
```

**新方式**（直接使用配置）：
```python
orchestrator = LLMOrchestrator.from_specialization("default")
```

更简单、更清晰、更易维护！

## 下一步

1. 根据需求编辑 `agents/llm_config.yaml`
2. 配置不同 specialization 使用不同模型
3. 使用环境变量管理敏感信息
4. 运行示例测试配置是否正确






