# draw.io MCP 科研画图 Prompt 模板

这份文档提供可直接复制的 prompt。推荐固定使用“先规范、后出图、再微调”的三段式流程。

## 1. 阶段一：图规范生成

把下面模板发给 AI，并把你的科研内容粘进去。

```text
你现在是科研论文作图规划助手，不直接开始画图。

请根据我提供的科研内容，先输出一份结构化 YAML 图规范，用于后续通过 draw.io MCP 自动生成科研图。

要求：
- 面向论文图，不要做成演示文稿风格
- 优先保证逻辑清晰，不追求装饰
- 每个节点只表达一个概念
- 节点文字尽量短
- 如果内容过长，请主动压缩成论文图里常见的简洁短语
- 默认只使用 workflow / mechanism / architecture / experiment / pipeline 之一
- 默认布局只使用 left_to_right 或 top_to_bottom

请输出以下字段：
- title
- diagram_type
- audience
- direction
- nodes
- edges
- groups
- annotations
- style
- review_checklist

其中：
- nodes: 每个节点包含 id, text, type, group, priority
- edges: 每条边包含 from, to, label, style
- groups: 每个组包含 id, name, purpose
- style: 包含 theme, palette, line_style, emphasis_rule
- review_checklist: 列出生成图前必须检查的 5-8 条事项

我的科研内容如下：

[在这里粘贴科研内容]
```

## 2. 阶段二：MCP 创建图

当 YAML 规范已经确认后，发送下面模板。

```text
根据刚才确认的 YAML 图规范，使用 draw.io MCP 创建一个可编辑的科研图。

执行要求：
- 先建立整体布局，再创建局部元素
- 优先按 group 分区
- 默认使用左到右布局，除非规范里明确要求 top_to_bottom
- 同组元素必须对齐，间距尽量一致
- 所有文字尽量控制在两行以内
- 优先使用圆角矩形、普通矩形、菱形、注释框、分组框
- 避免花哨图标、强渐变、重阴影
- 连线尽量减少交叉，优先使用正交连接或简洁折线
- 先完成整体图，再进行一次自检

自检内容：
- 是否有重叠元素
- 是否有明显交叉线
- 是否有文本过长
- 是否有未对齐的模块
- 是否有语义不清楚的边标签

如果发现问题，请直接在图中修正后再结束。
```

## 3. 阶段三：常用微调指令

### 布局

```text
把整张图改成三列结构：Input / Method / Output，并重新对齐。
```

```text
把方法部分改成上中下三层结构，减少交叉线。
```

```text
保持语义不变，把布局改得更紧凑一些，适合论文单栏宽度。
```

### 形状

```text
把所有 model 节点改成圆角矩形，把 decision 节点改成菱形。
```

```text
给最核心的模块加浅色背景框，但保持整体风格克制。
```

### 样式

```text
把整张图改成低饱和蓝灰配色，适合学术论文。
```

```text
去掉不必要的装饰，让它更像论文里的方法图。
```

```text
统一所有模块的边框、字号和箭头样式。
```

### 注释

```text
给每个主模块加编号 A/B/C/D，并把补充说明移到 annotation layer。
```

```text
保留主流程不变，在右下角增加一个 legend，解释颜色和虚线含义。
```

## 4. 中文科研内容压缩模板

如果你手头是很长的一段中文描述，先用下面这个模板压缩。

```text
请先不要画图。

请把下面这段科研方法描述压缩成适合论文方法图的短语化结构：
- 每个步骤不超过 12 个汉字
- 按输入、处理、输出组织
- 找出并标记 3 到 6 个最关键模块
- 找出模块之间的关系动词
- 删除不适合放进图里的解释性长句

输出格式：
1. 主标题
2. 关键模块列表
3. 模块关系
4. 推荐图类型

[在这里粘贴原始描述]
```

## 5. 适合你当前论文主题的模板

针对 `TrainAudit` 这类系统论文，可直接用下面这个模板。

```text
请根据下面内容，先生成一份用于 draw.io MCP 的 YAML 图规范，图类型为 architecture。

要求：
- 面向系统论文方法图
- 强调 Input -> Invariant Miner -> Data Collector -> Verifier -> Output 的主流程
- 表现多 Agent 协作，但不要让图过于拥挤
- 区分 online runtime path 和 offline analysis path
- 重点模块可用浅色背景框突出
- 辅助说明放 annotation，不要塞进主节点

建议 group：
- Training System
- Invariant Mining
- Data Collection
- Verification
- Outputs

请同时给出推荐边标签和推荐布局。
```
