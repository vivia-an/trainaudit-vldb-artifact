# draw.io MCP 科研画图工作流

这份文档给出一套可重复使用的方案，用 `Claude / Codex / Cursor + draw.io MCP server` 自动生成并反复编辑科研图。

目标不是让模型一次性吐出不可维护的 XML，而是把流程拆成三步：

1. 科研内容 -> 结构化图规范
2. 图规范 -> draw.io 可编辑图
3. 反复修改布局、形状、箭头、样式并导出

## 1. 推荐方案

优先使用 `drawio-mcp-server --editor`，原因是：

- 配置简单
- 支持浏览器内编辑器
- 支持 AI 持续改图，不只是导入一次 XML
- 适合论文方法图、流程图、系统架构图

参考：

- 官方 AI / draw.io 说明：<https://www.drawio.com/doc/faq/ai-drawio-generation>
- 社区 MCP Server：<https://github.com/lgazo/drawio-mcp-server>

## 2. 环境要求

- `Node.js >= 20`
- 一个支持 MCP 的客户端：`Claude Desktop`、`Cursor`、`Codex`
- 本地浏览器

检查 Node:

```bash
node -v
```

## 3. MCP 配置

### Claude Desktop

配置文件通常在：

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\\Claude\\claude_desktop_config.json`

示例：

```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "drawio-mcp-server", "--editor"]
    }
  }
}
```

保存后重启 Claude Desktop。

### Cursor

配置文件通常在：

- 项目级：`.cursor/mcp.json`
- 用户级：`~/.cursor/mcp.json`

示例：

```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "drawio-mcp-server", "--editor"]
    }
  }
}
```

保存后重启 Cursor。

### Codex

配置文件：

- `~/.codex/config.toml`

示例：

```toml
[mcp_servers.drawio]
command = "npx"
args = ["-y", "drawio-mcp-server", "--editor"]
```

保存后重启 Codex。

## 4. 启动方式

配置完成并重启客户端后，MCP server 会由客户端拉起。`--editor` 模式通常会在本地提供一个浏览器编辑器，默认常见地址为：

```text
http://localhost:3000
```

如果客户端侧没有自动打开，手动访问即可。

## 5. 最稳的生成流程

不要直接说“帮我画个图”。稳定做法是固定成两个阶段。

### 阶段 A：先产出图规范

先让 AI 把科研内容整理成结构化规范，只输出 YAML，不开始画图。

输入材料可以是：

- 论文方法描述
- 实验步骤
- 系统模块说明
- 你已有的草图
- 一段中文或英文摘要

规范至少要包含：

- `title`
- `diagram_type`
- `audience`
- `direction`
- `nodes`
- `edges`
- `groups`
- `annotations`
- `style`

### 阶段 B：让 AI 使用 MCP 创建图

等 YAML 确认后，再让 AI 调用 draw.io MCP：

- 新建 diagram
- 创建 groups / layers
- 创建 nodes
- 创建 edges
- 统一布局
- 调样式
- 自检重叠、交叉、未对齐

## 6. 科研图约束

为了让图更像论文图，而不是演示文稿，建议固定这些约束：

- 只用 `workflow / mechanism / architecture / experiment / pipeline`
- 默认布局只用 `left_to_right` 或 `top_to_bottom`
- 节点类型只用 `process / data / model / result / note / decision / group`
- 每个节点只表达一个概念
- 每个节点文本尽量短
- 一条边只表达一种关系
- 主色不超过 3 个
- 避免大面积渐变、阴影、装饰图标
- 同组元素必须对齐
- 线条尽量正交或简洁折线
- 文本不要超过 2 行

## 7. 推荐的自然语言修改指令

下面这些修改指令通常比较稳定：

- 把整张图改成从左到右布局
- 把输入、方法、输出分成三列
- 把方法模块改成上中下三层
- 所有 decision 节点改成菱形
- 所有 model 节点改成圆角矩形
- 把辅助说明移动到单独 annotation layer
- 把结果相关箭头改成虚线
- 让所有模块宽度一致、上下间距一致
- 减少交叉连线，优先使用正交连接
- 把颜色改成更适合论文的低饱和蓝灰配色
- 给每个主模块加编号 A/B/C/D

## 8. 推荐的图类型映射

### 方法框架图

- `Input / Method / Output` 三列布局
- 方法部分内部可再拆 `Stage 1 / Stage 2 / Stage 3`

### 系统架构图

- 按 `User / Agent / Tool / Storage / Output` 分层
- 外部系统用虚线边框

### 实验流程图

- 按时间或阶段串联
- 可增加 `control` 和 `treatment` 两条支路

### 机制示意图

- 中间主机制
- 上下游因素分布在左右
- 注释单独放 layer

## 9. 导出建议

投稿图建议优先导出：

- `SVG`：后续还能精修
- `PDF`：适合 LaTeX
- `PNG`：仅用于快速预览

如果后续还要排版，建议在 draw.io 中统一完成：

- 字号
- 线宽
- 对齐
- 留白

## 10. 常见失败模式

### 文本太长

处理方式：

- 先缩短节点文本
- 节点里只保留短语
- 细节移到 caption 或 annotation

### 连线太乱

处理方式：

- 先按 group 重排
- 限制方向为单向流动
- 增加中间汇聚节点

### 图过于像 PPT

处理方式：

- 降低饱和度
- 去掉阴影、渐变、花哨图标
- 控制色数和边框风格

### 一次生成结构不稳

处理方式：

- 强制先输出 YAML
- 人工确认逻辑后再生成图
- 局部修改，不整图重来

## 11. 实际使用建议

最稳妥的习惯是：

1. 用 `scientific_diagram_spec.yaml` 草拟结构
2. 把内容贴给 AI，让它先补全规范
3. 让 AI 通过 MCP 创建图
4. 用 2 到 5 轮自然语言微调
5. 导出 `SVG/PDF`
6. 放进论文或幻灯片

如果是数据图和框架图混排，建议：

- 数据图用 `Matplotlib / Plotly / GraphPad Prism`
- 框架图用 `draw.io`
- 最终统一导出为 `PDF/SVG`
