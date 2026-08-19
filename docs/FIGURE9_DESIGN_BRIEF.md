# Figure 9 设计 brief — Megatron-LM #4641（机制示意图）

**目标位置**：论文 §6 Hunt phase 段落，单栏单 panel
**输出文件**：`figures/fig_hunt_megatron_4641.pdf`（vector PDF）
**LaTeX 引用**：当前 `figures/fig_hunt_novel_bugs.pdf` —— 设计师交付后由开发改 `\includegraphics` 路径

---

## 一句话目的

让读者一眼看明白两件事：
1. **bug 怎么发生的**：tied embedding 在两个 PP stage 上走了不同的 tagging 分支，导致优化器路由分裂
2. **TrainAudit 怎么检测的**：在 param-group registration hook 上检查"tied 参数必须被路由到同一 optimizer"

**注意**：这是 **schematic / 机制图**，不需要任何实验数据，纯示意。

---

## 推荐布局：三层流图

整体竖向三层、左右两列对照（`PP rank 0` vs `last PP stage`）。

```
┌───────────────────────────────────────────────────┐
│                                                   │
│   PP rank 0                  last PP stage (MTP)  │   <- 第 1 层：模型初始化
│   ┌──────────┐               ┌──────────┐         │      显示 tied embedding
│   │ embed W  │ ◄── tied ──►  │ embed W  │         │      在两端各有副本
│   └────┬─────┘               └────┬─────┘         │
│        │ self.pre_process         │ self.pre_process│
│        │ = True                   │ = False        │
│        │ ✓ tagging runs           │ ✗ tagging skipped │
│        ▼                          ▼                │
│   [is_embedding_or_       [no tag]   ⚠            │   <- 第 2 层：tagging 分支
│    output_parameter]                               │      展示 bug 根因
│                                                   │
├───────────────────────────────────────────────────┤
│                                                   │
│   Optimizer param-group routing                   │   <- 第 3 层：路由决策
│   ┌──────────┐               ┌──────────┐         │      左走 Adam ✓
│   │  Adam    │               │  Muon    │   ⚠    │      右被 Muon 抢走 ✗
│   │  (✓)     │               │  (✗)     │         │
│   └────┬─────┘               └────┬─────┘         │
│        │ ΔW_adam                  │ ΔW_muon       │
│        ▼                          ▼                │
│   ════════════ DRIFT ════════════                  │   <- 第 4 层：后果
│   tied replicas diverge between all-reduces        │      tied 副本 drift
│                                                   │
└───────────────────────────────────────────────────┘
                       ▲
                       │ TrainAudit hook
              ┌────────┴────────────┐
              │ Rule:               │   <- 侧标：检测点
              │ MEGATRON_TIED_      │      画在第 2→3 层之间
              │ EMBED_OPT_ROUTING   │      用箭头指向 routing 决策
              │                     │
              │ "tied params must   │
              │  share opt group    │
              │  across PP ranks"   │
              └─────────────────────┘
```

---

## 视觉规格

**尺寸**：`\columnwidth × ~5.5cm`（单栏宽，竖向矩形）

**配色**：
- 正常路径（PP rank 0 一侧）：冷色，绿/蓝调（`#3B8E8E` 类）
- bug 路径（last PP stage 一侧）：暖色，橙红调（`#E07B39` 类）
- TrainAudit 检测框：中性灰底 + 边框，加 ⚙ 或锁形 icon
- DRIFT 箭头/带：醒目橙红，加粗，体现"两条线最终发散"

**字体**：Times / Latin Modern；正文与论文一致；代码片段（`self.pre_process`、`is_embedding_or_output_parameter` 等）用等宽字体（`\texttt`）

**关键标注（必须出现）**：
1. `tied` 双向箭头连接两份 embedding 副本（顶层）
2. `self.pre_process = True / False` 标在分支判断处
3. `✓ tagging runs` vs `✗ tagging skipped`（用对勾/叉号 icon）
4. Adam 和 Muon 两个 optimizer 框
5. `DRIFT` 大字标注在底部
6. TrainAudit rule 名 `MEGATRON_TIED_EMBED_OPT_ROUTING` 必须出现在检测框里
7. 一句 plain-English 规则描述：`"tied params must share opt group across PP ranks"`

---

## 审美原则

- **机制 > 美学**：读者必须能看懂"哪一步出错、哪一步被抓"
- 流向自上而下：init → tagging → routing → consequence
- 左右对照清晰：左 = 正常，右 = 异常（颜色 + icon 强化）
- TrainAudit 检测框不要喧宾夺主，但要明显（侧标 + 引线）
- 不要文字过多——每个 box 内 ≤ 3 行字
- **绝对不要堆叠 panel**

---

## 设计师可参考的同类示例

- 任何"compiler error path"示意图（左正常 / 右异常 + 检测点）
- Megatron-LM 论文里的 PP / TP 通信流图（同款竖向三层结构）
- TrainCheck (OSDI '25) 论文 Figure 3 风格（机制 + 检测点叠加）

---

## Caption 参考（已经在 main.tex 里，无需改动）

英文版：
> New silent-error candidate that TrainAudit surfaces on Megatron-LM HEAD without any fault injection (Muon + PP + MTP configuration). The tied `word_embeddings` on the MTP-only last PP stage is silently routed to Muon instead of Adam, producing divergent optimizer updates between PP rank 0 (Adam) and the last PP stage (Muon) that drift between embedding-grad all-reduces; the corresponding embedding-replica L2 distance grows monotonically rather than collapsing to zero. Disclosed via Megatron-LM issue #4641 (PR #4642).

> 注：caption 里提到的 "L2 distance grows monotonically" 是文字描述，**不需要画在图里**——图只画机制。如果你觉得 caption 与机制图不匹配，告诉开发也调整 caption。

---

## 交付清单

1. PDF（vector，single column ≈ 240pt 宽，约 5.5cm 高）
2. 源文件（`.svg` / `.drawio` / `.fig` 任一即可，方便后续微调）
3. 完成后告诉开发，更新 `\includegraphics` 路径并删除 `% TODO` 注释
