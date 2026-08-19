# E0 Audit — Suspect / Decision Points to Confirm

> Date: 2026-05-17 · Input: `table6_current_audit.csv`
> 这里只列出"我的初步判定 + 不确定点"，等你确认后再凿死 `keep_in_main` / `source_kind`。

## 1. 明确进主表（real_commit_pair, keep_in_main=true）— 9 行

| case | 疑点 | 需要你确认 |
|---|---|---|
| B1 / B3 / B8 / B11 / B12 | `config.json.reproduction_status=null`（不是 `"reproduced"`），但目录里 reproduce.sh + trainaudit_driver 都齐 | 这 5 行是否需要在 GPU 上再过一遍才能挂 `is_real_silent_error=true`？还是按"artifacts 齐 = 算数"通过 |
| M-020 | 有 reproduce.sh+runs/，没有 trainaudit_driver.py；fixed 行为是 `AssertionError`（即 fixed 不再 silent，是显式拒绝），这其实是边界情况：buggy silent / fixed crash | 仍按"真实 silent + 拒绝式 fix"算入主表？ |
| O-005 | issue_url 指向 commit 而非 issue；buggy/fixed commit 没带 `~1` 后缀 | 接受 commit URL 作为 issue_url 的替代？ |
| O-NEW-9 / OC-NEW-2 | 有 buggy/fixed commit pair，**但 `issue_url=null`**；需要补 PR/issue 链接 | 是否要求我先去 OLMo / OLMo-core 仓库找对应 PR 才能保留？还是 commit-pair 已经足够 |

## 2. surrogate → 真实 blueprint 替换候选（keep_in_main=false 待 E1）— 3 行

| 旧 surrogate | blueprint | 当前 artifacts | 替换可能性 |
|---|---|---|---|
| **CF1** | M-010 | M-010: config + detect + reproduce + run + runs/ + trainaudit_driver + trainaudit_run.sh，`reproduction_status=reproduced` | **高**：E1 大概率成功 |
| **CM1** | O-014 | O-014: **只有 config.json**，detection_method=`source_analysis`，**无 buggy/fixed commit**；config 里 `reproduction_status=reproduced` 与 `source_analysis` 互相矛盾 | **低**：除非真有上游 commit 否则不应作为 "runtime-observable silent" 进主表。文档 §1 第 2 条要求 buggy run 能产生 runtime violation——`source_analysis` 不满足 |
| **OF1** | D-029 | D-029: config + trainaudit_driver + trainaudit_run.sh，无 reproduce.sh；有 buggy/fixed commit + PR URL；detection_method=`source_analysis` | **中**：commit 全、driver 有，但"silent error"的运行时信号是 sub-percent drift（OF1 的 surrogate_bug_signal 写得很清楚），实际 runtime 检测路径就是 dtype 一致性——可以保留 |

⚠ **关键疑点**：CM1/O-014 与 OF1/D-029 的 detection_method 字段都是 `source_analysis`。按文档 §1 第 2 条，主表要求"buggy run 在目标触发条件下不以 crash/assert/NaN explosion 作为主要表象；训练或训练相关逻辑能够继续执行，但语义结果错误"——`source_analysis` bug 没有运行时违反 → 严格意义上不满足。如果你坚持只放 runtime-observable bug，**CM1 与 OF1 都应放弃**，主表会少 2 行。

## 3. 从主表移除（synthetic_surrogate, keep_in_main=false）— 8 行

ID1 / CC1 / PE1 / AV1 / TA1 / SC1 / CW1 / LN1 全部从主表移除，移到附录 sanity check。这点和文档 §0、§2.E2 完全一致，**无疑点**。

⚠ 唯一可以讨论的：P9–P16 中实际能找到真实 replay 的 pattern（见 E2 检索）是否需要在主表里"用真实 bug 替回"。目前 axial_clusters.json 显示：

- **P10 Config-Implied Coupling**：里面有 B12（已在主表）、O-005（已在主表）、M-022 等 → 可能 P10 的覆盖已经被 B12 / O-005 隐式覆盖了
- **P9 / P11–P16**：在 392 pool 里有不少候选（见 E2），但谁能真正端到端跑得动需要 E2 一一鉴别

## 4. boundary cases — 2 行

| case | 疑点 | 需要你确认 |
|---|---|---|
| LC1 (O-003) | buggy/fixed commit 都齐，`reproduction_status=reproduced` | 直接保留为 boundary，无疑点 |
| DL2 (O-022) | **没有 buggy/fixed commit**，`detection_method=source_analysis`；`reproduction_status=reproduced` 与无 commit pair 矛盾 | 严格按文档 §1 第 4 条（必须可对照），DL2 不合格 → 应该从主表移除，正文写"该 pattern 来自 392 真实 bug 的归纳，但未进入 replay 集合"。是否同意？ |

## 5. 主表最终规模初步估计

按"严格"口径（runtime-observable + commit pair 齐 + 必要时 GPU 验证通过）：
- 真实 detector-coverable：9 行（B1 B3 B8 B11 B12 M-020 O-005 O-NEW-9 OC-NEW-2）
- + E1 替换成功：最多再 +3 行（M-010, O-014, D-029），最少 +1 行（只有 M-010 大概率成功）
- + E2 真实 P9–P16 替换：取决于检索 + 复现结果，先按 0–4 行估
- boundary：1 行（LC1）；DL2 若放弃则 0 行

→ **N_real ∈ [10, 16]，N_boundary ∈ [1, 2]**，远低于当前 22 行的"凑数"。

这与文档 §3「不固定 22」一致。请确认你接受这个范围，再决定 E1/E2 是否要"宁缺毋滥"。

## 6. 待你拍板的决策清单

1. **B1/B3/B8/B11/B12**：是否要求我去 eval-gpu-0 先 smoke test 一遍 trainaudit_driver 才挂 `keep_in_main=true`？还是 artifacts 齐就算
2. **CM1/O-014**：源码型 bug 是否一律不进主表？（如是，CM1 直接 drop，不必跑 E1）
3. **OF1/D-029**：dtype-preserve 这个 invariant 是 runtime-observable 的（只是 surrogate 用 sub-percent drift 命题描述了一下），是否同意保留作 E1 候选
4. **O-NEW-9 / OC-NEW-2**：缺 issue_url 是否阻塞 keep_in_main
5. **M-020**：fixed 行为是 assert crash（拒绝式 fix），是否仍按 silent-error 入主表
6. **DL2/O-022**：缺 commit pair 是否直接降级为"survey-supported but not in main replay set"
