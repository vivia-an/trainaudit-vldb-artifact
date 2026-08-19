# E2 — P9–P16 Real Candidate Search · Summary

> Date: 2026-05-17 · Input: `axial_clusters.json` + `benchmark/bugs/<id>/config.json`
> Full data: `p9_p16_real_candidate_search.csv` (167 rows).

## Tier definition (`runnable_tier` column)

| Tier | 条件 | 含义 |
|---|---|---|
| `T_GREEN_full_replay` | `reproduction_status=reproduced` + `reproduce.sh` + (`detect.py` 或 `trainaudit_driver.py`) + commit pair 全 | 几乎可以直接 replay |
| `T_YELLOW_taudit_only` | commit pair 全 + `trainaudit_run.sh` + `run.sh`，无完整 reproduce.sh | TrainAudit 走得通，但 buggy/fixed 端到端需要补 reproduce wrapper |
| `T_YELLOW_detect_only` | commit pair 全 + `detect.py` + `run.sh`，无 trainaudit_driver | 需要补 TrainAudit adapter，但 baseline reproduce 已有 |
| `T_BLUE_config_only` | commit pair 全，仅 config.json | 候选已识别，所有 driver 都需要新写 |
| `T_RED_source_only` | `detection_method=source_analysis` | 不进主表（违反文档 §1.2 runtime observability） |
| `T_RED_no_commit_pair` | 无 buggy/fixed commit | 不进主表（违反文档 §1.4 可对照） |

## 每个 P9–P16 的现状

| Pattern | top candidates (tier) | 已被主表 real-set 覆盖? | 建议 |
|---|---|---|---|
| **P9** Init Distribution | `O-NEW-5` YELLOW, `O-NEW-7` YELLOW | ❌ | 推 `O-NEW-5` 替换 ID1；备选 `O-NEW-7` |
| **P10** Config-Implied Coupling | (B12 / O-005 已在主表，YELLOW) | ✅ **B12+O-005 已覆盖** | **不需要新加**；P10 标"已被 B12/O-005 覆盖" |
| **P11** Position Encoding & Doc Boundary | `O-NEW-2` YELLOW | ❌ | 推 `O-NEW-2` 替换 PE1 |
| **P12** Algorithm Variant | (B11 已在主表) + `B9` / `M-014` YELLOW | ✅ **B11 已覆盖** | P12 标"已被 B11 覆盖"；可选 `B9` 作为额外 |
| **P13** Tensor Aliasing | `D-NEW-9` YELLOW, `M-008` YELLOW | ❌ | 推 `D-NEW-9` 或 `M-008` 替换 TA1 |
| **P14** Sharded State | **全 BLUE**（无 YELLOW 候选） | ❌ | 没有现成 driver；从 BLUE 中挑 1 个（如 `O-NEW-27`）需要新写 reproduce + driver |
| **P15** Counter Width | `OC-NEW-6` YELLOW | ❌ | 推 `OC-NEW-6` 替换 CW1 |
| **P16** Loss Normalization | `O-NEW-3` YELLOW, `OC-NEW-4` YELLOW | ❌ | 推 `O-NEW-3` 替换 LN1 |

## 我建议的最终结论

- **P9 / P11 / P13 / P15 / P16**：各有 1–2 个 YELLOW 候选，写 reproduce wrapper 后大概率能跑通 → 把对应 surrogate 替换成真实 bug。
- **P10 / P12**：已被主表内的 B11/B12/O-005 隐式覆盖，**不需要再加新行**；正文 / 附录写"P10 由 B12+O-005 representative，P12 由 B11 representative，无需独立 case"。
- **P14**：8 个 BLUE 候选都没 driver，要新写代价高；按文档 §3 建议"survey-supported but not in end-to-end replay set"，**P14 退到 survey-only**。

## 候选清单（推荐 6 个新真实 bug 进 E1 验证）

| 替换关系 | 真实 bug | framework | tier | 备注 |
|---|---|---|---|---|
| ID1 → P9 | `O-NEW-5` | olmo | YELLOW_taudit_only | 主推 |
| PE1 → P11 | `O-NEW-2` | olmo | YELLOW_taudit_only | 仅此一个 YELLOW |
| TA1 → P13 | `D-NEW-9` | deepspeed | YELLOW_taudit_only | 备选 `M-008` |
| CW1 → P15 | `OC-NEW-6` | olmo-core | YELLOW_taudit_only | 仅此一个 YELLOW |
| LN1 → P16 | `O-NEW-3` | olmo | YELLOW_taudit_only | 备选 `OC-NEW-4` |
| AV1 → P12 | `B9` | (来自 axial cluster) | YELLOW_taudit_only | 已被 B11 隐式覆盖，**可选**补一行而已 |

> P10 / P14 不出新 case。

## 需要你拍板的疑点

1. **P10 / P12 真的"隐式覆盖"成立吗？**
   - axial_clusters.json 把 B11→P12、B12→P10、O-005→P10，但主表里 B11 现在挂的 invariant 是 `clip-grad-bounded`（P11 类）、B12 挂 `initial-lr-present`、O-005 挂 `checkpoint-preserve-rng`——和 P10/P12 不完全是一码事。
   - 是否同意我"P10 由 B12+O-005 二维覆盖"的论证？还是必须为 P10 / P12 各挑一个新真实 bug？
2. **P14 退到 survey-only 是否可接受？**8 个 BLUE 都要新写 reproduce + trainaudit driver，工作量大。文档 §3 给了这条退路。
3. **YELLOW_taudit_only 是否需要在 E1 阶段补 reproduce.sh 才算合格？**
   - 这些 bug 已经有 `trainaudit_run.sh`，能直接给 TrainAudit 喂数据；但 same-harness 重跑（E3）需要同时拿 buggy/fixed 两个 phase，没 reproduce.sh 就要手动 `git checkout` 两次。
   - 我可以给每个 YELLOW 候选补一个最小化的 `reproduce.sh`（基于现有 trainaudit_run.sh + git checkout 模板），但需要你授权。
4. **是否选 5 个全替换上去**？还是更保守，只挑 P11/P13/P15/P16 这种 surrogate 严格对位的 4 个（P9 因为 ID1 是 init-distribution-std-bound 单测，理论上 O-NEW-5 的真实 init bug 不一定走同一条 invariant）。
