# §6.6 Portability — D1 + D2 真相校正

## D2: "288/295 = 97.6%" 的真相

**Runbook 误读**：runbook §D2 把 288/295 解读为"driver pool 跑批 7 个失败"。
**实际**（`benchmark/eval/manifest_summary.md` + `manifest.json`）：
- 295 = 池子里有 `config.json` 的 bug 总数
- 288 = 已经写好 `trainaudit_run.sh` driver 的 bug 数
- 7 = **尚未写 driver** 的 bug 数，**不是"跑失败"**

见 `d2_missing_drivers.csv`：

| bug_id | framework | category | reproduction_status |
|---|---|---|---|
| O-006 | olmo | data_loading | reproduced |
| O-010 | olmo-core | optimizer_state | reproduced |
| O-014 | olmo-core | communication | reproduced |
| O-016 | olmo-core | lr_schedule | reproduced |
| O-021 | olmo-core | data_loading | reproduced |
| O-022 | olmo-core | data_loading | reproduced |
| O-025 | olmo-core | numerical | reproduced |

7 个全部 **OLMo / OLMo-core**，原 bug 都已验证可复现，但 trainaudit driver 还没写。
集中三类：**data_loading×3, optimizer_state×1, communication×1, lr_schedule×1, numerical×1**。

→ §6.6 应写："剩下 7 bug 还在 OLMo / OLMo-core 数据加载、optimizer state、communication
和 LR schedule 几条路径上等 adapter / detect.py 补完，不属于检测失败。" 不要继续讲
"GPU-OOM / PyTorch nightly / commit removed upstream" 这种 runbook 编出来的故事。

## D1: 121 tests 的真相

`pytest trainaudit/tests -q --collect-only` 实测 **121 tests**，分布：
- **104 generic**：store / sampling / DSL / mining / streaming / async / RCA agent 等
  基础设施测试，框架无关
- **5 cross-framework integration**（在 `test_cross_framework.py` 里）：
  - DS→Megatron clip predicate migration
  - OLMo→Megatron structural predicate migration
  - OLMo-core→DS optim step monotonic migration
  - Megatron→OLMo-core no-nan-inf migration
  - paper §4.4 claim summary
- 其余 12 个零星引用 megatron/deepspeed/olmo 字样的 unit test（但 body 用的也是
  framework-flavoured surrogate，不是真框架）

**Runbook 想要的 framework-bucket 列（Megatron 32/32, OLMo 24/24…）数据不存在**。
所有 framework-flavoured 测试都是用 surrogate module 模拟 framework 风格，并不在
真实 Megatron / OLMo 仓库上跑 unit test。

### §6.6 Table 5 修改方案（待 user 决定）

| 方案 | 描述 |
|---|---|
| **A. 删 Tests pass 列** | 承认 framework-bucket pytest metric 不存在，Table 5 只留 T0/T1/Total bugs/Adapter LoC |
| **B. 换"Cross-framework migration pairs"** | 加一列 "Migration pairs covered" = 4 pairs (DS→Meg, OLMo→Meg, OC→DS, Meg→OC)，所有 4 pair test 全 pass |
| **C. 换"Adapter touchpoints exercised in suite"** | 列每个框架在测试套件里被引用的次数（megatron 7, deepspeed 3, olmo 5, olmo_core 2）—— 数字最弱、最容易被审稿人怀疑灌水 |

**推荐方案 B**：和 §4.4 已有 claim 对齐，全部 pair 都 pass，数字真实。
