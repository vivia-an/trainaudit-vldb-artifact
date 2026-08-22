# §6.2 Clean-run FP audit — 现状与补跑计划

## 现有可复用的 clean baseline（4 条）

见 `e_clean_baseline_summary.csv`。结论：
- 4 条 single-rank clean run（megatron_clean, megatron_moe, olmo_core_baseline, olmo_core_moe_hybrid）累计 **111,308 events / 0 fires / 0 FP**。
- Megatron 两条跑满 20 步；OLMo-core 两条只到 step-0（hunt 设计就只观测 init 阶段）。

## 不能复用的 sweep 行

- `olmo_core_moe_ep2_actckpt / olmo_core_moe_reordered_norm / olmo_core_olmo2_271M / olmo_core_tp2`：fires=0 是因为 **events=0**（hunt 跑挂或没写出 trace），不是真 clean。
- 其余 15 个 hunt trace 是 hunt 模式触发场景，fires>0 多数是真阳性（参见 doc 22 §2.1 13/14 真实 bug 检出记录）。

## §6.2 runbook 想要的并行配置（必须 GPU 补跑）

| Parallel config | 状态 |
|---|---|
| DP=8 | **需要 GPU 补跑** |
| TP=2/DP=4 | **需要 GPU 补跑** |
| TP=2/PP=2/DP=2 | **需要 GPU 补跑** |
| EP=2/DP=4 (MoE) | **需要 GPU 补跑** |
| FSDP zero3 | **需要 GPU 补跑** |

## 补跑方案（待 GPU 阶段执行）

复用 `benchmark/sweep/run_one.py`，加入下列 5 个 multi-rank 变体；每个 200 step × 3 repeat，
全部期望 **0 FP**：

```bash
# 在 eval-gpu-0 上跑
cd /volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025
for cfg in DP8 TP2DP4 TP2PP2DP2 EP2DP4 FSDP_ZERO3; do
  for repeat in 1 2 3; do
    bash benchmark/eval/rebuttal_v1/E_clean_run_fp_audit/run_${cfg}.sh $repeat
  done
done
```

驱动脚本待写：`benchmark/eval/rebuttal_v1/E_clean_run_fp_audit/run_{DP8,TP2DP4,...}.sh`。
