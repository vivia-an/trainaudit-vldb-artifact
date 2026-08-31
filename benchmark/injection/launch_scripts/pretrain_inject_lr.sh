#!/bin/bash
#
# 测试 optimizer前DP optimizer_state_dict.lr一致性检查 约束的错误注入
#
# 约束名称：optimizer前DP optimizer_state_dict.lr一致性检查
# 用途：通过在 optimizer step 之前修改 lr，验证 SDCCheck 约束检测系统能否检测不一致
# 预期：训练继续运行但可能出现训练分歧，lr 在不同 DP rank 间不一致
#
# 注意：
# 1. 注入发生在 training.py 的 train_step() 中、optimizer.step() 之前
# 2. 使用 [corrupt-lr] 日志标识
# 3. 数据收集使用 dump_optimizer_state("optimizer-state-before-optimizer-step") 方法
# 4. VTimeline 已扩展收集 lr 字段
#

set -e

# 检查是否在正确的目录
if [ ! -f "pretrain_gpt.py" ]; then
    echo "错误：请在 Megatron-LM 根目录运行此脚本"
    exit 1
fi

echo "=========================================="
echo "测试 optimizer前 DP optimizer_state_dict.lr 一致性错误注入"
echo "=========================================="

# 清理之前的数据
export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/lr_test_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# ========================================
# 错误注入配置（lr 注入）
# ========================================
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP=lr
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_STEP=2  # 在第2步注入（因为 MegatronCollector.step_ 在 train_step 开始时递增）
export MEGATRON_CORRUPT_LR_DELTA=0.001  # lr 修改量

# ========================================
# VTimeline 数据收集配置
# ========================================
export VTIMELINE_DUMP_STEP=5  # 收集前5步

# ========================================
# Megatron-LM 环境配置
# ========================================
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo ""
echo "配置信息："
echo "  错误注入: ${MEGATRON_INJECT_PARAM_CORRUPTION}"
echo "  操作类型: ${MEGATRON_CORRUPT_OP}"
echo "  目标DP rank: ${MEGATRON_CORRUPT_DP_RANK}"
echo "  注入步数: ${MEGATRON_CORRUPT_STEP}"
echo "  LR修改量: ${MEGATRON_CORRUPT_LR_DELTA}"
echo "  数据库路径: ${VTIMELINE_LOGGER_DIR}"
echo "  收集步数: ${VTIMELINE_DUMP_STEP}"
echo ""

# ========================================
# 运行训练
# ========================================
echo "开始训练..."
echo "预期行为："
echo "  1. 训练正常运行（lr 不一致不会导致崩溃）"
echo "  2. 在 step 2，DP rank 0 的 lr 被修改（增加 ${MEGATRON_CORRUPT_LR_DELTA}）"
echo "  3. optimizer-state-before-optimizer-step 阶段，DP rank 0 和 1 的 lr 不一致"
echo ""
echo "日志标识："
echo "  [corrupt-lr] - training.py 中的 lr 注入日志"
echo ""

# 使用 || true 防止脚本因训练失败而退出
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 12 \
    --hidden-size 128 \
    --num-attention-heads 4 \
    --seq-length 64 \
    --max-position-embeddings 512 \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --rotary-percent 1.0 \
    --attention-dropout 0.1 \
    --hidden-dropout 0.1 \
    --swiglu \
    --init-method-std 0.02 \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --train-iters 5 \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.1 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --grad-reduce-in-bf16 \
    --cross-entropy-loss-fusion \
    --calculate-per-token-loss \
    --manual-gc \
    --empty-unused-memory-level 1 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 50257 \
    --data-cache-path ./data_cache \
    --split 99,1,0 \
    --no-create-attention-mask-in-dataloader \
    --no-mmap-bin-files \
    --num-workers 1 \
    --log-interval 1 \
    --eval-iters 10 \
    --eval-interval 100 \
    --save-interval 250 \
    --log-throughput \
    --ckpt-format torch_dist \
    --distributed-timeout-minutes 30 \
    --no-gradient-accumulation-fusion \
    2>&1 | tee lr_injection_test.log || true

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "训练结束（退出码: ${EXIT_CODE}）"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 训练正常完成"
    echo "  lr 注入不会导致崩溃，但 lr 不一致已被记录"
else
    echo "⚠ 训练失败（退出码: ${EXIT_CODE}）"
    echo "  请检查日志查看失败原因"
fi

echo ""
echo "检查数据库文件..."
if [ -d "$VTIMELINE_LOGGER_DIR/Collector" ]; then
    ls -lh $VTIMELINE_LOGGER_DIR/Collector/ | grep -E "\.db$" || echo "未找到数据库文件"
    DB_COUNT=$(ls $VTIMELINE_LOGGER_DIR/Collector/*.db 2>/dev/null | wc -l)
    echo "找到 $DB_COUNT 个数据库文件"
else
    echo "数据库目录不存在：$VTIMELINE_LOGGER_DIR/Collector"
fi

echo ""
echo "检查日志中的错误注入信息..."
if [ -f "lr_injection_test.log" ]; then
    echo ""
    echo "--- LR 注入日志（[corrupt-lr]）---"
    grep "\[corrupt-lr\]" lr_injection_test.log | head -30
    
    if [ $? -ne 0 ]; then
        echo "⚠️  未找到 [corrupt-lr] 日志"
    fi
    echo ""
    
    # 检查是否有 LR_CHANGED 成功标识
    if grep -q "LR_CHANGED" lr_injection_test.log; then
        echo "✓ 检测到 LR_CHANGED 成功标识"
        grep "LR_CHANGED" lr_injection_test.log | head -5
    fi
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="
echo ""

# 创建验证脚本
cat > check_lr_injection.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/lr_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")
print("")

# 检查数据库文件是否存在
db0_path = f"{db_dir}/coredump_dp0_tp0_pp0_cp0.db"
db1_path = f"{db_dir}/coredump_dp1_tp0_pp0_cp0.db"

if not os.path.exists(db0_path):
    print(f"❌ 数据库不存在: {db0_path}")
    sys.exit(1)
if not os.path.exists(db1_path):
    print(f"❌ 数据库不存在: {db1_path}")
    sys.exit(1)

# 收集各 DP rank 的 lr 数据
# 使用 param_group_idx 作为 key（因为分布式优化器下不同 rank 持有不同参数）
# 结构: {(param_group_idx, step): {dp_rank: lr}}
lr_by_param_group = {}

for dp in [0, 1]:
    print(f"=== DP Rank {dp} ===")
    db_path = f"{db_dir}/coredump_dp{dp}_tp0_pp0_cp0.db"
    conn = duckdb.connect(db_path, read_only=True)
    
    # 检查有哪些 stage
    stages = conn.execute("SELECT DISTINCT stage FROM coredump").fetchall()
    stage_list = [s[0] for s in stages]
    print(f"  可用 stages: {stage_list}")
    
    # 查找 optimizer-state 相关的 stage
    optim_stages = [s for s in stage_list if 'optimizer-state' in s]
    print(f"  optimizer-state 相关 stages: {optim_stages}")
    
    if not optim_stages:
        print("  ⚠️  未找到 optimizer-state 相关的 stage")
        conn.close()
        continue
    
    # 优先使用 optimizer-state-before-optimizer-step
    target_stage = None
    for s in optim_stages:
        if 'before-optimizer-step' in s:
            target_stage = s
            break
    if not target_stage and optim_stages:
        target_stage = optim_stages[0]
    
    if target_stage:
        result = conn.execute(f'''
            SELECT
                step,
                stage,
                json_extract(data, '$.name') as name,
                json_extract(data, '$.lr') as lr,
                json_extract(data, '$.param_group_idx') as param_group_idx
            FROM coredump
            WHERE stage = '{target_stage}'
            AND step = 2
            ORDER BY step, name
            LIMIT 30
        ''').fetchall()
        
        print(f"  {target_stage} (step=2) 的数据:")
        if not result:
            print("    ⚠️  未找到数据")
            steps = conn.execute(f"SELECT DISTINCT step FROM coredump WHERE stage = '{target_stage}'").fetchall()
            print(f"    该 stage 可用 steps: {[s[0] for s in steps]}")
        else:
            for row in result:
                step, stg, name, lr, pg_idx = row
                print(f"    step={step}, name={name}, lr={lr}, param_group_idx={pg_idx}")
                
                if lr is not None and pg_idx is not None:
                    # 使用 param_group_idx 作为 key
                    key = (int(pg_idx), step)
                    if key not in lr_by_param_group:
                        lr_by_param_group[key] = {}
                    # 保存该 rank 的 lr（如果同一 param_group 有多个参数，lr 应该相同）
                    lr_by_param_group[key][dp] = float(lr)
    
    conn.close()
    print("")

# 比对两个 DP rank 的 lr（按 param_group_idx）
print("========================================")
print("验证结果比对（按 param_group_idx）")
print("========================================")
print("")
print("说明：使用分布式优化器时，不同 DP rank 持有不同参数，")
print("      但同一 param_group 的 lr 应该在所有 rank 间一致。")
print("")

if not lr_by_param_group:
    print("❌ 无法比对：未找到有效的 lr 数据")
    sys.exit(1)

mismatch_count = 0
match_count = 0

for key in sorted(lr_by_param_group.keys()):
    param_group_idx, step = key
    dp_lrs = lr_by_param_group[key]
    
    if len(dp_lrs) == 2:
        lr_0 = dp_lrs.get(0)
        lr_1 = dp_lrs.get(1)
        
        if lr_0 is not None and lr_1 is not None:
            diff = abs(lr_0 - lr_1)
            if diff > 1e-8:
                mismatch_count += 1
                print(f"❌ 不一致 - param_group={param_group_idx}, step={step}")
                print(f"   DP0 lr: {lr_0}")
                print(f"   DP1 lr: {lr_1}")
                print(f"   差异: {diff}")
                print(f"   差异倍数: {lr_0/lr_1 if lr_1 != 0 else 'inf'}x")
            else:
                match_count += 1
                print(f"✓ 一致 - param_group={param_group_idx}, step={step}: lr={lr_0}")

print("")
print(f"✓ 一致的 param_group 数量: {match_count}")
print(f"❌ 不一致的 param_group 数量: {mismatch_count}")

if mismatch_count > 0:
    print("")
    print("🎉 验证成功：检测到注入导致的 lr 不一致！")
    print("   约束 'optimizer前DP optimizer_state_dict.lr一致性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("")
    print("⚠️  验证失败：所有 param_group 的 lr 都一致，注入可能未生效")
    print("")
    print("请检查：")
    print("  1. MEGATRON_CORRUPT_STEP 是否与 current_step 匹配")
    print("  2. 注入代码 _inject_lr_corruption 是否被调用")
    print("  3. dump_optimizer_state 是否在注入之后被调用")
    sys.exit(1)
PYTHON_EOF

echo "运行验证脚本..."
python check_lr_injection.py
VERIFY_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="

if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "✅ 注入实验成功！"
    echo "   成功检测到 DP rank 间的 lr 不一致"
else
    echo "⚠️  注入实验可能未成功"
    echo "   请检查日志和数据库内容"
fi

# 清理临时文件
rm -f check_lr_injection.py

echo ""
echo "关键信息："
echo "  - 约束名称: optimizer前DP optimizer_state_dict.lr一致性检查"
echo "  - 注入位置: training.py::train_step() optimizer.step() 之前"
echo "  - 日志标识: [corrupt-lr]"
echo "  - 数据收集: dump_optimizer_state()"
echo "  - 验证 stage: optimizer-state-before-optimizer-step"
echo "  - 验证字段: lr"
echo ""
