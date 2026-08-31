#!/bin/bash
#
# 测试 backward后DP参数bitwise-level一致性检查 约束的错误注入
#
# 约束名称：backward后DP参数bitwise-level一致性检查
# 用途：通过在 backward 后注入极小参数修改，验证 SDCCheck 约束检测系统能否检测 bitwise 不一致
# 预期：训练正常运行，但 cksum 校验会发现 bitwise-level 不一致
#
# 注意：
# 1. 注入发生在 schedules.py 的 backward 后、dump_model("model-after-backward") 之前
# 2. 使用 [corrupt-param-bitwise-after-bwd] 日志标识
# 3. 数据收集使用 dump_model("model-after-backward") 方法
# 4. 使用极小的 delta (1e-7) 破坏 bitwise 一致性但不影响训练
#

set -e

# 检查是否在正确的目录
if [ ! -f "pretrain_gpt.py" ]; then
    echo "错误：请在 Megatron-LM 根目录运行此脚本"
    exit 1
fi

echo "=========================================="
echo "测试 backward后 DP参数 bitwise-level 一致性错误注入"
echo "=========================================="

# 清理之前的数据
export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/param_bitwise_after_backward_test_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# ========================================
# 错误注入配置（param_bitwise_after_backward 注入）
# ========================================
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP=param_bitwise_after_backward
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_STEP=2  # 在第2步注入
export MEGATRON_CORRUPT_PARAM_SUBSTR="layers.0.mlp.linear_fc1.weight"
export MEGATRON_CORRUPT_DELTA=1e-7  # 极小值，破坏 bitwise 一致性但不影响训练

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
echo "  参数模式: ${MEGATRON_CORRUPT_PARAM_SUBSTR}"
echo "  修改量: ${MEGATRON_CORRUPT_DELTA}"
echo "  数据库路径: ${VTIMELINE_LOGGER_DIR}"
echo "  收集步数: ${VTIMELINE_DUMP_STEP}"
echo ""

# ========================================
# 约束血缘细节
# ========================================
echo "约束血缘细节："
echo "  约束名称: backward后DP参数bitwise-level一致性检查"
echo "  检测阶段: model-after-backward"
echo "  检测字段: cksum (参数的 bitwise checksum)"
echo "  约束条件: dp > 1"
echo ""
echo "  bitwise-level 检测说明:"
echo "    - 使用 cksum 对参数进行 bitwise 校验"
echo "    - 即使参数值变化极小 (1e-7)，cksum 也会完全不同"
echo "    - 可检测 NaN/Inf 注入、silent corruption、bit-flip 等问题"
echo ""
echo "  注入位置: schedules.py backward 后 → dump_model('model-after-backward') 之前"
echo "  注入方式: 对参数的第一个元素添加极小值 (${MEGATRON_CORRUPT_DELTA})"
echo ""

# ========================================
# 运行训练
# ========================================
echo "开始训练..."
echo "预期行为："
echo "  1. 训练正常运行（极小修改不影响收敛）"
echo "  2. 在 step ${MEGATRON_CORRUPT_STEP}，DP rank ${MEGATRON_CORRUPT_DP_RANK} 的参数被修改"
echo "  3. model-after-backward 阶段，DP rank 0 和 1 的 cksum 不一致"
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
    2>&1 | tee param_bitwise_after_backward_test.log || true

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "训练结束（退出码: ${EXIT_CODE}）"
echo "=========================================="

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
if [ -f "param_bitwise_after_backward_test.log" ]; then
    echo ""
    echo "--- Param Bitwise After Backward 注入日志 ---"
    grep "\[corrupt-param-bitwise-after-bwd\]" param_bitwise_after_backward_test.log | head -30 || echo "未找到注入日志"
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="
echo ""

# 创建验证脚本
cat > check_param_bitwise_after_backward.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/param_bitwise_after_backward_test_db") + "/Collector"

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

# 收集各 DP rank 的 cksum 数据
cksum_data = {}
target_stage = 'model-after-backward'

for dp in [0, 1]:
    print(f"=== DP Rank {dp} ===")
    db_path = f"{db_dir}/coredump_dp{dp}_tp0_pp0_cp0.db"
    conn = duckdb.connect(db_path, read_only=True)
    
    stages = conn.execute("SELECT DISTINCT stage FROM coredump").fetchall()
    stage_list = [s[0] for s in stages]
    print(f"  可用 stages: {stage_list}")
    
    if target_stage not in stage_list:
        print(f"  ⚠️  未找到目标 stage: {target_stage}")
        conn.close()
        continue
    
    result = conn.execute(f'''
        SELECT
            step,
            stage,
            json_extract(data, '$.name') as name,
            json_extract(data, '$.cksum') as cksum,
            json_extract(data, '$.requires_grad') as requires_grad
        FROM coredump
        WHERE stage = '{target_stage}'
        AND step = 2
        AND json_extract(data, '$.requires_grad') = true
        ORDER BY name
        LIMIT 30
    ''').fetchall()
    
    print(f"  {target_stage} (step=2) 的参数 cksum:")
    for row in result:
        step, stg, name, cksum, requires_grad = row
        print(f"    name={name}, cksum={cksum}")
        
        if name and cksum:
            key = (name, step)
            if key not in cksum_data:
                cksum_data[key] = {}
            cksum_data[key][dp] = cksum
    
    conn.close()
    print("")

# 比对
print("========================================")
print("验证结果比对")
print("========================================")

if not cksum_data:
    print("❌ 无法比对：未找到有效的 cksum 数据")
    sys.exit(1)

mismatch_count = 0
match_count = 0

for key in sorted(cksum_data.keys(), key=lambda x: str(x)):
    name, step = key
    dp_cksums = cksum_data[key]
    
    if len(dp_cksums) == 2:
        cksum_0 = dp_cksums.get(0)
        cksum_1 = dp_cksums.get(1)
        
        if cksum_0 != cksum_1:
            mismatch_count += 1
            print(f"❌ 不一致 - {name}")
            print(f"   DP0 cksum: {cksum_0}")
            print(f"   DP1 cksum: {cksum_1}")
        else:
            match_count += 1

print("")
print(f"✓ 一致的参数数量: {match_count}")
print(f"❌ 不一致的参数数量: {mismatch_count}")

if mismatch_count > 0:
    print("")
    print("🎉 验证成功：检测到注入导致的 bitwise 不一致！")
    print("   约束 'backward后DP参数bitwise-level一致性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("")
    print("⚠️  验证失败：所有参数 cksum 都一致，注入可能未生效")
    sys.exit(1)
PYTHON_EOF

echo "运行验证脚本..."
python check_param_bitwise_after_backward.py
VERIFY_EXIT_CODE=$?

rm -f check_param_bitwise_after_backward.py

echo ""
echo "关键信息："
echo "  - 约束名称: backward后DP参数bitwise-level一致性检查"
echo "  - 注入位置: schedules.py backward 后"
echo "  - 日志标识: [corrupt-param-bitwise-after-bwd]"
echo "  - 验证 stage: model-after-backward"
echo "  - 验证字段: cksum"
echo ""

