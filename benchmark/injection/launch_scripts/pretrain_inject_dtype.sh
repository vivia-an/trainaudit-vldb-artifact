#!/bin/bash
#
# 测试 DP 参数 dtype 一致性约束的错误注入
#
# 用途：通过注入 dtype 不一致错误，验证 SDCCheck 约束检测系统能否正确识别
# 预期：训练可能崩溃（NCCL 错误）或产生 silent error
#
# 注意：
# 1. 注入发生在 schedules.py 的 backward_step() 之后
# 2. 使用 [corrupt-dtype] 日志标识
# 3. 数据收集使用 dump_model() 方法
#

set -e

# 检查是否在正确的目录
if [ ! -f "pretrain_gpt.py" ]; then
    echo "错误：请在 Megatron-LM 根目录运行此脚本"
    exit 1
fi

echo "=========================================="
echo "测试 DP 参数 dtype 一致性错误注入"
echo "=========================================="

# 清理之前的数据
export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/dtype_test_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# ========================================
# 错误注入配置（dtype 注入）
# ========================================
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP=dtype
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_STEP=1  # 在第1步注入
export MEGATRON_CORRUPT_PARAM_SUBSTR="layers.0.mlp.linear_fc1.weight"
export MEGATRON_TARGET_DTYPE=float32  # 目标 dtype: float32 | float16 | bfloat16

# ========================================
# VTimeline 数据收集配置
# ========================================
export VTIMELINE_DUMP_STEP=2  # 收集前2步

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
echo "  目标dtype: ${MEGATRON_TARGET_DTYPE}"
echo "  数据库路径: ${VTIMELINE_LOGGER_DIR}"
echo "  收集步数: ${VTIMELINE_DUMP_STEP}"
echo ""

# ========================================
# 运行训练
# ========================================
echo "开始训练..."
echo "预期行为："
echo "  1. 训练可能正常运行或崩溃（取决于 NCCL 对 dtype 不一致的处理）"
echo "  2. 在 step 1，DP rank 0 的某参数 dtype 被修改"
echo "  3. model-after-backward 阶段，DP rank 0 和 1 的 dtype 不一致"
echo ""
echo "日志标识："
echo "  [corrupt-dtype] - schedules.py 中的 dtype 注入日志"
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
    2>&1 | tee dtype_injection_test.log || true

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "训练结束（退出码: ${EXIT_CODE}）"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 训练正常完成"
    echo "  dtype 注入可能未导致崩溃，但 dtype 不一致已被记录"
else
    echo "⚠ 训练失败（退出码: ${EXIT_CODE}）"
    echo "  这可能是因为 dtype 不一致导致的 NCCL 错误"
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
if [ -f "dtype_injection_test.log" ]; then
    echo ""
    echo "--- dtype 注入日志（[corrupt-dtype]）---"
    grep "\[corrupt-dtype\]" dtype_injection_test.log | head -30
    
    if [ $? -ne 0 ]; then
        echo "⚠️  未找到 [corrupt-dtype] 日志"
    fi
    echo ""
    
    # 检查是否有 DTYPE_CHANGED 成功标识
    if grep -q "DTYPE_CHANGED" dtype_injection_test.log; then
        echo "✓ 检测到 DTYPE_CHANGED 成功标识"
        grep "DTYPE_CHANGED" dtype_injection_test.log | head -5
    fi
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="
echo ""

# 创建验证脚本
cat > check_dtype_injection.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/dtype_test_db") + "/Collector"

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

# 存储两个 DP rank 的结果用于比对
results = {}

for dp in [0, 1]:
    print(f"=== DP Rank {dp} ===")
    db_path = f"{db_dir}/coredump_dp{dp}_tp0_pp0_cp0.db"
    conn = duckdb.connect(db_path, read_only=True)
    
    # 查询 model-after-backward 阶段的数据
    # VTimeline dump_model 存储的字段是 "type"（包含 dtype 信息）
    result = conn.execute('''
        SELECT step, stage,
               json_extract(data, '$.name') as name,
               json_extract(data, '$.type') as dtype
        FROM coredump 
        WHERE stage = 'model-after-backward'
          AND json_extract(data, '$.name') LIKE '%layers.0.mlp.linear_fc1%'
          AND step = 1
    ''').fetchall()
    
    if not result:
        print("  ⚠️  未找到 model-after-backward 数据")
        # 尝试查询所有 stage
        all_stages = conn.execute("SELECT DISTINCT stage FROM coredump").fetchall()
        print(f"  可用的 stages: {[s[0] for s in all_stages]}")
    else:
        for row in result:
            print(f"  step={row[0]}, stage={row[1]}")
            print(f"    name={row[2]}")
            print(f"    dtype={row[3]}")
        
        # 保存结果用于比对
        results[dp] = {row[2]: row[3] for row in result}
    
    conn.close()
    print("")

# 比对两个 DP rank 的 dtype
print("========================================")
print("验证结果比对")
print("========================================")

if 0 in results and 1 in results:
    all_keys = set(results[0].keys()) | set(results[1].keys())
    
    mismatch_count = 0
    match_count = 0
    
    for key in sorted(all_keys):
        dtype0 = results[0].get(key, "N/A")
        dtype1 = results[1].get(key, "N/A")
        
        if dtype0 != dtype1:
            mismatch_count += 1
            print(f"❌ 不一致: {key}")
            print(f"   DP0 dtype: {dtype0}")
            print(f"   DP1 dtype: {dtype1}")
        else:
            match_count += 1
    
    print("")
    print(f"✓ 一致的参数数量: {match_count}")
    print(f"❌ 不一致的参数数量: {mismatch_count}")
    
    if mismatch_count > 0:
        print("")
        print("🎉 验证成功：检测到注入导致的 dtype 不一致！")
        sys.exit(0)
    else:
        print("")
        print("⚠️  验证失败：所有 dtype 都一致，注入可能未生效")
        sys.exit(1)
else:
    print("❌ 无法比对：缺少一个或多个 DP rank 的数据")
    sys.exit(1)
PYTHON_EOF

echo "运行验证脚本..."
python check_dtype_injection.py
VERIFY_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="

if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "✅ 注入实验成功！"
    echo "   成功检测到 DP rank 间的 dtype 不一致"
else
    echo "⚠️  注入实验可能未成功"
    echo "   请检查日志和数据库内容"
fi

# 清理临时文件
rm -f check_dtype_injection.py

echo ""
echo "关键信息："
echo "  - 约束名称: backward后DP参数dtype一致性检查"
echo "  - 注入位置: schedules.py::backward_step() 之后"
echo "  - 日志标识: [corrupt-dtype]"
echo "  - 数据收集: dump_model()"
echo "  - 验证 stage: model-after-backward"
echo "  - 验证字段: type (包含 dtype 信息)"
echo ""


