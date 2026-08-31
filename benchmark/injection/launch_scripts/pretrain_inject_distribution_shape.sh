#!/bin/bash
#
# 测试 DP参数分布形态一致性检查 约束的错误注入
#
# 约束名称：DP参数分布形态一致性检查
# 用途：通过修改参数分布破坏 histogram/entropy/skewness/kurtosis 一致性
# 预期：训练正常运行，但分布形态统计量在不同 DP rank 间不一致
#
# 注意：
# 1. 注入发生在 training.py 的 train_step() 中、dump_model() 之前
# 2. 使用 [corrupt-distribution-shape] 日志标识
# 3. 数据收集使用 dump_model() 方法
# 4. VTimeline 已扩展收集 histogram_cksum, entropy, skewness, kurtosis 字段
#

set -e

# 检查是否在正确的目录
if [ ! -f "pretrain_gpt.py" ]; then
    echo "错误：请在 Megatron-LM 根目录运行此脚本"
    exit 1
fi

echo "=========================================="
echo "测试 DP参数分布形态一致性错误注入"
echo "=========================================="

# 清理之前的数据
export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/distribution_shape_test_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# ========================================
# 错误注入配置（distribution_shape 注入）
# ========================================
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP=distribution_shape
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_STEP=2  # 在第2步注入
export MEGATRON_CORRUPT_PARAM_SUBSTR="layers.0.mlp.linear_fc1.weight"

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
echo "  数据库路径: ${VTIMELINE_LOGGER_DIR}"
echo "  收集步数: ${VTIMELINE_DUMP_STEP}"
echo ""

# ========================================
# 约束血缘细节
# ========================================
echo "约束血缘细节："
echo "  约束名称: DP参数分布形态一致性检查"
echo "  检测阶段: model-after-backward, model-before-optimizer-step, model-after-optimizer-step"
echo "  检测字段: histogram_cksum, entropy, skewness, kurtosis"
echo "  约束条件: dp > 1"
echo ""
echo "  分布形态统计量定义:"
echo "    - histogram_cksum: 64-bin 直方图的 checksum"
echo "    - entropy: 信息熵 -sum(p * log(p))"
echo "    - skewness: 偏度 E[(X-μ)³] / σ³"
echo "    - kurtosis: 峰度 E[(X-μ)⁴] / σ⁴ - 3"
echo ""
echo "  注入方式: 将参数前 20% 元素置为正值，改变分布形态"
echo "           这会改变 histogram/entropy/skewness/kurtosis 但可能不改变 mean/std"
echo ""

# ========================================
# 运行训练
# ========================================
echo "开始训练..."
echo "预期行为："
echo "  1. 训练正常运行"
echo "  2. 在 step ${MEGATRON_CORRUPT_STEP}，DP rank ${MEGATRON_CORRUPT_DP_RANK} 的参数分布被修改"
echo "  3. 相关阶段，DP rank 0 和 1 的分布形态统计量不一致"
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
    2>&1 | tee distribution_shape_injection_test.log || true

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "训练结束（退出码: ${EXIT_CODE}）"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 训练正常完成"
else
    echo "⚠ 训练失败（退出码: ${EXIT_CODE}）"
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
if [ -f "distribution_shape_injection_test.log" ]; then
    echo ""
    echo "--- Distribution Shape 注入日志 ---"
    grep -E "\[corrupt-distribution-shape\]|\[corrupt-higher-order-stats\]" distribution_shape_injection_test.log | head -30 || echo "未找到注入日志"
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="
echo ""

# 创建验证脚本
cat > check_distribution_shape_injection.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/distribution_shape_test_db") + "/Collector"

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

# 收集各 DP rank 的分布形态数据
# 结构: {(name, step, stage): {dp_rank: {histogram_cksum, entropy, skewness, kurtosis}}}
shape_data = {}

# 目标阶段
target_stages = ['model-after-backward', 'model-before-optimizer-step', 'model-after-optimizer-step']

for dp in [0, 1]:
    print(f"=== DP Rank {dp} ===")
    db_path = f"{db_dir}/coredump_dp{dp}_tp0_pp0_cp0.db"
    conn = duckdb.connect(db_path, read_only=True)
    
    # 检查有哪些 stage
    stages = conn.execute("SELECT DISTINCT stage FROM coredump").fetchall()
    stage_list = [s[0] for s in stages]
    print(f"  可用 stages: {stage_list}")
    
    # 查找目标 stage
    available_stages = [s for s in stage_list if s in target_stages]
    print(f"  目标 stages: {available_stages}")
    
    for stage in available_stages:
        result = conn.execute(f'''
            SELECT
                step,
                stage,
                json_extract(data, '$.name') as name,
                json_extract(data, '$.histogram_cksum') as histogram_cksum,
                json_extract(data, '$.entropy') as entropy,
                json_extract(data, '$.skewness') as skewness,
                json_extract(data, '$.kurtosis') as kurtosis
            FROM coredump
            WHERE stage = '{stage}'
            AND step = 2
            AND json_extract(data, '$.requires_grad') = true
            ORDER BY name
            LIMIT 20
        ''').fetchall()
        
        print(f"  {stage} (step=2) 的分布形态数据:")
        for row in result:
            step, stg, name, hist_cksum, entropy, skewness, kurtosis = row
            print(f"    name={name}")
            print(f"      histogram_cksum={hist_cksum}, entropy={entropy}")
            print(f"      skewness={skewness}, kurtosis={kurtosis}")
            
            if name:
                key = (name, step, stage)
                if key not in shape_data:
                    shape_data[key] = {}
                shape_data[key][dp] = {
                    'histogram_cksum': hist_cksum,
                    'entropy': float(entropy) if entropy else None,
                    'skewness': float(skewness) if skewness else None,
                    'kurtosis': float(kurtosis) if kurtosis else None
                }
    
    conn.close()
    print("")

# 比对两个 DP rank 的分布形态
print("========================================")
print("验证结果比对")
print("========================================")

if not shape_data:
    print("❌ 无法比对：未找到有效的分布形态数据")
    sys.exit(1)

total_mismatch = 0
total_match = 0
metrics = ['histogram_cksum', 'entropy', 'skewness', 'kurtosis']

for key in sorted(shape_data.keys(), key=lambda x: str(x)):
    name, step, stage = key
    dp_data = shape_data[key]
    
    if len(dp_data) == 2:
        data_0 = dp_data.get(0, {})
        data_1 = dp_data.get(1, {})
        
        mismatches = []
        for metric in metrics:
            v0 = data_0.get(metric)
            v1 = data_1.get(metric)
            
            if v0 is not None and v1 is not None:
                if metric == 'histogram_cksum':
                    # 整数比较
                    if v0 != v1:
                        mismatches.append(f"{metric}: DP0={v0}, DP1={v1}")
                else:
                    # 浮点数比较
                    if abs(v0 - v1) > 1e-6:
                        mismatches.append(f"{metric}: DP0={v0:.6f}, DP1={v1:.6f}")
        
        if mismatches:
            total_mismatch += 1
            print(f"❌ 不一致 - {name} ({stage})")
            for m in mismatches:
                print(f"   {m}")
        else:
            total_match += 1

print("")
print(f"✓ 一致的参数数量: {total_match}")
print(f"❌ 不一致的参数数量: {total_mismatch}")

if total_mismatch > 0:
    print("")
    print("🎉 验证成功：检测到注入导致的分布形态不一致！")
    print("   约束 'DP参数分布形态一致性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("")
    print("⚠️  验证失败：所有分布形态统计量都一致，注入可能未生效")
    print("")
    print("请检查：")
    print("  1. MEGATRON_CORRUPT_STEP 是否与 current_step 匹配")
    print("  2. 注入代码是否被调用")
    print("  3. VTimeline 是否收集了 histogram_cksum 和 entropy")
    sys.exit(1)
PYTHON_EOF

echo "运行验证脚本..."
python check_distribution_shape_injection.py
VERIFY_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="

if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "✅ 注入实验成功！"
    echo "   成功检测到 DP rank 间的分布形态不一致"
else
    echo "⚠️  注入实验可能未成功"
    echo "   请检查日志和数据库内容"
fi

# 清理临时文件
rm -f check_distribution_shape_injection.py

echo ""
echo "关键信息："
echo "  - 约束名称: DP参数分布形态一致性检查"
echo "  - 注入位置: training.py::train_step()"
echo "  - 日志标识: [corrupt-distribution-shape]"
echo "  - 数据收集: dump_model()"
echo "  - 验证 stages: model-after-backward, model-before-optimizer-step, model-after-optimizer-step"
echo "  - 验证字段: histogram_cksum, entropy, skewness, kurtosis"
echo ""

