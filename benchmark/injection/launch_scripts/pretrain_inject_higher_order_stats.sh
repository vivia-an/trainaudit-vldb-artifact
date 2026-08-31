#!/bin/bash
#
# 测试 model-before-optimizer-step阶段DP参数分布高阶统计量一致性检查 约束的错误注入
#
# 约束名称：model-before-optimizer-step阶段DP参数分布高阶统计量一致性检查
# 用途：通过修改参数分布破坏偏度/峰度一致性，验证 SDCCheck 约束检测系统能否检测不一致
# 预期：训练正常运行，但 skewness/kurtosis 在不同 DP rank 间不一致
#
# 注意：
# 1. 注入发生在 training.py 的 train_step() 中、dump_model() 之前
# 2. 使用 [corrupt-higher-order-stats] 日志标识
# 3. 数据收集使用 dump_model("model-before-optimizer-step") 方法
# 4. VTimeline 已扩展收集 skewness 和 kurtosis 字段
#

set -e

# 检查是否在正确的目录
if [ ! -f "pretrain_gpt.py" ]; then
    echo "错误：请在 Megatron-LM 根目录运行此脚本"
    exit 1
fi

echo "=========================================="
echo "测试 DP参数分布高阶统计量一致性错误注入"
echo "=========================================="

# 清理之前的数据
export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/higher_order_stats_test_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# ========================================
# 错误注入配置（higher_order_stats 注入）
# ========================================
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP=higher_order_stats
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_STEP=1  # 在第1步注入
export MEGATRON_CORRUPT_PARAM_SUBSTR="layers.0.mlp.linear_fc1.weight"
export MEGATRON_CORRUPT_SKEW_DELTA=0.1  # 偏度修改量

# ========================================
# VTimeline 数据收集配置
# ========================================
export VTIMELINE_DUMP_STEP=3  # 收集前3步

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
echo "  偏度修改量: ${MEGATRON_CORRUPT_SKEW_DELTA}"
echo "  数据库路径: ${VTIMELINE_LOGGER_DIR}"
echo "  收集步数: ${VTIMELINE_DUMP_STEP}"
echo ""

# ========================================
# 约束血缘细节
# ========================================
echo "约束血缘细节："
echo "  约束名称: model-before-optimizer-step阶段DP参数分布高阶统计量一致性检查"
echo "  检测阶段: model-before-optimizer-step"
echo "  检测字段: skewness (偏度), kurtosis (峰度)"
echo "  约束条件: dp > 1"
echo ""
echo "  高阶统计量定义:"
echo "    - 偏度 (Skewness): 衡量分布的不对称性"
echo "      公式: E[(X-μ)³] / σ³"
echo "      值为 0 表示对称分布"
echo "    - 峰度 (Kurtosis): 衡量分布的尖锐程度"
echo "      公式: E[(X-μ)⁴] / σ⁴ - 3 (Fisher's definition)"
echo "      值为 0 表示正态分布"
echo ""
echo "  注入位置: training.py::train_step() → dump_model('model-before-optimizer-step') 之前"
echo "  注入方式: 修改参数分布的前 10% 元素，添加偏移以改变偏度/峰度"
echo ""
echo "  数据流:"
echo "    1. train_step() 开始"
echo "    2. [注入点] _inject_higher_order_stats_corruption() 修改参数分布"
echo "    3. dump_model('model-before-optimizer-step') 收集 skewness/kurtosis"
echo "    4. 不同 DP rank 的高阶统计量不一致"
echo ""

# ========================================
# 运行训练
# ========================================
echo "开始训练..."
echo "预期行为："
echo "  1. 训练正常运行"
echo "  2. 在 step ${MEGATRON_CORRUPT_STEP}，DP rank ${MEGATRON_CORRUPT_DP_RANK} 的参数分布被修改"
echo "  3. model-before-optimizer-step 阶段，DP rank 0 和 1 的 skewness/kurtosis 不一致"
echo ""
echo "日志标识："
echo "  [corrupt-higher-order-stats] - training.py 中的注入日志"
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
    2>&1 | tee higher_order_stats_injection_test.log || true

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
if [ -f "higher_order_stats_injection_test.log" ]; then
    echo ""
    echo "--- 高阶统计量注入日志（[corrupt-higher-order-stats]）---"
    grep "\[corrupt-higher-order-stats\]" higher_order_stats_injection_test.log | head -30
    
    if [ $? -ne 0 ]; then
        echo "⚠️  未找到 [corrupt-higher-order-stats] 日志"
    fi
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="
echo ""

# 创建验证脚本
cat > check_higher_order_stats_injection.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/higher_order_stats_test_db") + "/Collector"

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
inject_step = int(os.environ.get("MEGATRON_CORRUPT_STEP", "1"))

for dp in [0, 1]:
    print(f"=== DP Rank {dp} ===")
    db_path = f"{db_dir}/coredump_dp{dp}_tp0_pp0_cp0.db"
    conn = duckdb.connect(db_path, read_only=True)
    
    # 先查看有哪些 stage
    all_stages = conn.execute("SELECT DISTINCT stage FROM coredump").fetchall()
    print(f"可用 stages: {[s[0] for s in all_stages]}")
    
    # 查询 model-before-optimizer-step 阶段的数据
    target_stage = 'model-before-optimizer-step'
    result = conn.execute(f'''
        SELECT step, stage,
               json_extract(data, '$.name') as name,
               json_extract(data, '$.skewness') as skewness,
               json_extract(data, '$.kurtosis') as kurtosis
        FROM coredump 
        WHERE stage = '{target_stage}'
          AND json_extract(data, '$.name') LIKE '%linear_fc1%'
          AND step = {inject_step}
        LIMIT 10
    ''').fetchall()
    
    if not result:
        print(f"  ⚠️  未找到 {target_stage} 阶段的数据")
    else:
        print(f"使用 stage: {target_stage}")
        for row in result:
            print(f"  step={row[0]}, stage={row[1]}")
            print(f"    name={row[2]}")
            print(f"    skewness={row[3]}, kurtosis={row[4]}")
        
        # 保存结果用于比对
        results[dp] = {row[2]: {'skewness': row[3], 'kurtosis': row[4]} for row in result}
    
    conn.close()
    print("")

# 比对两个 DP rank 的高阶统计量
print("========================================")
print("验证结果比对")
print("========================================")

if 0 in results and 1 in results:
    all_keys = set(results[0].keys()) & set(results[1].keys())
    
    mismatch_count = 0
    match_count = 0
    
    for key in sorted(all_keys):
        stats0 = results[0].get(key, {})
        stats1 = results[1].get(key, {})
        
        skew0 = stats0.get('skewness')
        skew1 = stats1.get('skewness')
        kurt0 = stats0.get('kurtosis')
        kurt1 = stats1.get('kurtosis')
        
        # 检查是否有不一致（允许微小误差）
        has_mismatch = False
        if skew0 is not None and skew1 is not None:
            try:
                if abs(float(skew0) - float(skew1)) > 1e-4:
                    has_mismatch = True
            except:
                pass
        if kurt0 is not None and kurt1 is not None:
            try:
                if abs(float(kurt0) - float(kurt1)) > 1e-4:
                    has_mismatch = True
            except:
                pass
        
        if has_mismatch:
            mismatch_count += 1
            print(f"❌ 不一致: {key}")
            print(f"   DP0: skewness={skew0}, kurtosis={kurt0}")
            print(f"   DP1: skewness={skew1}, kurtosis={kurt1}")
        else:
            match_count += 1
    
    print("")
    print(f"✓ 一致的参数数量: {match_count}")
    print(f"❌ 不一致的参数数量: {mismatch_count}")
    
    if mismatch_count > 0:
        print("")
        print("🎉 验证成功：检测到注入导致的高阶统计量不一致！")
        sys.exit(0)
    else:
        print("")
        print("⚠️  验证失败：所有高阶统计量都一致，注入可能未生效")
        sys.exit(1)
else:
    print("❌ 无法比对：缺少一个或多个 DP rank 的数据")
    sys.exit(1)
PYTHON_EOF

echo "运行验证脚本..."
python check_higher_order_stats_injection.py
VERIFY_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="

if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "✅ 注入实验成功！"
    echo "   成功检测到 DP rank 间的高阶统计量不一致"
else
    echo "⚠️  注入实验可能未成功"
    echo "   请检查日志和数据库内容"
fi

# 清理临时文件
rm -f check_higher_order_stats_injection.py

echo ""
echo "=========================================="
echo "血缘细节总结"
echo "=========================================="
echo "  约束名称: model-before-optimizer-step阶段DP参数分布高阶统计量一致性检查"
echo "  检测阶段: model-before-optimizer-step"
echo "  检测字段: skewness (偏度), kurtosis (峰度)"
echo "  约束条件: dp > 1"
echo ""
echo "  注入位置: training.py::train_step() → dump_model() 之前"
echo "  日志标识: [corrupt-higher-order-stats]"
echo "  数据收集: MegatronCollector.dump_model('model-before-optimizer-step')"
echo ""
echo "  VTimeline 修改: megatron_collector.py::dump_model() 添加了 skewness/kurtosis 收集"
echo "  Megatron-LM 修改: training.py 添加了 _inject_higher_order_stats_corruption() 函数"
echo ""

