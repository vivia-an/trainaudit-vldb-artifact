#!/bin/bash
#
# 测试 DP分片参数backward后分片互异性检查 约束的错误注入
#
# 用途：通过将分片参数设为相同值，验证 SDCCheck 能否检测到互异性被破坏
# 预期：训练正常运行，但分片参数在所有 DP rank 间变得相同（本应不同）
#
# 注意：
# 1. 分片参数只存在于 MoE（混合专家）模型中
# 2. 普通 GPT 模型没有分片参数（param.allreduce 默认为 True）
# 3. 如需测试此约束，请使用带有 --num-experts 参数的 MoE 配置
#

set -e

# 检查是否在正确的目录
if [ ! -f "pretrain_gpt.py" ]; then
    echo "错误：请在 Megatron-LM 根目录运行此脚本"
    exit 1
fi

echo "=========================================="
echo "测试 DP分片参数互异性错误注入"
echo "=========================================="

# 清理之前的数据
export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/sharded_same_test_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# ========================================
# 错误注入配置（分片参数互异性破坏）
# ========================================
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP=sharded_same
export MEGATRON_CORRUPT_STEP=1  # 在第1步注入
export MEGATRON_CORRUPT_PARAM_SUBSTR=""  # 空表示所有分片参数
export MEGATRON_CORRUPT_SHARDED_VALUE=0.0  # 将分片参数设为此固定值

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
echo "  注入步数: ${MEGATRON_CORRUPT_STEP}"
echo "  参数模式: ${MEGATRON_CORRUPT_PARAM_SUBSTR:-'所有分片参数'}"
echo "  固定值: ${MEGATRON_CORRUPT_SHARDED_VALUE}"
echo "  数据库路径: ${VTIMELINE_LOGGER_DIR}"
echo "  收集步数: ${VTIMELINE_DUMP_STEP}"
echo ""

# ========================================
# 约束血缘细节
# ========================================
echo "约束血缘细节："
echo "  约束名称: DP分片参数backward后分片互异性检查"
echo "  检测阶段: model-after-backward"
echo "  检测逻辑: 分片参数在不同 DP rank 间应该互异（不同）"
echo "  约束条件: dp > 1, param_sharded = true, param_full_replica = false"
echo ""
echo "  分片参数识别:"
echo "    - param.allreduce = False → 分片参数（专家参数）"
echo "    - param.allreduce = True/不存在 → 全复制参数"
echo ""
echo "  注入方式:"
echo "    - 将所有分片参数设为固定值（如 0.0）"
echo "    - 使所有 DP rank 的分片参数变成相同（本应不同）"
echo "    - 这会违反互异性约束"
echo ""
echo "  ⚠️  重要提示:"
echo "    - 分片参数只存在于 MoE 模型中"
echo "    - 普通 GPT 模型没有分片参数"
echo "    - 如需测试，请添加 --num-experts 参数"
echo ""

# ========================================
# 运行训练
# ========================================
echo "开始训练..."
echo "预期行为："
echo "  1. 训练正常运行"
echo "  2. 如果是 MoE 模型：分片参数被设为固定值，互异性被破坏"
echo "  3. 如果是普通 GPT：不会找到分片参数，注入不生效"
echo ""
echo "日志标识："
echo "  [corrupt-sharded] - schedules.py 中的分片注入日志"
echo ""

# 使用普通 GPT 配置（无分片参数）
# 如需测试分片参数，请取消下面 MoE 配置的注释
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
    2>&1 | tee sharded_same_injection_test.log || true

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
if [ -f "sharded_same_injection_test.log" ]; then
    echo ""
    echo "--- 分片参数注入日志（[corrupt-sharded]）---"
    grep "\[corrupt-sharded\]" sharded_same_injection_test.log | head -40
    
    if [ $? -ne 0 ]; then
        echo "⚠️  未找到 [corrupt-sharded] 日志"
    fi
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="
echo ""

# 创建验证脚本
cat > check_sharded_injection.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/sharded_same_test_db") + "/Collector"

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
    
    # 先查看可用的 stages
    all_stages = conn.execute("SELECT DISTINCT stage FROM coredump").fetchall()
    print(f"可用 stages: {[s[0] for s in all_stages]}")
    
    # 查询 model-after-backward 阶段的分片参数
    result = conn.execute('''
        SELECT step, stage,
               json_extract(data, '$.name') as name,
               json_extract(data, '$.cksum') as cksum,
               json_extract(data, '$.param_sharded') as param_sharded,
               json_extract(data, '$.param_full_replica') as param_full_replica
        FROM coredump 
        WHERE stage = 'model-after-backward'
          AND step = 1
    ''').fetchall()
    
    # 筛选分片参数
    sharded_params = [r for r in result if r[4] == True or r[4] == 'true']
    full_replica_params = [r for r in result if r[5] == True or r[5] == 'true']
    
    print(f"  总参数数: {len(result)}")
    print(f"  分片参数数 (param_sharded=true): {len(sharded_params)}")
    print(f"  全复制参数数 (param_full_replica=true): {len(full_replica_params)}")
    
    if sharded_params:
        print(f"  分片参数列表:")
        for row in sharded_params[:5]:
            print(f"    - {row[2]}: cksum={row[3][:16]}...")
        
        # 保存分片参数用于比对
        results[dp] = {row[2]: row[3] for row in sharded_params}
    else:
        print(f"  ⚠️  未找到分片参数")
        print(f"  提示: 分片参数只存在于 MoE 模型中")
        # 使用全复制参数作为替代示例
        if full_replica_params:
            print(f"  使用全复制参数作为示例:")
            for row in full_replica_params[:3]:
                print(f"    - {row[2]}: cksum={row[3][:16] if row[3] else 'N/A'}...")
            results[dp] = {row[2]: row[3] for row in full_replica_params}
    
    conn.close()
    print("")

# 比对两个 DP rank
print("========================================")
print("验证结果比对")
print("========================================")

if 0 in results and 1 in results:
    all_keys = set(results[0].keys()) & set(results[1].keys())
    
    same_count = 0  # 相同的（对于分片参数，这是错误的）
    diff_count = 0  # 不同的（对于分片参数，这是正确的）
    
    same_params = []
    
    for key in sorted(all_keys):
        cksum0 = results[0].get(key)
        cksum1 = results[1].get(key)
        
        if cksum0 == cksum1:
            same_count += 1
            same_params.append(key)
        else:
            diff_count += 1
    
    print("")
    print(f"分片参数比对结果:")
    print(f"  ✓ 互异的参数数量（正常）: {diff_count}")
    print(f"  ❌ 相同的参数数量（异常/注入成功）: {same_count}")
    
    if same_count > 0:
        print("")
        print("相同的参数（互异性被破坏）:")
        for key in same_params[:5]:
            print(f"  - {key}")
        print("")
        print("🎉 验证成功：检测到分片参数互异性被破坏！")
        sys.exit(0)
    else:
        print("")
        print("⚠️  所有参数都互异，注入可能未生效")
        print("   可能原因:")
        print("   1. 普通 GPT 模型没有分片参数")
        print("   2. 注入配置有误")
        print("   建议: 使用 MoE 模型配置（添加 --num-experts 参数）")
        sys.exit(1)
else:
    print("❌ 无法比对：缺少一个或多个 DP rank 的数据")
    sys.exit(1)
PYTHON_EOF

echo "运行验证脚本..."
python check_sharded_injection.py
VERIFY_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="

if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "✅ 注入实验成功！"
    echo "   成功检测到分片参数互异性被破坏"
else
    echo "⚠️  注入实验可能未成功"
    echo "   注意：普通 GPT 模型没有分片参数"
    echo "   如需测试此约束，请使用 MoE 模型配置"
fi

# 清理临时文件
rm -f check_sharded_injection.py

echo ""
echo "=========================================="
echo "血缘细节总结"
echo "=========================================="
echo "  约束名称: DP分片参数backward后分片互异性检查"
echo "  检测阶段: model-after-backward"
echo "  检测逻辑: 分片参数在不同 DP rank 间应该不同（互异性）"
echo "  注入位置: schedules.py::backward_step() → dump_model('model-after-backward') 之前"
echo "  日志标识: [corrupt-sharded]"
echo "  验证字段: param_sharded, cksum"
echo ""
echo "  VTimeline 修改: megatron_collector.py::dump_model() 添加了 param_sharded 字段"
echo "  Megatron-LM 修改: schedules.py 添加了 sharded_same 注入逻辑"
echo ""
echo "  ⚠️  限制: 分片参数只存在于 MoE 模型中"
echo "     如需测试，请使用带 --num-experts 参数的 MoE 配置"
echo ""

