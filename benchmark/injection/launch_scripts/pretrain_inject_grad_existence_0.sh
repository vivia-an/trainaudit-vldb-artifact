#!/bin/bash
#
# 测试 backward后DP参数grad存在性一致性检查 约束的错误注入
#
# 约束名称：backward后DP参数grad存在性一致性检查
# 用途：通过在 backward 后将特定参数的 grad 设为 None，验证 SDCCheck 约束检测系统能否检测不一致
# 预期：训练可能出现异常，grad 存在性在不同 DP rank 间不一致
#
# 注意：
# 1. 注入发生在 schedules.py 的 backward_step() 后、dump_model("model-after-backward") 之前
# 2. 使用 [corrupt-grad-existence] 日志标识
# 3. 数据收集使用 dump_model("model-after-backward") 方法
# 4. 检测 grad_cksum 是否为 None
#

set -e

# 检查是否在正确的目录
if [ ! -f "pretrain_gpt.py" ]; then
    echo "错误：请在 Megatron-LM 根目录运行此脚本"
    exit 1
fi

echo "=========================================="
echo "测试 backward后 DP参数 grad 存在性一致性错误注入"
echo "=========================================="

# 清理之前的数据
export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/grad_existence_test_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# ========================================
# 错误注入配置（grad_existence 注入）
# ========================================
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP=grad_existence
export MEGATRON_CORRUPT_DP_RANK=0
# 注意：由于 step 在 iteration 开始时就更新，backward 执行时 step 可能已经是当前值
# 使用 -1 表示在所有步都注入，或者用具体步数匹配实际时序
export MEGATRON_CORRUPT_STEP=-1  # 在所有步注入（首次满足条件时注入一次）
# 注意：使用 --use-distributed-optimizer 时，不同 DP rank 持有不同参数
# 移除 param_substr 限制，让它注入任意第一个可用参数
export MEGATRON_CORRUPT_PARAM_SUBSTR=""  # 空字符串表示注入第一个可用参数
# 添加注入标记，确保只注入一次
export MEGATRON_CORRUPT_ONCE=1

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
echo "  约束名称: backward后DP参数grad存在性一致性检查"
echo "  检测阶段: model-after-backward"
echo "  检测字段: grad_cksum (是否为 None)"
echo "  约束条件: dp > 1, requires_grad = true"
echo ""
echo "  注入位置: schedules.py::forward_backward_no_pipelining() → backward 后"
echo "  注入方式: 将参数的 grad 设为 None"
echo ""
echo "  数据流:"
echo "    1. forward_backward_no_pipelining() 执行 backward"
echo "    2. [注入点] _inject_grad_existence_corruption() 将 grad 设为 None"
echo "    3. dump_model('model-after-backward') 收集"
echo "    4. 不同 DP rank 的 grad_cksum 存在性不一致"
echo ""

# ========================================
# 运行训练
# ========================================
echo "开始训练..."
echo "预期行为："
echo "  1. 训练可能出现警告或异常"
echo "  2. 在 step ${MEGATRON_CORRUPT_STEP}，DP rank ${MEGATRON_CORRUPT_DP_RANK} 的参数 grad 被设为 None"
echo "  3. model-after-backward 阶段，DP rank 0 的 grad_cksum 为 None，DP rank 1 有值"
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
    2>&1 | tee grad_existence_test.log || true

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
if [ -f "grad_existence_test.log" ]; then
    echo ""
    echo "--- Grad Existence 注入日志 ---"
    grep "\[corrupt-grad-existence\]" grad_existence_test.log | head -30 || echo "未找到注入日志"
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="
echo ""

# 创建验证脚本
# 注意：使用 distributed optimizer 时，grad 存储在 main_grad 中，
# 需要检查 main-grad-after-backward 阶段的 cksum 字段
cat > check_grad_existence.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/grad_existence_test_db") + "/Collector"

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

# 收集各 DP rank 的 main_grad 存在性数据
# 使用 main-grad-after-backward 阶段（distributed optimizer 场景）
grad_data = {}

# 优先检查 main-grad-after-backward，其次 model-after-backward
target_stages = ['main-grad-after-backward', 'model-after-backward']

for dp in [0, 1]:
    print(f"=== DP Rank {dp} ===")
    db_path = f"{db_dir}/coredump_dp{dp}_tp0_pp0_cp0.db"
    conn = duckdb.connect(db_path, read_only=True)
    
    # 检查有哪些 stage
    stages = conn.execute("SELECT DISTINCT stage FROM coredump").fetchall()
    stage_list = [s[0] for s in stages]
    print(f"  可用 stages: {stage_list}")
    
    # 找到可用的目标 stage
    target_stage = None
    for ts in target_stages:
        if ts in stage_list:
            target_stage = ts
            break
    
    if not target_stage:
        print(f"  ⚠️  未找到目标 stage")
        conn.close()
        continue
    
    print(f"  使用 stage: {target_stage}")
    
    # 查询所有 step 的数据
    result = conn.execute(f'''
        SELECT DISTINCT step FROM coredump WHERE stage = '{target_stage}' ORDER BY step
    ''').fetchall()
    steps = [r[0] for r in result]
    print(f"  可用 steps: {steps}")
    
    # 使用第一个有效 step
    check_step = steps[0] if steps else 1
    
    # 查询目标 stage 的参数 grad 存在性
    # main-grad 阶段使用 cksum 字段，model 阶段使用 grad_cksum 字段
    cksum_field = 'cksum' if 'main-grad' in target_stage else 'grad_cksum'
    
    result = conn.execute(f'''
        SELECT
            step,
            stage,
            json_extract(data, '$.name') as name,
            json_extract(data, '$.{cksum_field}') as grad_cksum
        FROM coredump
        WHERE stage = '{target_stage}'
        AND step = {check_step}
        ORDER BY name
        LIMIT 30
    ''').fetchall()
    
    print(f"  {target_stage} (step={check_step}) 的参数 grad 存在性:")
    for row in result:
        step, stg, name, grad_cksum = row
        has_grad = grad_cksum is not None and str(grad_cksum) != 'null'
        print(f"    name={name}, {cksum_field}={grad_cksum}, has_grad={has_grad}")
        
        if name:
            key = (name, step)
            if key not in grad_data:
                grad_data[key] = {}
            grad_data[key][dp] = has_grad
    
    conn.close()
    print("")

# 比对两个 DP rank 的 grad 存在性
print("========================================")
print("验证结果比对")
print("========================================")

if not grad_data:
    print("❌ 无法比对：未找到有效的 grad 数据")
    sys.exit(1)

mismatch_count = 0
match_count = 0

for key in sorted(grad_data.keys(), key=lambda x: str(x)):
    name, step = key
    dp_grads = grad_data[key]
    
    if len(dp_grads) == 2:
        has_grad_0 = dp_grads.get(0)
        has_grad_1 = dp_grads.get(1)
        
        if has_grad_0 != has_grad_1:
            mismatch_count += 1
            print(f"❌ 不一致 - {name}")
            print(f"   DP0 has_grad: {has_grad_0}")
            print(f"   DP1 has_grad: {has_grad_1}")
        else:
            match_count += 1

print("")
print(f"✓ 一致的参数数量: {match_count}")
print(f"❌ 不一致的参数数量: {mismatch_count}")

if mismatch_count > 0:
    print("")
    print("🎉 验证成功：检测到注入导致的 grad 存在性不一致！")
    print("   约束 'backward后DP参数grad存在性一致性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("")
    print("⚠️  验证失败：所有参数 grad 存在性都一致，注入可能未生效")
    print("   (注意：如果所有 grad 都是 None，可能是 dump 时机问题)")
    sys.exit(1)
PYTHON_EOF

echo "运行验证脚本..."
python check_grad_existence.py
VERIFY_EXIT_CODE=$?

# 清理临时文件
rm -f check_grad_existence.py

echo ""
echo "关键信息："
echo "  - 约束名称: backward后DP参数grad存在性一致性检查"
echo "  - 注入位置: schedules.py backward 后"
echo "  - 日志标识: [corrupt-grad-existence]"
echo "  - 验证 stage: model-after-backward"
echo "  - 验证字段: grad_cksum (None vs 有值)"
echo ""

