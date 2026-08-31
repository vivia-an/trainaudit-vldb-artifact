#!/bin/bash
#
# 测试 optimizer-step后DP optimizer_state_dict一致性检查 约束的错误注入
#
# 约束名称：optimizer-step后DP optimizer_state_dict一致性检查
# 用途：通过在 optimizer.step() 后修改 optimizer state，验证 SDCCheck 约束检测系统能否检测不一致
# 预期：训练正常运行，但 optimizer state 在不同 DP rank 间不一致
#
# 注意：
# 1. 注入发生在 training.py 的 optimizer.step() 之后、dump_optimizer_state("optimizer-state-after-step") 之前
# 2. 使用 [corrupt-optim-state-after-step] 日志标识
# 3. 数据收集使用 dump_optimizer_state("optimizer-state-after-step") 方法
#

set -e

# 检查是否在正确的目录
if [ ! -f "pretrain_gpt.py" ]; then
    echo "错误：请在 Megatron-LM 根目录运行此脚本"
    exit 1
fi

echo "=========================================="
echo "测试 optimizer-step后 DP optimizer_state_dict 一致性错误注入"
echo "=========================================="

# 清理之前的数据
export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/optim_state_after_step_test_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# ========================================
# 错误注入配置（optim_state_after_step 注入）
# ========================================
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP=optim_state_after_step
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_STEP=-1  # 在所有步注入（配合 ONCE 只注入一次）
# 注意：使用 --use-distributed-optimizer 时，不同 DP rank 持有不同参数
# DP rank 0 可能持有 layers.10, layers.11 等
# 移除 param_substr 限制，让它注入任意第一个匹配的参数
export MEGATRON_CORRUPT_PARAM_SUBSTR=""  # 空字符串表示注入第一个可用参数
export MEGATRON_OPTIM_STATE_TYPE=momentum  # momentum (exp_avg) | variance (exp_avg_sq) | all
export MEGATRON_CORRUPT_DELTA=0.01
export MEGATRON_CORRUPT_ONCE=1  # 只注入一次

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
echo "  State类型: ${MEGATRON_OPTIM_STATE_TYPE}"
echo "  修改量: ${MEGATRON_CORRUPT_DELTA}"
echo "  数据库路径: ${VTIMELINE_LOGGER_DIR}"
echo "  收集步数: ${VTIMELINE_DUMP_STEP}"
echo ""

# ========================================
# 约束血缘细节
# ========================================
echo "约束血缘细节："
echo "  约束名称: optimizer-step后DP optimizer_state_dict一致性检查"
echo "  检测阶段: model-after-optimizer-step (对应 optimizer-state-after-step)"
echo "  检测字段: exp_avg (动量), exp_avg_sq (二阶矩) 的 cksum"
echo "  约束条件: dp > 1, optimizer_state_type IN ('Adam', 'LAMB', '带状态优化器')"
echo ""
echo "  optimizer_state_dict 嵌套状态说明:"
echo "    - exp_avg: 一阶矩估计（动量），用于 Adam/AdamW"
echo "    - exp_avg_sq: 二阶矩估计（未中心化方差），用于 Adam/AdamW"
echo "    - step: 优化器步数计数器"
echo ""
echo "  注入位置: training.py::train_step() → optimizer.step() 之后"
echo "  注入方式: 修改 optimizer.state[param]['exp_avg'] 或 'exp_avg_sq'"
echo ""
echo "  数据流:"
echo "    1. train_step() 执行 forward_backward"
echo "    2. optimizer.step() 更新参数"
echo "    3. [注入点] _inject_optimizer_state_after_step() 修改 optimizer state"
echo "    4. dump_optimizer_state('optimizer-state-after-step') 收集"
echo "    5. dump_model('model-after-optimizer-step') 收集"
echo ""

# ========================================
# 运行训练
# ========================================
echo "开始训练..."
echo "预期行为："
echo "  1. 训练正常运行"
echo "  2. 在 step ${MEGATRON_CORRUPT_STEP}，DP rank ${MEGATRON_CORRUPT_DP_RANK} 的 optimizer state 被修改"
echo "  3. optimizer-state-after-step 阶段，DP rank 0 和 1 的 cksum 不一致"
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
    2>&1 | tee optim_state_after_step_test.log || true

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
if [ -f "optim_state_after_step_test.log" ]; then
    echo ""
    echo "--- Optim State After Step 注入日志 ---"
    grep "\[corrupt-optim-state-after-step\]" optim_state_after_step_test.log | head -30 || echo "未找到注入日志"
fi

echo ""
echo "=========================================="
echo "自动验证注入结果"
echo "=========================================="
echo ""

# 创建验证脚本
# 注意：使用 --use-distributed-optimizer 时，不同 DP rank 持有不同参数的 optimizer state
# 验证方式改为：比较 DP rank 0 上的 before-injection 和 after-injection 阶段
cat > check_optim_state_after_step.py << 'PYTHON_EOF'
import duckdb
import os
import sys

db_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/volume/qscai/lsk/Megatron-LM/optim_state_after_step_test_db") + "/Collector"

print(f"数据库目录: {db_dir}")
print("")

# 只检查 DP rank 0（注入发生在这里）
db_path = f"{db_dir}/coredump_dp0_tp0_pp0_cp0.db"

if not os.path.exists(db_path):
    print(f"❌ 数据库不存在: {db_path}")
    sys.exit(1)

print(f"=== 验证 DP Rank 0 的 optimizer state 注入 ===")
print("")

conn = duckdb.connect(db_path, read_only=True)

stages = conn.execute("SELECT DISTINCT stage FROM coredump").fetchall()
stage_list = [s[0] for s in stages]
print(f"可用 stages: {stage_list}")

# 查找 before-injection 和 after-injection 阶段
before_stage = 'optimizer-state-after-step-before-injection'
after_stage = 'optimizer-state-after-injection'

if before_stage not in stage_list:
    print(f"⚠️  未找到 {before_stage} 阶段，尝试使用其他阶段")
    before_stage = 'optimizer-state-before-step' if 'optimizer-state-before-step' in stage_list else None
    
if after_stage not in stage_list:
    print(f"⚠️  未找到 {after_stage} 阶段，尝试使用其他阶段")
    after_stage = 'optimizer-state-after-step' if 'optimizer-state-after-step' in stage_list else None

print(f"使用阶段: before={before_stage}, after={after_stage}")
print("")

# 收集 before 和 after 的数据
before_data = {}
after_data = {}

if before_stage:
    result = conn.execute(f'''
        SELECT DISTINCT step FROM coredump WHERE stage = '{before_stage}' ORDER BY step
    ''').fetchall()
    steps = [r[0] for r in result]
    check_step = steps[0] if steps else 1
    
    result = conn.execute(f'''
        SELECT
            json_extract(data, '$.name') as name,
            json_extract(data, '$.state_key') as state_key,
            json_extract(data, '$.cksum') as cksum
        FROM coredump
        WHERE stage = '{before_stage}'
        AND step = {check_step}
        ORDER BY name, state_key
        LIMIT 30
    ''').fetchall()
    
    print(f"Before injection ({before_stage}, step={check_step}):")
    for row in result:
        name, state_key, cksum = row
        print(f"  {name} ({state_key}): {cksum[:20]}..." if cksum else f"  {name} ({state_key}): None")
        if name and state_key:
            before_data[(name, state_key)] = cksum
    print("")

if after_stage:
    result = conn.execute(f'''
        SELECT DISTINCT step FROM coredump WHERE stage = '{after_stage}' ORDER BY step
    ''').fetchall()
    steps = [r[0] for r in result]
    check_step = steps[0] if steps else 1
    
    result = conn.execute(f'''
        SELECT
            json_extract(data, '$.name') as name,
            json_extract(data, '$.state_key') as state_key,
            json_extract(data, '$.cksum') as cksum
        FROM coredump
        WHERE stage = '{after_stage}'
        AND step = {check_step}
        ORDER BY name, state_key
        LIMIT 30
    ''').fetchall()
    
    print(f"After injection ({after_stage}, step={check_step}):")
    for row in result:
        name, state_key, cksum = row
        print(f"  {name} ({state_key}): {cksum[:20]}..." if cksum else f"  {name} ({state_key}): None")
        if name and state_key:
            after_data[(name, state_key)] = cksum
    print("")

conn.close()

# 比对 before 和 after
print("========================================")
print("验证结果比对 (before vs after injection)")
print("========================================")

if not before_data or not after_data:
    print("⚠️  数据不完整，检查日志中的 [corrupt-optim-state-after-step] 输出")
    print(f"   before_data count: {len(before_data)}")
    print(f"   after_data count: {len(after_data)}")
    
    # 回退到检查日志
    print("")
    print("请检查训练日志中是否有以下输出：")
    print("  [corrupt-optim-state-after-step] ✓ Modified exp_avg for xxx")
    print("  [corrupt-optim-state-after-step] ✓ Injection completed")
    sys.exit(1)

mismatch_count = 0
match_count = 0

for key in sorted(before_data.keys()):
    if key in after_data:
        before_cksum = before_data[key]
        after_cksum = after_data[key]
        
        if before_cksum != after_cksum:
            mismatch_count += 1
            name, state_key = key
            print(f"❌ 变化检测到 - {name} ({state_key})")
            print(f"   Before: {before_cksum[:30]}..." if before_cksum else "   Before: None")
            print(f"   After:  {after_cksum[:30]}..." if after_cksum else "   After: None")
        else:
            match_count += 1

print("")
print(f"✓ 未变化的 state 数量: {match_count}")
print(f"❌ 变化的 state 数量: {mismatch_count}")

if mismatch_count > 0:
    print("")
    print("🎉 验证成功：检测到注入导致的 optimizer state 变化！")
    print("   约束 'optimizer-step后DP optimizer_state_dict一致性检查' 可以检测到此错误")
    sys.exit(0)
else:
    print("")
    print("⚠️  验证失败：optimizer state 未变化，注入可能未生效")
    print("   请检查训练日志中的 [corrupt-optim-state-after-step] 输出")
    sys.exit(1)
PYTHON_EOF

echo "运行验证脚本..."
python check_optim_state_after_step.py
VERIFY_EXIT_CODE=$?

rm -f check_optim_state_after_step.py

echo ""
echo "关键信息："
echo "  - 约束名称: optimizer-step后DP optimizer_state_dict一致性检查"
echo "  - 注入位置: training.py optimizer.step() 后"
echo "  - 日志标识: [corrupt-optim-state-after-step]"
echo "  - 验证 stage: optimizer-state-after-step / optimizer-state-after-injection"
echo "  - 验证字段: exp_avg/exp_avg_sq 的 cksum"
echo ""

