#!/bin/bash
#
# 测试 DP Optimizer State 一致性约束的错误注入
#
# 用途：通过注入optimizer_state不一致错误，验证SDCCheck约束检测系统能否正确识别
# 预期：训练正常运行，但在optimizer-state-after-injection阶段，不同DP rank的optimizer state不一致
#
# 注意：
# 1. 注入发生在 optimizer.py 的 step_with_ready_grads() 方法中
# 2. 使用 [corrupt-optim] 日志标识（不是 [corrupt]）
# 3. 数据收集使用 dump_optimizer_state() 方法
#

set -e

# 检查是否在正确的目录
if [ ! -f "pretrain_gpt.py" ]; then
    echo "错误：请在 Megatron-LM 根目录运行此脚本"
    exit 1
fi

echo "=========================================="
echo "测试 DP Optimizer State 一致性错误注入"
echo "=========================================="

# 清理之前的数据
#export VTIMELINE_LOGGER_DIR=${VTIMELINE_LOGGER_DIR:-$HOME/lsk/Megatron-LM/optimizer_state_test_db}
export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/optimizer_state_test_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# ========================================
# 错误注入配置（方案3：直接修改 optimizer.state）
# ========================================
export MEGATRON_INJECT_PARAM_CORRUPTION=1
export MEGATRON_CORRUPT_OP=optimizer_state
export MEGATRON_CORRUPT_DP_RANK=0
export MEGATRON_CORRUPT_STEP=1  # 在第1步注入（第0步optimizer state尚未创建）
export MEGATRON_CORRUPT_PARAM_SUBSTR="layers.0.mlp.linear_fc1.weight"
export MEGATRON_OPTIM_STATE_TYPE=momentum  # momentum (exp_avg) | variance (exp_avg_sq) | momentum_buffer | all
export MEGATRON_CORRUPT_DELTA=0.01

# ========================================
# VTimeline 数据收集配置
# ========================================
export VTIMELINE_DUMP_STEP=2  # 收集前2步（step 0 和 step 1，注入发生在 step 1）

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
echo "  扰动幅度: ${MEGATRON_CORRUPT_DELTA}"
echo "  数据库路径: ${VTIMELINE_LOGGER_DIR}"
echo "  收集步数: ${VTIMELINE_DUMP_STEP}"
echo ""

# ========================================
# 运行训练（预期正常运行，但optimizer state不一致）
# ========================================
echo "开始训练..."
echo "预期行为："
echo "  1. 训练正常运行（不会崩溃）"
echo "  2. 在 step 1，DP rank 0 的 optimizer state 被修改"
echo "  3. optimizer-state-after-injection 阶段，DP rank 0 和 1 的 state 不一致"
echo ""
echo "日志标识："
echo "  [corrupt-optim] - optimizer.py 中的注入日志"
echo ""

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
    2>&1 | tee optimizer_state_injection_test.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "训练结束（退出码: ${EXIT_CODE}）"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 预期行为：训练正常完成"
    echo "  optimizer_state注入不会导致训练崩溃，只会导致state不一致"
else
    echo "⚠ 意外：训练失败（退出码: ${EXIT_CODE}）"
    echo "  请检查日志中的错误信息"
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
if [ -f "optimizer_state_injection_test.log" ]; then
    echo ""
    echo "--- Optimizer State 注入日志（[corrupt-optim]）---"
    grep "\[corrupt-optim\]" optimizer_state_injection_test.log | head -30
    
    if [ $? -ne 0 ]; then
        echo "⚠️  未找到 [corrupt-optim] 日志"
        echo "   可能的原因："
        echo "   1. optimizer.py 未正确修改"
        echo "   2. VTimeline 未正确安装"
        echo "   3. 环境变量配置错误"
        echo ""
        echo "检查是否有旧的 [corrupt] 日志（来自 schedules.py）..."
        grep "\[corrupt\]" optimizer_state_injection_test.log | head -10
    fi
    echo ""
    
    # 统计注入次数
    INJECT_COUNT=$(grep -c "✓ Injection completed" optimizer_state_injection_test.log || echo "0")
    echo "✓ 检测到 $INJECT_COUNT 次成功注入"
    
    # 检查修改的 state 数量
    MODIFIED_COUNT=$(grep -c "Modified exp_avg" optimizer_state_injection_test.log || echo "0")
    echo "✓ 修改了 $MODIFIED_COUNT 个 exp_avg state"
fi

echo ""
echo "检查 optimizer state 数据收集..."
if [ -d "$VTIMELINE_LOGGER_DIR/Collector" ]; then
    for db in $VTIMELINE_LOGGER_DIR/Collector/*.db; do
        if [ -f "$db" ]; then
            echo "检查数据库: $(basename $db)"
            # 使用 duckdb 查询 optimizer state 数据
            if command -v duckdb &> /dev/null; then
                STAGE_COUNT=$(duckdb "$db" "SELECT COUNT(DISTINCT stage) FROM coredump WHERE stage LIKE '%optimizer-state%'" 2>/dev/null || echo "0")
                echo "  找到 $STAGE_COUNT 个 optimizer-state 相关的 stage"
            fi
        fi
    done
fi

echo ""
echo "=========================================="
echo "下一步：运行验证脚本"
echo "=========================================="
echo ""
echo "在sdccheck项目中运行以下命令验证optimizer_state不一致："
echo ""
echo "  cd /path/to/sdccheck/sdccheck"
echo "  python verify_dp_optimizer_state_consistency.py"
echo ""
echo "验证脚本会："
echo "  1. 连接到 $VTIMELINE_LOGGER_DIR 中的数据库"
echo "  2. 查询 'optimizer-state-after-injection' stage 的数据"
echo "  3. 比较不同 DP rank 的 optimizer state checksum"
echo "  4. 生成 JSON 报告"
echo ""
echo "预期结果："
echo "  ❌ DP rank 0 和 1 的 optimizer state 不一致"
echo "  ✓ 成功检测到注入的错误"
echo ""
echo "关键信息："
echo "  - 注入位置: optimizer.py::step_with_ready_grads()"
echo "  - 日志标识: [corrupt-optim]"
echo "  - 数据收集: dump_optimizer_state()"
echo "  - 验证 stage: optimizer-state-after-injection"
echo ""

