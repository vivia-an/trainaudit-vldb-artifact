#!/bin/bash
#
# 多方向分布式训练正常数据生成脚本
# 生成不同分布式方向的 normal 数据用于 TrainCheck 不变量挖掘
#
# 支持的方向:
# 1. data_parallel (DP) - DP rank 间参数一致性
# 2. tensor_parallel (TP) - TP rank 间 dtype/requires_grad 一致性
# 3. mixed_precision - 混合精度一致性检查
# 4. distributed_optimizer - 分布式优化器状态检查
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export MEGATRON_ROOT="$SCRIPT_DIR"
export PYTHONPATH="${MEGATRON_ROOT}/../VTimeline/src:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 检查 GPU 数量
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
echo "=========================================="
echo "多方向分布式训练正常数据生成"
echo "检测到 GPU 数量: $GPU_COUNT"
echo "=========================================="

# 不注入错误
unset MEGATRON_INJECT_PARAM_CORRUPTION
export VTIMELINE_DUMP_STEP=5

#############################################
# 方向 1: Data Parallel (DP) 正常数据
#############################################
generate_dp_normal() {
    echo ""
    echo "=========================================="
    echo "[1/4] 生成 Data Parallel (DP) 正常数据"
    echo "=========================================="
    
    export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/normal_db/dp_normal"
    rm -rf "$VTIMELINE_LOGGER_DIR"
    mkdir -p "$VTIMELINE_LOGGER_DIR"

    # 与 mixed_precision / dist_optimizer 对齐：1 GPU、GBS=8、train-iters=5、VTIMELINE_DUMP_STEP=5；
    # 去掉真实 2DP + dist-opt（易卡在 step1，事件量远小于其他 normal）。训练后复制 dp0→dp1 模拟第二 rank。
    echo "使用 1 GPU + dp0→dp1 复制（事件量与其他 normal 同级）..."
    python -m torch.distributed.run \
        --nproc_per_node=1 \
        --nnodes=1 \
        pretrain_gpt.py \
        --use-mcore-models \
        --num-layers 12 \
        --hidden-size 128 \
        --num-attention-heads 4 \
        --seq-length 64 \
        --max-position-embeddings 512 \
        --position-embedding-type rope \
        --swiglu \
        --disable-bias-linear \
        --micro-batch-size 2 \
        --global-batch-size 8 \
        --train-iters 5 \
        --lr 1.0e-4 \
        --min-lr 1.0e-5 \
        --bf16 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50257 \
        --data-cache-path ./data_cache \
        --no-gradient-accumulation-fusion \
        2>&1 | tee "${MEGATRON_ROOT}/normal_db/dp_normal_training.log"

    cd "$VTIMELINE_LOGGER_DIR/Collector"
    ORIG_DB=$(ls coredump_dp0_*.db 2>/dev/null | head -1)
    if [ -n "$ORIG_DB" ]; then
        NEW_DB=$(echo "$ORIG_DB" | sed 's/dp0/dp1/')
        cp "$ORIG_DB" "$NEW_DB"
        echo "模拟 DP rank 1: $NEW_DB"
    else
        echo "⚠️ 未找到 coredump_dp0_*.db，跳过 dp1 复制" >&2
    fi
    cd "$MEGATRON_ROOT"
    
    echo "✅ DP 正常数据生成完成"
}

#############################################
# 方向 2: Tensor Parallel (TP) 正常数据
#############################################
generate_tp_normal() {
    echo ""
    echo "=========================================="
    echo "[2/4] 生成 Tensor Parallel (TP) 正常数据"
    echo "=========================================="
    
    export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/normal_db/tp_normal"
    rm -rf $VTIMELINE_LOGGER_DIR
    mkdir -p $VTIMELINE_LOGGER_DIR
    
    if [ "$GPU_COUNT" -ge 2 ]; then
        echo "使用 2 GPU 生成真实 TP 数据..."
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
            --swiglu \
            --disable-bias-linear \
            --micro-batch-size 2 \
            --global-batch-size 8 \
            --train-iters 5 \
            --lr 1.0e-4 \
            --min-lr 1.0e-5 \
            --bf16 \
            --tensor-model-parallel-size 2 \
            --pipeline-model-parallel-size 1 \
            --sequence-parallel \
            --mock-data \
            --tokenizer-type NullTokenizer \
            --vocab-size 50257 \
            --data-cache-path ./data_cache \
            --no-gradient-accumulation-fusion \
            2>&1 | tee normal_db/tp_normal_training.log
    else
        echo "⚠️ TP 需要至少 2 GPU, 使用已有的 tp_normal_db"
        if [ -d "${MEGATRON_ROOT}/tp_normal_db/Collector" ]; then
            mkdir -p $VTIMELINE_LOGGER_DIR
            cp -r "${MEGATRON_ROOT}/tp_normal_db/Collector" $VTIMELINE_LOGGER_DIR/
            echo "已复制现有 TP 数据"
        else
            echo "❌ 无可用 TP 数据, 跳过"
            return 0
        fi
    fi
    
    echo "✅ TP 正常数据生成完成"
}

#############################################
# 方向 3: Mixed Precision (混合精度) 正常数据
#############################################
generate_mixed_precision_normal() {
    echo ""
    echo "=========================================="
    echo "[3/4] 生成 Mixed Precision 正常数据"
    echo "=========================================="
    
    export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/normal_db/mixed_precision_normal"
    rm -rf $VTIMELINE_LOGGER_DIR
    mkdir -p $VTIMELINE_LOGGER_DIR
    
    # 混合精度训练主要验证 dtype 一致性
    python -m torch.distributed.run \
        --nproc_per_node=1 \
        --nnodes=1 \
        pretrain_gpt.py \
        --use-mcore-models \
        --num-layers 12 \
        --hidden-size 128 \
        --num-attention-heads 4 \
        --seq-length 64 \
        --max-position-embeddings 512 \
        --position-embedding-type rope \
        --swiglu \
        --disable-bias-linear \
        --micro-batch-size 2 \
        --global-batch-size 8 \
        --train-iters 5 \
        --lr 1.0e-4 \
        --min-lr 1.0e-5 \
        --bf16 \
        --grad-reduce-in-bf16 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50257 \
        --data-cache-path ./data_cache \
        --no-gradient-accumulation-fusion \
        2>&1 | tee normal_db/mixed_precision_training.log
    
    # 复制为第二个 rank (模拟一致性)
    cd $VTIMELINE_LOGGER_DIR/Collector
    ORIG_DB=$(ls coredump_dp0_*.db 2>/dev/null | head -1)
    if [ -n "$ORIG_DB" ]; then
        NEW_DB=$(echo $ORIG_DB | sed 's/dp0/dp1/')
        cp "$ORIG_DB" "$NEW_DB"
    fi
    cd "$MEGATRON_ROOT"
    
    echo "✅ Mixed Precision 正常数据生成完成"
}

#############################################
# 方向 4: Distributed Optimizer 正常数据
#############################################
generate_dist_optimizer_normal() {
    echo ""
    echo "=========================================="
    echo "[4/4] 生成 Distributed Optimizer 正常数据"
    echo "=========================================="
    
    export VTIMELINE_LOGGER_DIR="${MEGATRON_ROOT}/normal_db/dist_optimizer_normal"
    rm -rf $VTIMELINE_LOGGER_DIR
    mkdir -p $VTIMELINE_LOGGER_DIR
    
    python -m torch.distributed.run \
        --nproc_per_node=1 \
        --nnodes=1 \
        pretrain_gpt.py \
        --use-mcore-models \
        --num-layers 12 \
        --hidden-size 128 \
        --num-attention-heads 4 \
        --seq-length 64 \
        --max-position-embeddings 512 \
        --position-embedding-type rope \
        --swiglu \
        --disable-bias-linear \
        --micro-batch-size 2 \
        --global-batch-size 8 \
        --train-iters 5 \
        --lr 1.0e-4 \
        --min-lr 1.0e-5 \
        --bf16 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --use-distributed-optimizer \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50257 \
        --data-cache-path ./data_cache \
        --no-gradient-accumulation-fusion \
        2>&1 | tee normal_db/dist_optimizer_training.log
    
    # 复制为第二个 rank
    cd $VTIMELINE_LOGGER_DIR/Collector
    ORIG_DB=$(ls coredump_dp0_*.db 2>/dev/null | head -1)
    if [ -n "$ORIG_DB" ]; then
        NEW_DB=$(echo $ORIG_DB | sed 's/dp0/dp1/')
        cp "$ORIG_DB" "$NEW_DB"
    fi
    cd "$MEGATRON_ROOT"
    
    echo "✅ Distributed Optimizer 正常数据生成完成"
}

#############################################
# 主逻辑
#############################################
mkdir -p "${MEGATRON_ROOT}/normal_db"

# 解析参数
DIRECTION="${1:-all}"

case "$DIRECTION" in
    dp)
        generate_dp_normal
        ;;
    tp)
        generate_tp_normal
        ;;
    mixed)
        generate_mixed_precision_normal
        ;;
    optimizer)
        generate_dist_optimizer_normal
        ;;
    all)
        generate_dp_normal
        generate_tp_normal
        generate_mixed_precision_normal
        generate_dist_optimizer_normal
        ;;
    *)
        echo "用法: $0 [dp|tp|mixed|optimizer|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "数据生成完成!"
echo "=========================================="
echo ""
echo "生成的 normal 数据目录:"
ls -la "${MEGATRON_ROOT}/normal_db/"
echo ""
echo "下一步: 运行不变量推断"
echo "  python batch_traincheck_sdc.py --infer-multi-direction"
