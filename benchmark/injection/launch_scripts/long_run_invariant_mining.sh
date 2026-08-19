#!/bin/bash
#
# 长时间不变量挖掘实验
# 目标：生成 2-3 天的训练日志，进行更充分的不变量挖掘
#
# 使用方法：
#   nohup ./long_run_invariant_mining.sh > mining_experiment.log 2>&1 &
#

set -e

# ========================================
# 配置参数
# ========================================
EXPERIMENT_NAME="long_mining_$(date +%Y%m%d_%H%M%S)"
BASE_DIR="/volume/qscai/lsk/Megatron-LM"
OUTPUT_DIR="${BASE_DIR}/long_mining_experiment/${EXPERIMENT_NAME}"

# 训练配置 (半天训练，可随时停止)
TRAIN_STEPS=999999         # 设置很大，手动停止
SAVE_INTERVAL=5000         # checkpoint 保存间隔
LOG_INTERVAL=100           # 日志打印间隔
EVAL_INTERVAL=2000         # 评估间隔

# VTimeline 收集配置
VTIMELINE_DUMP_STEP=999999 # 收集所有步数的数据

# GPU 配置
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")

echo "========================================"
echo "长时间不变量挖掘实验"
echo "========================================"
echo "实验名称: ${EXPERIMENT_NAME}"
echo "输出目录: ${OUTPUT_DIR}"
echo "训练步数: ${TRAIN_STEPS}"
echo "可用GPU:  ${GPU_COUNT}"
echo "开始时间: $(date)"
echo "========================================"

# 创建目录结构
mkdir -p "${OUTPUT_DIR}"/{dp_normal,tp_normal,logs,invariants}

cd "${BASE_DIR}"

# ========================================
# 清理环境变量
# ========================================
unset MEGATRON_INJECT_PARAM_CORRUPTION
unset MEGATRON_CORRUPT_OP
unset MEGATRON_CORRUPT_DP_RANK
unset MEGATRON_CORRUPT_STEP
unset MEGATRON_CORRUPT_PARAM_SUBSTR
unset MEGATRON_RESHAPE_STRATEGY

# ========================================
# 设置 PYTHONPATH (重要!)
# ========================================
export PYTHONPATH=/volume/qscai/lsk/VTimeline/src:/volume/qscai/lsk/traincheck/TrainCheck:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

# ========================================
# 阶段 1: 生成 DP 正常数据
# ========================================
echo ""
echo "========================================"
echo "阶段 1/3: 生成 DP 正常训练数据"
echo "========================================"

export VTIMELINE_LOGGER_DIR="${OUTPUT_DIR}/dp_normal"
export VTIMELINE_DUMP_STEP=${VTIMELINE_DUMP_STEP}
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ "$GPU_COUNT" -ge 2 ]; then
    echo "使用 ${GPU_COUNT} GPU 进行 DP 训练..."
    
    python -m torch.distributed.run \
        --nproc_per_node=2 \
        --nnodes=1 \
        pretrain_gpt.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --num-layers 12 \
        --hidden-size 128 \
        --num-attention-heads 4 \
        --seq-length 64 \
        --max-position-embeddings 64 \
        --micro-batch-size 2 \
        --global-batch-size 4 \
        --train-iters ${TRAIN_STEPS} \
        --lr 0.0001 \
        --min-lr 0.00001 \
        --lr-decay-style cosine \
        --lr-warmup-fraction 0.01 \
        --weight-decay 0.01 \
        --clip-grad 1.0 \
        --bf16 \
        --log-interval ${LOG_INTERVAL} \
        --save-interval ${SAVE_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters 10 \
        --save "${OUTPUT_DIR}/checkpoints/dp" \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50257 \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion \
        --no-async-tensor-model-parallel-allreduce \
        --use-flash-attn \
        --transformer-impl local \
        --attention-softmax-in-fp32 \
        --accumulate-allreduce-grads-in-fp32 \
        2>&1 | tee "${OUTPUT_DIR}/logs/dp_training.log"
else
    echo "⚠️ 只有 1 GPU，DP 训练需要至少 2 GPU"
    echo "使用单 GPU 训练..."
    
    python pretrain_gpt.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --num-layers 12 \
        --hidden-size 128 \
        --num-attention-heads 4 \
        --seq-length 64 \
        --max-position-embeddings 64 \
        --micro-batch-size 2 \
        --global-batch-size 2 \
        --train-iters ${TRAIN_STEPS} \
        --lr 0.0001 \
        --min-lr 0.00001 \
        --lr-decay-style cosine \
        --lr-warmup-fraction 0.01 \
        --weight-decay 0.01 \
        --clip-grad 1.0 \
        --bf16 \
        --log-interval ${LOG_INTERVAL} \
        --save-interval ${SAVE_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters 10 \
        --save "${OUTPUT_DIR}/checkpoints/dp" \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50257 \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion \
        --no-async-tensor-model-parallel-allreduce \
        --use-flash-attn \
        --transformer-impl local \
        --attention-softmax-in-fp32 \
        --accumulate-allreduce-grads-in-fp32 \
        2>&1 | tee "${OUTPUT_DIR}/logs/dp_training.log"
fi

echo "DP 训练完成: $(date)"

# ========================================
# 阶段 2: 生成 TP 正常数据 (如果有足够 GPU)
# ========================================
if [ "$GPU_COUNT" -ge 2 ]; then
    echo ""
    echo "========================================"
    echo "阶段 2/3: 生成 TP 正常训练数据"
    echo "========================================"
    
    export VTIMELINE_LOGGER_DIR="${OUTPUT_DIR}/tp_normal"
    
    python -m torch.distributed.run \
        --nproc_per_node=2 \
        --nnodes=1 \
        pretrain_gpt.py \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 1 \
        --sequence-parallel \
        --num-layers 12 \
        --hidden-size 128 \
        --num-attention-heads 4 \
        --seq-length 64 \
        --max-position-embeddings 64 \
        --micro-batch-size 2 \
        --global-batch-size 2 \
        --train-iters ${TRAIN_STEPS} \
        --lr 0.0001 \
        --min-lr 0.00001 \
        --lr-decay-style cosine \
        --lr-warmup-fraction 0.01 \
        --weight-decay 0.01 \
        --clip-grad 1.0 \
        --bf16 \
        --log-interval ${LOG_INTERVAL} \
        --save-interval ${SAVE_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters 10 \
        --save "${OUTPUT_DIR}/checkpoints/tp" \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50257 \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion \
        --no-async-tensor-model-parallel-allreduce \
        --use-flash-attn \
        --transformer-impl local \
        --attention-softmax-in-fp32 \
        --accumulate-allreduce-grads-in-fp32 \
        2>&1 | tee "${OUTPUT_DIR}/logs/tp_training.log"
    
    echo "TP 训练完成: $(date)"
else
    echo "跳过 TP 训练 (需要至少 2 GPU)"
fi

# ========================================
# 阶段 3: 运行 TrainCheck 不变量挖掘
# ========================================
echo ""
echo "========================================"
echo "阶段 3/3: TrainCheck 不变量挖掘"
echo "========================================"

export PYTHONPATH=/volume/qscai/lsk/traincheck/TrainCheck:/volume/qscai/lsk/VTimeline/src:$PYTHONPATH

# 挖掘 DP 不变量
echo "挖掘 DP 不变量..."
if [ -d "${OUTPUT_DIR}/dp_normal/Collector" ]; then
    python batch_traincheck_sdc.py \
        --infer "${OUTPUT_DIR}/dp_normal" \
        --output-dir "${OUTPUT_DIR}/invariants" \
        2>&1 | tee "${OUTPUT_DIR}/logs/dp_infer.log"
    
    # 复制结果
    if [ -f "${OUTPUT_DIR}/invariants/inferred_invariants.json" ]; then
        mv "${OUTPUT_DIR}/invariants/inferred_invariants.json" \
           "${OUTPUT_DIR}/invariants/dp_inferred_invariants.json"
    fi
fi

# 挖掘 TP 不变量
echo "挖掘 TP 不变量..."
if [ -d "${OUTPUT_DIR}/tp_normal/Collector" ]; then
    python batch_traincheck_sdc.py \
        --infer "${OUTPUT_DIR}/tp_normal" \
        --output-dir "${OUTPUT_DIR}/invariants" \
        2>&1 | tee "${OUTPUT_DIR}/logs/tp_infer.log"
    
    # 复制结果
    if [ -f "${OUTPUT_DIR}/invariants/inferred_invariants.json" ]; then
        mv "${OUTPUT_DIR}/invariants/inferred_invariants.json" \
           "${OUTPUT_DIR}/invariants/tp_inferred_invariants.json"
    fi
fi

# ========================================
# 汇总报告
# ========================================
echo ""
echo "========================================"
echo "实验完成汇总"
echo "========================================"
echo "结束时间: $(date)"
echo ""
echo "数据目录:"
echo "  DP 数据: ${OUTPUT_DIR}/dp_normal"
echo "  TP 数据: ${OUTPUT_DIR}/tp_normal"
echo ""
echo "挖掘的不变量:"
ls -la "${OUTPUT_DIR}/invariants/"*.json 2>/dev/null || echo "  (无不变量文件)"
echo ""
echo "日志文件:"
ls -la "${OUTPUT_DIR}/logs/"
echo ""

# 统计数据量
echo "数据统计:"
for dir in dp_normal tp_normal; do
    if [ -d "${OUTPUT_DIR}/${dir}/Collector" ]; then
        db_count=$(ls "${OUTPUT_DIR}/${dir}/Collector"/*.db 2>/dev/null | wc -l)
        total_size=$(du -sh "${OUTPUT_DIR}/${dir}/Collector" 2>/dev/null | cut -f1)
        echo "  ${dir}: ${db_count} 个 DB 文件, 总大小 ${total_size}"
    fi
done

echo ""
echo "========================================"
echo "实验完成！"
echo "========================================"
