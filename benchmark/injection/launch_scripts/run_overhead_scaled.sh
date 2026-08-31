#!/bin/bash
#
# 正常 DP 训练脚本（不注入错误）
# 用途：生成正常的 DP 并行数据，供 TrainCheck 不变量推断
#

set -e

cd /volume/posttrain/users/lsk/sdc/lsk/Megatron-LM

# 设置 vtimeline 路径
export PYTHONPATH=/volume/posttrain/users/lsk/sdc/lsk/VTimeline/src:$PYTHONPATH

echo "=========================================="
echo "正常 DP 训练 - 生成 TrainCheck 不变量数据"
echo "=========================================="

# 数据收集目录
export VTIMELINE_LOGGER_DIR=${LOGDIR:-/volume/posttrain/users/lsk/sdc/lsk/Megatron-LM/overhead_run_db}
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

# 关键：不启用错误注入
unset MEGATRON_INJECT_PARAM_CORRUPTION
unset MEGATRON_CORRUPT_OP

# 收集步数
export VTIMELINE_DUMP_STEP=${DUMP:-8}

export CUDA_DEVICE_MAX_CONNECTIONS=1

echo ""
echo "配置信息："
echo "  错误注入: 禁用"
echo "  并行方式: DP=2, TP=1 (纯 DP 并行)"
echo "  数据库路径: ${VTIMELINE_LOGGER_DIR}"
echo "  收集步数: ${VTIMELINE_DUMP_STEP}"
echo ""

# DP 并行训练: 2 GPU, tensor-model-parallel-size=1
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
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
    --train-iters 8 \
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
    2>&1 | tee ${TEELOG:-dp_normal_training.log}

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo "检查数据库文件..."
ls -la $VTIMELINE_LOGGER_DIR/Collector/*.db 2>/dev/null || echo "未找到 db 文件"
