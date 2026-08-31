#!/bin/bash
#
# 拓扑 overhead 变体：TP/PP/DP 由环境变量指定（复用 run_overhead_scaled.sh 的 1.2B 配置）
# 用法: TP=2 PP=1 DP=1 DUMP=8 LOGDIR=... TEELOG=... bash run_overhead_topo.sh
#

set -e

cd /volume/posttrain/users/lsk/sdc/lsk/Megatron-LM

export PYTHONPATH=/volume/posttrain/users/lsk/sdc/lsk/VTimeline/src:$PYTHONPATH

TP=${TP:-1}
PP=${PP:-1}
DP=${DP:-2}
NPROC=$((TP * PP * DP))

export VTIMELINE_LOGGER_DIR=${LOGDIR:-/tmp/oh_topo}
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

unset MEGATRON_INJECT_PARAM_CORRUPTION
unset MEGATRON_CORRUPT_OP

export VTIMELINE_DUMP_STEP=${DUMP:-8}
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "topo: TP=$TP PP=$PP DP=$DP nproc=$NPROC dump=$VTIMELINE_DUMP_STEP fast=${VTIMELINE_FAST:-0}"

SP_FLAG="--sequence-parallel"
if [ "$TP" = "1" ]; then SP_FLAG=""; fi

python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
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
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --context-parallel-size 1 \
    $SP_FLAG \
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
    2>&1 | tee ${TEELOG:-topo_training.log}

echo "done TP=$TP PP=$PP DP=$DP"
