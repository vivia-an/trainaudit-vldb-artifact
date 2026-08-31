#!/bin/bash
#
# 单 GPU 正常训练 + 模拟 DP 数据
#

set -e
cd /volume/qscai/lsk/Megatron-LM

export PYTHONPATH=/volume/qscai/lsk/VTimeline/src:$PYTHONPATH

echo "=========================================="
echo "单 GPU 正常训练 - 生成模拟 DP 数据"
echo "=========================================="

export VTIMELINE_LOGGER_DIR=/volume/qscai/lsk/Megatron-LM/dp_normal_db
rm -rf $VTIMELINE_LOGGER_DIR
mkdir -p $VTIMELINE_LOGGER_DIR

unset MEGATRON_INJECT_PARAM_CORRUPTION
export VTIMELINE_DUMP_STEP=5
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "配置: 1 GPU, 5 steps"

# 单 GPU 训练
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
    2>&1 | tee dp_normal_training.log

echo ""
echo "=========================================="
echo "复制数据模拟 DP rank 1"
echo "=========================================="

cd $VTIMELINE_LOGGER_DIR/Collector

# 原始文件
ORIG_DB=$(ls coredump_dp0_*.db 2>/dev/null | head -1)
if [ -z "$ORIG_DB" ]; then
    echo "未找到 dp0 数据库文件"
    ls -la
    exit 1
fi

echo "原始文件: $ORIG_DB"

# 复制为 dp1 (正常情况下 DP rank 数据一致)
NEW_DB=$(echo $ORIG_DB | sed 's/dp0/dp1/')
cp "$ORIG_DB" "$NEW_DB"
echo "复制为: $NEW_DB"

echo ""
ls -la *.db
echo ""
echo "✅ 模拟 DP 数据生成完成!"
