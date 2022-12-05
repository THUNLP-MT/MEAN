#!/bin/bash
if [ -z ${GPU} ]; then
    GPU=0
fi
echo "Using GPU: ${GPU}"

if [ -z ${LR} ]; then
    LR=1e-3
fi

export CUDA_VISIBLE_DEVICES=$GPU
DATA_DIR=${CKPT_DIR}/../../..

# checkpoint at certain Epoch
if [ $1 ]; then
    CKPT=${CKPT_DIR}/checkpoint/`ls ${CKPT_DIR}/checkpoint -l | grep -o "epoch$2_step[0-9]*.ckpt"`
else
    CKPT=${CKPT_DIR}/checkpoint/`ls ${CKPT_DIR}/checkpoint -l -t | head -n 2 | grep -o 'epoch[0-9]*_step[0-9]*.ckpt'`
fi

echo "Using checkpoint: ${CKPT}"

python ita_train.py \
    --pretrain_ckpt ${CKPT} \
    --test_set ${DATA_DIR}/all.json \
    --save_dir ${CKPT_DIR}/ita \
    --batch_size 4 \
    --update_freq 4 \
    --gpu 0 \
    --lr ${LR}
