#!/bin/bash
if [ -z ${GPU} ]; then
    GPU=0
fi
if [ -z ${MODE} ]; then
    MODE=111
fi
if [ -z ${MODEL} ]; then
    MODEL=mcatt
fi
if [ -z ${CDR} ]; then
    CDR=3
fi

echo "Using GPU: ${GPU}"

export CUDA_VISIBLE_DEVICES=$GPU

VERSION=0
if [ $1 ]; then
    VERSION=$1
fi
if [ $2 ]; then
    CKPT_DIR=$(dirname $2)
    CKPT=$2
else
    CKPT_DIR=${DATA_DIR}/ckpt/${MODEL}_CDR${CDR}_${MODE}/version_${VERSION}
    CKPT=${CKPT_DIR}/checkpoint/`ls ${CKPT_DIR}/checkpoint -l -t | head -n 2 | grep -o 'epoch[0-9]*_step[0-9]*.ckpt'`
fi

echo "Using checkpoint: ${CKPT}"

python generate.py \
    --ckpt ${CKPT} \
    --test_set ${DATA_DIR}/test.json \
    --out ${CKPT_DIR}/results \
    --gpu 0 \
    --rabd_test \
    --rabd_sample 1 \
    --topk 100 \
    --mode ${MODE} | tee -a ${CKPT_DIR}/rabd_log.txt
