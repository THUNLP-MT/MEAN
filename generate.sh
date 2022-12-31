#!/bin/bash
if [ -z ${GPU} ]; then
    GPU=0
fi
if [ -z ${MODE} ]; then
    MODE=111
fi
if [ -z ${DATA_DIR} ]; then
    DATA_DIR=/data/private/kxz/antibody/data/SAbDab
fi
if [ -z ${MODEL} ]; then
    MODEL=mcatt
fi
if [ -z ${CDR} ]; then
    CDR=3
fi
if [ -z ${RUN} ]; then
    RUN=5
fi

echo "Using GPU: ${GPU}"

export CUDA_VISIBLE_DEVICES=$GPU

VERSION=0
if [ $1 ]; then
    VERSION=$1
fi
CKPT_DIR=${DATA_DIR}/ckpt/${MODEL}_CDR${CDR}_${MODE}/version_${VERSION}
if [ $2 ]; then
    CKPT=${CKPT_DIR}/checkpoint/`ls ${CKPT_DIR}/checkpoint -l | grep -o "epoch$2_step[0-9]*.ckpt"`
else
    CKPT=${CKPT_DIR}/checkpoint/`ls ${CKPT_DIR}/checkpoint -l -t | head -n 2 | grep -o 'epoch[0-9]*_step[0-9]*.ckpt'`
fi

echo "Using checkpoint: ${CKPT}"

python generate.py \
    --ckpt ${CKPT} \
    --test_set ${DATA_DIR}/test.json \
    --out ${CKPT_DIR}/results \
    --gpu 0 \
    --run ${RUN} \
    --mode ${MODE} | tee -a ${CKPT_DIR}/eval_log.txt
