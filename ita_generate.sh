#!/bin/bash
if [ -z ${GPU} ]; then
    GPU=0
fi
echo "Using GPU: ${GPU}"

export CUDA_VISIBLE_DEVICES=$GPU
if [ -z ${DATA_DIR} ]; then
    echo "DATA_DIR should be specified"
    exit 1;
fi
CKPT_DIR=$(dirname $1)
CKPT=$1

echo "Using checkpoint: ${CKPT}"

python ita_generate.py \
    --ckpt ${CKPT} \
    --test_set ${DATA_DIR}/skempi_all.json \
    --save_dir ${CKPT_DIR}/ita_results \
    --gpu 0 \
    --n_samples 100