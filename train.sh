#!/bin/bash

########## adjust configs according to your needs ##########
TRAIN_SET=${DATA_DIR}/train.json
DEV_SET=${DATA_DIR}/valid.json
SAVE_DIR=${DATA_DIR}/ckpt
BATCH_SIZE=16  # need four 12G GPU
MAX_EPOCH=20
if [ -z ${LR} ]; then
	LR=1e-3
fi
######### end of adjust ##########

########## Instruction ##########
# This script takes three optional environment variables:
# GPU / ADDR / PORT
# e.g. Use gpu 0, 1 and 4 for training, set distributed training
# master address and port to localhost:9901, the command is as follows:
#
# GPU="0,1,4" ADDR=localhost PORT=9901 bash train.sh
#
# Default value: GPU=-1 (use cpu only), ADDR=localhost, PORT=9901
# Note that if your want to run multiple distributed training tasks,
# either the addresses or ports should be different between
# each pair of tasks.
######### end of instruction ##########

# set master address and port e.g. ADDR=localhost PORT=9901 bash train.sh
MASTER_ADDR=localhost
MASTER_PORT=9901
if [ $ADDR ]; then MASTER_ADDR=$ADDR; fi
if [ $PORT ]; then MASTER_PORT=$PORT; fi
echo "Master address: ${MASTER_ADDR}, Master port: ${MASTER_PORT}"

# set gpu, e.g. GPU="0,1,2,3" bash train.sh
if [ -z "$GPU" ]; then
    GPU="-1"  # use CPU
fi
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPUs: $GPU"
GPU_ARR=(`echo $GPU | tr ',' ' '`)

if [ ${#GPU_ARR[@]} -gt 1 ]; then
	PREFIX="python -m torch.distributed.launch --nproc_per_node=${#GPU_ARR[@]} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"
else
    PREFIX="python"
fi

if [ -z ${MODE} ]; then
    MODE=111
fi

MODEL=mean
if [ $1 ]; then
	MODEL=$1
fi

CDR=3
if [ $2 ]; then
    CDR=$2
fi

SAVE_DIR=${SAVE_DIR}/${MODEL}_CDR${CDR}_${MODE}

${PREFIX} train.py \
    --train_set $TRAIN_SET \
    --valid_set $DEV_SET \
    --save_dir $SAVE_DIR \
    --batch_size ${BATCH_SIZE} \
    --max_epoch ${MAX_EPOCH} \
    --gpu "${!GPU_ARR[@]}" \
    --mode ${MODE} \
	--model ${MODEL} \
    --cdr_type ${CDR} \
    --lr ${LR} \
    --alpha 0.8 \
    --anneal_base 0.95 \
	--n_iter 3
