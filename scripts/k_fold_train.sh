#!/bin/bash
PROJ_FOLDER=$(cd "$(dirname "$0")";cd ..;pwd)
echo "Locate project at ${PROJ_FOLDER}"

cd ${PROJ_FOLDER}


if [ $# != 4 ]; then
    echo "Usage: GPU=x1,x2... bash $0 /directory/to/all_data <mode: (100, 111)> <model type> <port: (9901, ...)>"
    exit 1;
fi
ROOT_DIR=$1
_MODE=$2
_MODEL=$3
_PORT=$4

if [ -z ${LR} ]; then
    LR=1e-3
fi

echo "Data from ${ROOT_DIR}"
echo "Data mode ${_MODE}"
echo "Model type ${_MODEL}"
echo "Using port ${_PORT}"

if [ -z ${GPU} ]; then
    GPU=0,1,2,3
fi
echo "Using GPUs: ${GPU}"

K=10
for ((i=1;i<=3;i++));
do
    echo "CDR ${i}"
    for ((k=0;k<${K};k++));
    do
        echo "Fold ${k}"
        LR=${LR} DATA_DIR=${ROOT_DIR}/cdrh${i}/fold_${k} GPU=${GPU} MODE=${_MODE} PORT=${_PORT} bash train.sh ${_MODEL} ${i}
    done
done
