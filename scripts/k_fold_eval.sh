#!/bin/bash
PROJ_FOLDER=$(cd "$(dirname "$0")";cd ..;pwd)
echo "Locate project at ${PROJ_FOLDER}"

cd ${PROJ_FOLDER}


if [ $# != 4 ]; then
    echo "Usage: GPU=x bash $0 /directory/to/all_data <mode: (100, 111)> <model type> <version>"
    exit 1;
fi
ROOT_DIR=$1
_MODE=$2
_MODEL=$3
VERSION=$4

echo "Data from ${ROOT_DIR}"
echo "Model type ${_MODEL}"

if [ -z ${GPU} ]; then
    GPU=0
fi
echo "Using GPUs: ${GPU}"

K=10
for ((i=1;i<=3;i++));
do
    echo "CDR ${i}"
    for ((k=0;k<${K};k++));
    do
        echo "Fold ${k}"
        DATA_DIR=${ROOT_DIR}/cdrh${i}/fold_${k} \
        GPU=${GPU} MODE=${_MODE} CDR=${i} \
        MODEL=${_MODEL} \
        RUN=1 bash generate.sh ${VERSION}
    done
done

# aggregate results from kfold
for ((i=1;i<=3;i++));
do
    echo "Results of CDR ${i}"
    DATA_DIR=${ROOT_DIR}/cdrh${i}
    LOG=${DATA_DIR}/${_MODEL}_${_MODE}_version${VERSION}_results.txt
    python evaluation/get_k_fold_res.py \
        --data_dir ${DATA_DIR} \
        --cdr_type ${i} \
        --model ${_MODEL} \
        --version ${VERSION} \
        --mode ${_MODE} | tee -a ${LOG}
done
