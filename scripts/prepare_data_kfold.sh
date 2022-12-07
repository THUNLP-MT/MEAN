#!/bin/bash
PROJ_FOLDER=$(cd "$(dirname "$0")";cd ..;pwd)
echo "Locate project at ${PROJ_FOLDER}"

# add python path
export PYTHONPATH=$PYTHONPATH:${PROJ_FOLDER}
cd ${PROJ_FOLDER}

if [ $# != 2 ]; then
    echo "Usage: bash $0 /path/to/summary /directory/to/pdb"
    exit 1;
fi
# path to summary file of SAbDab
SUMMARY=$1
# path to structure folder of SAbDab (our paper use IMGT numbering)
PDB_DIR=$2
# extract data directory
DATA_DIR=$(dirname ${SUMMARY})
echo "Summary file at ${SUMMARY}. PDB folder at ${PDB_DIR}. Data working directory at ${DATA_DIR}"

ALL=${DATA_DIR}/sabdab_all.json

# download data
python -m data.download \
    --summary ${SUMMARY} \
    --pdb_dir ${PDB_DIR} \
    --fout ${ALL} \
    --type sabdab \
    --numbering imgt \
    --pre_numbered \
    --n_cpu 4

K=10  # 10-fold
# cdrh1/2/3
for ((i=1;i<=3;i++));
do
    CDR=cdrh${i}
    PROCESS_DIR=${DATA_DIR}/${CDR}
    
    echo "Processing ${CDR}"
    # split data
    python data/split.py \
        --data ${ALL} \
		--out_dir ${PROCESS_DIR} \
        --k_fold ${K} \
        --cdr ${CDR} \
        --filter 111
    # process
    for ((k=0;k<${K};k++));
    do
        python data/dataset.py --dataset ${PROCESS_DIR}/fold_${k}/test.json
        python data/dataset.py --dataset ${PROCESS_DIR}/fold_${k}/valid.json
        python data/dataset.py --dataset ${PROCESS_DIR}/fold_${k}/train.json
    done
done
