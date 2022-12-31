#!/bin/bash
PROJ_FOLDER=$(cd "$(dirname "$0")";cd ..;pwd)
echo "Locate project at ${PROJ_FOLDER}"

# add python path
export PYTHONPATH=$PYTHONPATH:${PROJ_FOLDER}
cd ${PROJ_FOLDER}

if [ $# != 3 ]; then
    echo "Usage: bash $0 /path/to/summary /directory/to/pdb /path/to/SAbDab_summary"
    exit 1;
fi
# path to summary file of RAbD
SUMMARY=$1
# path to structure folder of SAbDab since many of them have been included(our paper use IMGT numbering)
PDB_DIR=$2
# path to SAbDab summary
SABDAB=$3
# extract data directory
DATA_DIR=$(dirname ${SUMMARY})
echo "Summary file at ${SUMMARY}. SAbDab file at ${SABDAB}. PDB folder at ${PDB_DIR}. Data working directory at ${DATA_DIR}"

ALL=${DATA_DIR}/rabd_all.json

# download data
python -m data.download \
    --summary ${SUMMARY} \
    --pdb_dir ${PDB_DIR} \
    --fout ${ALL} \
    --type rabd \
    --numbering imgt \
    --pre_numbered \
    --n_cpu 4


# cdrh1/2/3, merge SAbDab data for training
# currently we set i = 3, namely, for cdrh3 design only
for ((i=3;i<=3;i++));
do
    CDR=cdrh${i}
    PROCESS_DIR=${DATA_DIR}/${CDR}
    
    echo "Processing ${CDR}"
    # split data
    python data/split.py \
        --data ${SABDAB} \
        --rabd ${ALL} \
        --out_dir ${PROCESS_DIR} \
        --valid_ratio 0.1 \
        --cdr ${CDR} \
        --filter 111
    # process
    python data/dataset.py --dataset ${PROCESS_DIR}/test.json
    python data/dataset.py --dataset ${PROCESS_DIR}/valid.json
    python data/dataset.py --dataset ${PROCESS_DIR}/train.json
done
