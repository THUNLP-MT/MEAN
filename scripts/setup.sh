#!/bin/bash
PROJ_FOLDER=$(cd "$(dirname "$0")";cd ..;pwd)
echo "Locate project at ${PROJ_FOLDER}"

conda env create -f ${PROJ_FOLDER}/env.yml

EVAL_FOLDER=${PROJ_FOLDER}/evaluation
g++ -static -O3 -ffast-math -lm -o ${EVAL_FOLDER}/TMscore ${EVAL_FOLDER}/TMscore.cpp
