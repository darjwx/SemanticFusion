#!/bin/bash
# $1 gpu id
# $2 log name
# $3 dataset configs

echo "---- Training model with gpu ${1} ----"
CUDA_VISIBLE_DEVICES=${1} python train_model.py --config_path ${3} 2>&1 | tee logs/log-${2}.txt
