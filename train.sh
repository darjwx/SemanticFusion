#!/bin/bash
# $1 gpu id
# $2 log name

echo "---- Training model with gpu ${1} ----"
CUDA_VISIBLE_DEVICES=${1} python train_model.py 2>&1 | tee logs/log-${2}.txt
