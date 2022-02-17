#!/bin/bash

echo "---- Training model with gpu ${1} ----"
CUDA_VISIBLE_DEVICES=${1} python train_model.py
