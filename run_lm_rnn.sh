#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 sh run_lm_rnn.sh 'LSTM' 50 0.5 60

MODEL=$1
EMB_DIM=$2
DROP_OUT=$3
HID=$4

python3 ./train.py \
--model $MODEL \
--emb_dim $EMB_DIM \
--hid $HID \
--dropout_ratio $DROP_OUT \
--n_epochs 2