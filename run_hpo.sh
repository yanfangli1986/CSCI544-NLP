#!/usr/bin/env bash

declare -i idx=0

for emb_dim in 10 20 30 40 50
do
    for hid in 60 70 80
    do
        for dropout_ratio in 0.4 0.5 0.6
        do
            let idx+=1
            # echo "($idx) emb_dim = $emb_dim hid_dim = $hid_dim dropout_ratio = $dropout_ratio"
            CUDA_VISIBLE_DEVICES=1 sh run_lm_rnn.sh 'LSTM' $emb_dim $dropout_ratio $hid
        done
    done
done

