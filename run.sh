#!/bin/bash

export DATAROOT=./data/

datasets=($(ls $DATAROOT))

for dataset in ${datasets[@]}
do
    if [ -d ./results/$dataset ]; then
        continue
    fi
    cmd="python scripts_pose_tracking/pose_tracking.py \
        experiments/pretrained_models/pose_tracking/checkpoints/best-v*.ckpt \
        --do_test --dataset $dataset"
    echo $cmd
    eval $cmd
done