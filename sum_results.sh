#!/bin/bash

export DATAROOT=./results/

datasets=($(ls $DATAROOT))

for dataset in ${datasets[@]}
do
    result=$(tail -n 1 "$DATAROOT$dataset/test_odom_error.txt")
    echo "$dataset $result" >> results.txt
done