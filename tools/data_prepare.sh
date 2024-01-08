#!/bin/bash

# set -x
set -e

dataset_name=$1
dataset_source=$2
dataset_root=$3

if [[ -z $dataset_filetype ]]; then
dataset_filetype=npy
fi

export PYTHONPATH=.

python -u data_utils/data_prepare.py \
    --dataset_name $dataset_name \
    --dataset_source $dataset_source \
    --dataset_root $dataset_root