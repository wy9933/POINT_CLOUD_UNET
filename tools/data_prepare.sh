#!/bin/bash

# set -x
set -e

dataset_name=$1
dataset_root=$2
dataset_target=$3
dataset_filetype=$4

if [[ -z $dataset_filetype ]]; then
dataset_filetype=npy
fi

export PYTHONPATH=.

python -u data_utils/data_prepare.py \
    --dataset_name $dataset_name \
    --dataset_root $dataset_root \
    --dataset_target $dataset_target \
    --file_type $dataset_filetype