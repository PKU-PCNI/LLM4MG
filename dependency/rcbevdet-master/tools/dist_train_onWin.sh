#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
