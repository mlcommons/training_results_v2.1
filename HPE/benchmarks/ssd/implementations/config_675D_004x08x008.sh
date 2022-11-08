#!/bin/bash

## DL params
export BATCHSIZE=${BATCHSIZE:-8}
export NUMEPOCHS=${NUMEPOCHS:-6}
export LR=${LR:-0.0001}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-head-fusion --disable-ddp-broadcast-buffers --fp16-allreduce --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --skip-metric-loss --cuda-graphs-syn --sync-after-graph-replay --async-coco'}

## System run parms
export DGXNNODES=4

source $(dirname ${BASH_SOURCE[0]})/config_675D_common.sh

WALLTIME_MINUTES=90
export WALLTIME=$((${NEXP:-1} * ${WALLTIME_MINUTES}))
