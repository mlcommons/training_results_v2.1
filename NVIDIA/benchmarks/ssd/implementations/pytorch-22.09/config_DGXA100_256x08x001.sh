#!/bin/bash

## DL params
export BATCHSIZE=${BATCHSIZE:-1}
export NUMEPOCHS=${NUMEPOCHS:-10}
export LR=${LR:-0.000135}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-head-fusion --disable-ddp-broadcast-buffers --fp16-allreduce --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --dali-eval-cache --skip-metric-loss --cuda-graphs-syn --sync-after-graph-replay --async-coco'}

## System run parms
export DGXNNODES=256
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=15
export WALLTIME=$((${NEXP:-1} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1

## System data paths
MLPERF_LOGIN_HOST="${MLPERF_LOGIN_HOST:-$(hostname | sed -E 's/-.*$//')}"
MLPERF_HOST_CONFIG=$(dirname "${BASH_SOURCE[0]}")/config_data_"${MLPERF_LOGIN_HOST}".sh
echo "${MLPERF_HOST_CONFIG}"
if [ -f "${MLPERF_HOST_CONFIG}" ]; then
    source "${MLPERF_HOST_CONFIG}"
fi
