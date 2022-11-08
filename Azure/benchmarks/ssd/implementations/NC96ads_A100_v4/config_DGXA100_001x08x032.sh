#!/usr/bin/env bash

## DL params
export BATCHSIZE=${BATCHSIZE:-32}
export NUMEPOCHS=${NUMEPOCHS:-8}
export DATASET_DIR=${DATASET_DIR:-"/datasets/open-images-v6"}
export LR=${LR:-0.0001}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-head-fusion --disable-ddp-broadcast-buffers --fp16-allreduce --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --skip-metric-loss --cuda-graphs-syn'}

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=160
export WALLTIME=$((${NEXP:-1} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=4
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=1  # HT is on is 2, HT off is 1

# Set NCCL env variables in the config_DGXA100_1x4x56x2.sh file
## NCCL parameters
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/opt/microsoft/ncv4/topo.xml
export NCCL_GRAPH_FILE=/opt/microsoft/ncv4/graph.xml
export NCCL_ALGO=Tree
export NCCL_SHM_USE_CUDA_MEMCPY=1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_CREATE_THREAD_CONTEXT=1 
export NCCL_DEBUG_SUBSYS=ENV
export NCCL_IB_PCI_RELAXED_ORDERING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
