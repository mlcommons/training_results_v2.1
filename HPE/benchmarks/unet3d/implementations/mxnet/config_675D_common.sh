#!/bin/bash

# Minimum runs for a production run is 40
# https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#12-benchmark-results
export NEXP=${NEXP:-45}

export DGXSYSTEM=675D

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export CLEAR_CACHES=1
export DGXHT=1         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export MELLANOX_VISIBLE_DEVICES="0,1,2,4"
export MELLANOX_VISIBLE_DEVICES="all"
export NCCL_TOPO_FILE="/workspace/image_segmentation/675D_nic_affinity.xml"

## DL params
export OPTIMIZER="nag"
export LR="2.0"
export LR_WARMUP_EPOCHS="1000"
export MAX_EPOCHS=${MAX_EPOCHS:-10000}
export START_EVAL_AT=1000
export QUALITY_THRESHOLD="0.908"
export INPUT_BATCH_MULTIPLIER=4
export NUM_WORKERS=4

export OMP_NUM_THREADS=1
export HOROVOD_CYCLE_TIME=0.1
#export MXNET_HOROVOD_NUM_GROUPS=20
export OMPI_MCA_btl=^openib
#export NCCL_MAX_RINGS=8
#export NCCL_BUFFSIZE=2097152
#export NCCL_NET_GDR_READ=1
#export HOROVOD_FUSION_THRESHOLD=67108864
#export HOROVOD_NUM_NCCL_STREAMS=1
#export HOROVOD_BATCH_D2D_MEMCOPIES=1
#export HOROVOD_GROUPED_ALLREDUCES=1
