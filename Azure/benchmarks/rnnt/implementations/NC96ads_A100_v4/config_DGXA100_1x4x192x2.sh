## System config params
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXNGPU=4
export DWU_GROUP_SIZE=4
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1

## Run specific params
export DATADIR="/raid/datasets/rnnt/"
export METADATA_DIR="/lustre/fsw/mlperf-ci/tokenized/"
export SENTENCEPIECES_DIR="/lustre/fsw/mlperf-ci/sentpiece"
export BATCHSIZE=384
export EVAL_BATCHSIZE=676
export GRAD_ACCUMULATION_STEPS=2
WALLTIME_MINUTES=120
export WALLTIME=${WALLTIME:-$(( ${NEXP:-1} * ${WALLTIME_MINUTES} ))}
export MAX_SYMBOL=300
export DATA_CPU_THREADS=16

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_1536.sh

## Opt flag
export FUSE_RELU_DROPOUT=true
export MULTI_TENSOR_EMA=true
export BATCH_EVAL_MODE=cg_unroll_pipeline
export APEX_LOSS=fp16
export APEX_JOINT=pack_w_relu_dropout
export AMP_LVL=2
export BUFFER_PREALLOC=true
export VECTORIZED_SA=true
export EMA_UPDATE_TYPE=fp16
export DIST_LAMB=true
export MULTILAYER_LSTM=true
export ENABLE_PREFETCH=true
export TOKENIZED_TRANSCRIPT=true
export VECTORIZED_SAMPLER=true
export DIST_SAMPLER=true
export MIN_SEQ_SPLIT_LEN=20
export FC_IMPL=apex_fused_dense
export PRE_SORT_FOR_SEQ_SPLIT=true
export LOG_FREQUENCY=1000

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
