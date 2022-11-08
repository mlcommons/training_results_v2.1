export OMPI_MCA_btl="^openib" #To prevent deadlock between Horovd and NCCL at 96 nodes
export DALI_DONT_USE_MMAP=0 # 0 for /raid and 1 for lustre
export MXNET_EXTENDED_NORMCONV_SUPPORT=1 # supports Arch 80 NormConv fusion

## System config params
export DGXNGPU=4
export DGXNSOCKET=2
export DGXSOCKETCORES=48
export DGXHT=1  # HT is on is 2, HT off is 1
export HOROVOD_NUM_NCCL_STREAMS=1
export MXNET_HOROVOD_NUM_GROUPS=1
export HOROVOD_CYCLE_TIME=0.1
export MXNET_OPTIMIZER_AGGREGATION_SIZE=54
export MXNET_ENABLE_CUDA_GRAPHS=1
# Remove mxnet warnings
export MXNET_CUDNN_WARN_ON_IGNORED_FLAGS=0
# MxNet PP BN Heuristic
export MXNET_CUDNN_NHWC_BN_HEURISTIC_FWD=1
export MXNET_CUDNN_NHWC_BN_HEURISTIC_BWD=1
export MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_BWD=1
export MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_FWD=1

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
