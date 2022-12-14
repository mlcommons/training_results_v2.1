## System config params
export DGXNGPU=4
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
export CUDA_VISIBLE_DEVICES="0,1,2,3"

## Data Paths
export DATADIR="/raid/datasets/bert/hdf5/4320_shards"
export EVALDIR="/raid/datasets/bert/hdf5/eval_4320_shard"
export DATADIR_PHASE2="/raid/datasets/bert/hdf5/4320_shards"
export CHECKPOINTDIR="$CI_BUILDS_DIR/$SLURM_ACCOUNT/$CI_JOB_ID/ci_checkpoints"
export RESULTSDIR="$CI_BUILDS_DIR/$SLURM_ACCOUNT/$CI_JOB_ID/results"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/raid/datasets/bert/checkpoints/checkpoint_phase1"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"

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
