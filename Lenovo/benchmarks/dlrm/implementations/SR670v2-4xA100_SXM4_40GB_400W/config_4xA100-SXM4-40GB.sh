## DL params
export BATCH_SIZE=55296
export DGXNGPU=4

export CONFIG="dgx_a100_40g.py"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:10:00
export OMPI_MCA_btl="^openib"
export MOUNTS=/mlperf/training-2.1/dlrm/data:/raid
export CUDA_DEVICE_MAX_CONNECTIONS=2
