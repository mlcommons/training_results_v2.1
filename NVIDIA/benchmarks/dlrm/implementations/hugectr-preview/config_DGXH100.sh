## DL params
export CONFIG="dgx_h100.py"
REDACTED

## System run parms
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_BASE=$(( 5 + 30 * ${API_LOGGING:-0} ))
WALLTIME_MINUTES=5
export WALLTIME=$(( WALLTIME_BASE + (${NEXP:-1} * WALLTIME_MINUTES) ))
export OMPI_MCA_btl="^openib"
export CUDA_DEVICE_MAX_CONNECTIONS=3
