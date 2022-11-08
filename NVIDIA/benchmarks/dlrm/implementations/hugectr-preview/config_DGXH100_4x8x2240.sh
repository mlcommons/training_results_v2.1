## DL params
export CONFIG="dgx_h100_4x8x2240.py"
REDACTED

## System run parms
export DGXNNODES=4
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_BASE=$(( 5 + 60 * ${API_LOGGING:-0} ))
WALLTIME_MINUTES=2
export WALLTIME=$(( WALLTIME_BASE + (${NEXP:-1} * WALLTIME_MINUTES) ))
export CUDA_DEVICE_MAX_CONNECTIONS=3
export SBATCH_NETWORK=sharp
export SBATCH_OTHER_PARAMS="--switches 1@00:10:00"
