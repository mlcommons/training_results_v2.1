#
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_multi.sh
export NUM_GPUS_TRAIN=8
# System run parms
export DGXNNODES=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

#
export NUM_ITERATIONS=80
