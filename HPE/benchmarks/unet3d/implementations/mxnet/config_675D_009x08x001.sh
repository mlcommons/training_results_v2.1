source $(dirname ${BASH_SOURCE[0]})/config_675D_common.sh

## DL params
export BATCH_SIZE="1"
export VAL_BATCH_SIZE="7"
export EVALUATE_EVERY=20
export ASYNC_PARAMS=" --nodes_for_eval 2 "
export EXTRA_PARAMS=${EXTRA_PARAMS:-"-sts -ucl"}
export PRECISION=${PRECISION:-"--static_cast -sls 8192 -gpf 16 --fp16in "}

export SBATCH_NETWORK=sharp

## System run parms
export DGXNNODES=9
WALLTIME_MINUTES=24
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))
