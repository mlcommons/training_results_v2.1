## DL params
export BATCHSIZE=112
#export BATCHSIZE=56
#export GRADIENT_STEPS=1
export GRADIENT_STEPS=2
export LR=3.5e-4
#export LR=3.75e-4
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=8041
#export MAX_STEPS=30000
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0
export WEIGHT_DECAY_RATE=0.01
export INIT_LOSS_SCALE=1024.0

export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --dwu-group-size=4 "

export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:15:00

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_R750xax4A100-PCIE-80GB_common.sh
