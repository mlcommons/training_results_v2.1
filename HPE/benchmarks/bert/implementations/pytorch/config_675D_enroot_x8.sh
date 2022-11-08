## DL params
export BATCHSIZE=48
#export BATCHSIZE=56
export GRADIENT_STEPS=1
export LR=0.0020992
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=1083
export OPT_LAMB_BETA_1=0.83
export OPT_LAMB_BETA_2=0.925
export START_WARMUP_STEP=-25
export WARMUP_STEPS=100

export SBATCH_NETWORK=sharp
export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add  --fused_gemm_gelu "
export PHASE=2
export EVAL_ITER_START_SAMPLES=175000
export EVAL_ITER_SAMPLES=175000

## System run parms
export DGXNNODES=8
export DGXSYSTEM=675D #$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_MINUTES=30
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

export CLEAR_CACHES=1

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_675D_common.sh
