## DL params
export BATCHSIZE=48
export GRADIENT_STEPS=1
#export LR=0.0020992
export LR=0.002
export MAX_SAMPLES_TERMINATION=4500000
#export MAX_STEPS=1059
export MAX_STEPS=2254
export OPT_LAMB_BETA_1=0.66
export OPT_LAMB_BETA_2=0.996
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0
#export WARMUP_STEPS=0
export WEIGHT_DECAY_RATE=0.01
export INIT_LOSS_SCALE=4096.0

export SBATCH_NETWORK=sharp
export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --dwu-group-size=4 --fused_bias_fc --fused_bias_mha --fused_dropout_add  --fused_gemm_gelu "
export PHASE=2
export EVAL_ITER_START_SAMPLES=175000
export EVAL_ITER_SAMPLES=175000

## System run parms
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_MINUTES=15
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

## System config params
export DGXNGPU=4
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1
#export SLURM_NTASKS=${DGXNGPU}
#: "${SLURM_NTASKS:= DGXNGPU}"
#export SLURM_NTASKS
export CUDA_VISIBLE_DEVICES="0,1,2,3"

NCCL_SOCKET_IFNAME=mlx5_0,mlx5_1
# System name
export MLPERF_SUBMISSION_ORG="Dell"
export MLPERF_SUBMISSION_PLATFORM="${DGXSYSTEM}"
export OMP_NUM_THREADS=8

