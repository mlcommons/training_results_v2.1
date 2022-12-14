#!/bin/bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

USE_NV_INPUT=1
USE_UNCOMPRESSED_DATASET=0

export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/cuda-11.8/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

BASE_DATA_DIR=${BASE_DATA_DIR:-"/data2/zengjinle/dataset/bert_data"}

unset LC_ALL
export USE_NV_INPUT
UNCOMPRESSED_DATA_DIR=$BASE_DATA_DIR/hdf5/training-4320/hdf5_4320_shards_uncompressed
VARLENGTH_DATA_DIR=$BASE_DATA_DIR/hdf5/training-4320/hdf5_4320_shards_varlength

export DATA_DIR=$UNCOMPRESSED_DATA_DIR
export EVAL_DIR=$BASE_DATA_DIR/hdf5/eval
if [[ "$USE_NV_INPUT" == "1" && "$USE_UNCOMPRESSED_DATASET" == "0" ]]; then
  export DATA_DIR="$VARLENGTH_DATA_DIR"
  export EVAL_DIR=$BASE_DATA_DIR/hdf5/eval
else
  USE_UNCOMPRESSED_DATASET=1
fi
export USE_UNCOMPRESSED_DATASET
export TF_CKPT_PATH=$BASE_DATA_DIR/phase1/model.ckpt-28252.tf_pickled
export BERT_CONFIG_PATH=$BASE_DATA_DIR/phase1/bert_config.json

export PYTHON=${PYTHON:-"python3.8"}
export MPIRUN_LOCAL=${MPIRUN_LOCAL:-"0"}

WORLD_RANK=${OMPI_COMM_WORLD_RANK:-"0"}
WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-"1"}


if [[ $MPIRUN_LOCAL == "1" ]]; then
  WORLD_RANK=$(($WORLD_RANK+$NODE_RANK*$GPU_NUM_PER_NODE*2)) 
  WORLD_SIZE=$(($WORLD_SIZE*$PADDLE_TRAINERS_NUM))
fi

ENV_FILE="env.${WORLD_RANK}"
$PYTHON prepare_env.py ${WORLD_RANK} ${WORLD_SIZE} >$ENV_FILE 
source $ENV_FILE

if [[ $IS_TRAINER -eq "1" ]]; then
  # NSYS_CMD="nsys profile --stats=true -t cuda,nvtx,cublas,osrt,mpi -y 40 -f true -o paddle_perf_${PADDLE_TRAINER_ID}"

  BIND_CMD=`$PYTHON print_bind_cmd.py --nsockets_per_node=${DGXNSOCKET:-"2"} \
	     --ncores_per_socket=${DGXSOCKETCORES:-"32"} \
	     --local_rank ${LOCAL_RANK} \
	     --nproc_per_node ${NPROC_PER_NODE}`
fi


export FLAGS_sync_nccl_allreduce=0
export FLAGS_fraction_of_gpu_memory_to_use=0.99
#export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_call_stack_level=2
export FLAGS_use_fast_math=0
#export FLAGS_check_nan_inf=1
#export CUDA_LAUNCH_BLOCKING=1
#export FLAGS_benchmark=1
export FLAGS_enable_nvtx=1

export FLAGS_inplace_addto_external_ops=custom_fused_dense_grad

batch_size=48
eval_batch_size=63
use_amp=True
use_pure_fp16=True

max_steps=1059
log_freq=0
eval_iter_start_samples=150000
eval_iter_samples=150000
max_seq_length=512

dense_seq_output=True
unpad=True
unpad_fmha=True
use_flash_attn=True
fused_bias_mha=True
fused_bias_fc=True
## can be False or True 
weight_transpose=True

fused_dropout_add_ln=True
exchange_padding=True
cpu_exchange_padding=True

distributed_lamb=True

unpad_embed=True
unpad_fmha_mke_opt=True

sort_eval_data=True

fused_dense_gelu_dense=True

LOG_DIR="log_${PADDLE_TRAINERS_NUM}"
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/worker.${ROLE}.${PADDLE_TRAINER_ID}

#export FLAGS_lamb_allreduce_first=1
#export FLAGS_use_multi_tensor_apply=1
export FLAGS_max_inplace_grad_add=2

if [[ "$exchange_padding" == "true" || "$exchange_padding" == "True" ]]; then
  if [[ "$cpu_exchange_padding" == "true" || "$cpu_exchange_padding" == "True" ]]; then
    export DATA_DIR="$UNCOMPRESSED_DATA_DIR"
  fi
fi

#export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=xgbe0
export NCCL_IB_GID_INDEX=0
export NCCL_COLLNET_ENABLE=1
export SHARP_COLL_SHARPD_SOCKET_NAME=sharpd_2.5.0

if [[ $LOCAL_RANK -lt 4 ]]; then
  mlx_id=$LOCAL_RANK
else
  mlx_id=$(($LOCAL_RANK+2))
fi
# export OMPI_MCA_btl_openib_if_include=mlx5_${mlx_id}
# export UCX_NET_DEVICES=${OMPI_MCA_btl_openib_if_include}:1  

export INIT_LOSS_SCALE=4096.0

$NSYS_CMD $BIND_CMD $PYTHON -u run_pretrain.py \
   --max_predictions_per_seq 76 \
   --train_batch_size $batch_size   \
   --eval_batch_size $eval_batch_size \
   --sort_eval_data $sort_eval_data \
   --learning_rate 0.0020992 \
   --weight_decay 0.1 \
   --lamb_epsilon 1e-6 \
   --start_warmup_step 0 \
   --warmup_proportion 0.0 \
   --warmup_steps 0 \
   --input_dir $DATA_DIR \
   --log_freq $log_freq \
   --max_steps $max_steps \
   --tf_ckpt_path $TF_CKPT_PATH \
   --bert_config_path $BERT_CONFIG_PATH \
   --unpad $unpad \
   --unpad_fmha $unpad_fmha \
   --unpad_fmha_mke_opt $unpad_fmha_mke_opt \
   --use_flash_attn $use_flash_attn \
   --unpad_embed $unpad_embed \
   --fused_bias_mha $fused_bias_mha \
   --fused_bias_fc $fused_bias_fc \
   --fused_dropout_add_ln $fused_dropout_add_ln \
   --fused_dense_gelu_dense $fused_dense_gelu_dense \
   --weight_transpose $weight_transpose \
   --max_seq_length $max_seq_length \
   --eval_dir $EVAL_DIR \
   --distributed_lamb $distributed_lamb \
   --exchange_padding $exchange_padding \
   --cpu_exchange_padding $cpu_exchange_padding \
   --local_exchange_padding False \
   --seed $SEED \
   --exchange_padding_nbatch 100 \
   --use_uncompressed_dataset $USE_UNCOMPRESSED_DATASET \
   --dense_seq_output $dense_seq_output \
   --gradient_accumulation_steps 1 \
   --opt_lamb_beta_1 0.60466 \
   --opt_lamb_beta_2 0.85437 \
   --enable_addto True \
   --nproc_per_node $NPROC_PER_NODE \
   --use_pure_fp16 $use_pure_fp16 \
   --use_amp $use_amp 2>&1 | tee $LOG_FILE
