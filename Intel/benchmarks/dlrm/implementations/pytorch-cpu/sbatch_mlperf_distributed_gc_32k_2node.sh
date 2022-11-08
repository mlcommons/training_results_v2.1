#!/bin/bash

# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
source <CONDA_PATH>/bin/activate dlrm
cd <WORKSPACE>

MODEL_DIR=${MODEL_DIR-'./'}
if [ ! -e "${MODEL_DIR}/dlrm_s_pytorch.py"  ]; then
    echo "Could not find the script of dlrm_s_pytorch.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the dlrm_s_pytorch.py exist at the: \${MODEL_DIR}/dlrm_s_pytorch.py"
    exit 1
fi

MODEL_SCRIPT=${MODEL_DIR}/dlrm_s_pytorch.py
OUTPUT_DIR=${OUTPUT_DIR-'./'}
DATASET_DIR=${DATASET_DIR-'/data/terabyte_input/'}
PRECISION=${PRECISION-'bf16'}
NP=${NP-'4'}
PPN=${PPN-'1'}
echo "PRECISION: ${PRECISION}"
echo "DATASET_DIR: ${DATASET_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi


# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
LOG=${OUTPUT_DIR}/dlrm_training_log_32k_2node_slurm/${PRECISION}
rm -rf ${LOG}
mkdir -p ${LOG}

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "running bf16 path"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    echo "running fp32 path"
else
    echo "The specified PRECISION '${PRECISION}' is unsupported."
    echo "Supported PRECISIONs are: fp32, avx-fp32, bf16"
    exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`

CCL_WORKER_COUNT=8
PROC_MASK_STR=`BC_LINE_LENGTH=0 bc <<<"obase=16;(2^${CORES} - 1)"`
for (( I=0; I < 2 ; I++)) ; do
  SHFT=$(( I * CORES ))
  if [ $SHFT -lt 4 ] ; then
  ZEROS=""
  else
  ZEROS=`printf "%0*X" $(( SHFT / 4 ))`
  fi
  for((P=0;P < CCL_WORKER_COUNT ; P++)); do CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY} $((I * CORES + P + 2 * CORES))" ; done
  SMASK=${PROC_MASK_STR}${ZEROS}
  MASKS[$I]="0x$SMASK"
done

LOG_0="${LOG}/socket_0"
seed_num=$(date +%s)
BS=32768
TEST_BS=131072
MAX_RANGE=40000000
dlrm_extra_option="--hybrid-gradient-emb --ipex-merged-emb --bf16 --ipex-interaction --enable-mlp-fusion --allreduce-wait  "
torch_ccl_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $torch_ccl_path/env/setvars.sh
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
export KMP_AFFINITY=granularity='fine,compact,1,0'
export KMP_BLOCKTIME=1

scontrol show hostname $SLURM_NODELIST >hostfile_2
cat hostfile_2
export MASTER_ADDR=$(head -1 hostfile_2)
export I_MPI_PIN_DOMAIN=[`echo ${MASKS[@]} | tr " " ","`]
export CCL_WORKER_AFFINITY=`echo ${CCL_WORKER_AFFINITY} | tr " " ","`
export CCL_WORKER_COUNT=${CCL_WORKER_COUNT}
export CCL_MNIC=local
export CCL_MNIC_COUNT=2
export CCL_MNIC_NAME=irdma-cvl01tf2,irdma-cvl02tf2,irdma-cvl11tf2,irdma-cvl12tf2


FI_PROVIDER=psm3 CCL_ALLREDUCE=rabenseifner OMP_NUM_THREADS=${CORES}  PSM3_IDENTIFY=1 PSM3_ALLOW_ROUTERS=1 PSM3_RDMA=1 PSM3_RV_MR_CACHE_SIZE=8192 FI_PROVIDER_PATH=/usr/lib64/libfabric  mpiexec.hydra --np $NP --ppn $PPN --hostfile hostfile_2 python -u $MODEL_SCRIPT --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=${MAX_RANGE} --data-generation=dataset --data-set=terabyte --raw-data-file=$DATASET_DIR/day --processed-data-file=$DATASET_DIR/terabyte_processed.npz --loss-function=bce --round-targets=True --num-workers=0 --test-num-workers=0  --learning-rate=18.0 --mini-batch-size=${BS} --print-freq=1024 --print-time --test-freq=6400 --test-mini-batch-size=${TEST_BS} --lr-num-warmup-steps=8000   --lr-decay-start-step=70000 --lr-num-decay-steps=30000 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --numpy-rand-seed=$seed_num --ipex-interaction $dlrm_extra_option $ARGS $dlrm_extra_option 2>&1|tee $LOG_0


