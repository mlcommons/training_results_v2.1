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

set -e

ldconfig

export PYTHON=${PYTHON:-"python3.8"}
export DEFAULT_PORT=${DEFAULT_PORT:-"60001"}

export MPIRUN_LOCAL="0"

export GPU_NUM_PER_NODE=`$PYTHON -c "import paddle; print(paddle.device.cuda.device_count())"`
export PADDLE_TRAINERS_NUM=${PADDLE_TRAINERS_NUM:-"1"}
WORLD_SIZE=$(($GPU_NUM_PER_NODE*$PADDLE_TRAINERS_NUM))

for i in `seq 1 10`; do
  for port_idx in `seq 0 $(($PADDLE_TRAINERS_NUM-1))`;
  do
    port=$(($DEFAULT_PORT+$port_idx))
    ((lsof -i:$port | tail -n +2 | awk '{print $2}' | xargs kill -9) || true) >/dev/null 2>&1
  done
done

set -ex

export SEED=${SEED:-"$RANDOM"}

CMD="bash run_benchmark.sh"

echo $BASE_DATA_DIR

# bash kill_grep.sh $PYTHON || true

bash kill_grep.sh Nsight || true

export PADDLE_TRAINER_ENDPOINTS=`$PYTHON gen_ep.py`
export NODE_RANK=`$PYTHON gen_node_rank.py`

host_config=`$PYTHON print_mpirun_host_cmd.py`
mca_config="--mca btl self,tcp --mca btl_tcp_if_include xgbe0 "
num_process=$(($WORLD_SIZE*2))
export LD_PRELOAD=/usr/local/lib/libjemalloc.so:$LD_PRELOAD

if [[ $MPIRUN_LOCAL == "1" ]]; then
  host_config=""
  # mca_config="-mca"
  num_process=$(($GPU_NUM_PER_NODE*2))
fi

ORTERUN=`which orterun`
mpirun="$ORTERUN --allow-run-as-root \
     -np $num_process $host_config $mca_config \
     --bind-to none -x DEFAULT_PORT \
     -x PATH \
     -x LD_LIBRARY_PATH -x PYTHON \
     -x CUDA_VISIBLE_DEVICES -x SEED \
     -x PADDLE_TRAINER_ENDPOINTS \
     -x BASE_DATA_DIR \
     -x LD_PRELOAD \
     -x NODE_RANK \
     -x MPIRUN_LOCAL \
     -x PADDLE_TRAINERS_NUM \
     -x GPU_NUM_PER_NODE"

#NSYS_CMD="nsys profile --stats=true -t cuda,nvtx,cublas -y 40 -d 80 -f true -o paddle_perf"
$mpirun $CMD

# bash kill_grep.sh run_and_time || true
# bash kill_grep.sh python || true
