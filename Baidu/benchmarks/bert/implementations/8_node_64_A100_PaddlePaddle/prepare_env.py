# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import subprocess
import distutils.util


def str2bool(s):
    return True if distutils.util.strtobool(s) else False


def split(s, delim):
    return [item.strip() for item in s.split(delim) if item.strip()]


def get_gpu_list():
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        output = subprocess.check_output(
            ["sh", "-c", "nvidia-smi --list-gpus | wc -l"])
        gpu_num = int(output.strip())
        gpus = list(range(gpu_num))
    else:
        gpus = [int(g) for g in split(gpus, ',')]
    return gpus


def to_str(s):
    if isinstance(s, bool):
        return '1' if s else '0'
    elif isinstance(s, (list, tuple)):
        return ','.join([to_str(item) for item in s])
    else:
        return str(s)


def prepare_envs(global_rank, global_size):
    assert global_size % 2 == 0

    trainer_rank = int(global_rank / 2) * 2
    reader_rank = trainer_rank + 1

    ep_env_name = "PADDLE_TRAINER_ENDPOINTS"
    ep_env_value = os.getenv(ep_env_name)
    if not ep_env_value:
        local_size = global_size
    else:
        ips = [
            ip_port.split(':')[0] for ip_port in ep_env_value.split(",")
            if ip_port.strip()
        ]
        cur_ip = ips[int(trainer_rank / 2)]
        local_size = sum([1 for ip in ips if ip == cur_ip]) * 2

    trainer_ranks = list(range(0, global_size, 2))
    reader_ranks = list(range(1, global_size, 2))
    trainer_id = int(trainer_rank / 2)
    trainer_num = int(global_size / 2)
    local_trainer_num = int(local_size / 2)
    local_gpus = get_gpu_list()
    local_rank = trainer_id % local_trainer_num
    local_gpu_id = local_gpus[local_rank]

    is_trainer = (global_rank == trainer_rank)

    default_port = int(os.getenv('DEFAULT_PORT', '60001'))
    if not ep_env_value:
        assert global_size == local_size, "{} must be provided when running on multiple node".format(
            ep_env_name)
        ep_env_value = ','.join([
            '127.0.0.1:{}'.format(str(default_port + i))
            for i in range(local_trainer_num)
        ])

    eps = split(ep_env_value, ',')

    is_local_mpirun = str2bool(os.getenv('MPIRUN_LOCAL', '0'))
    if is_local_mpirun:
        mpi_envs = {
            "MPI_TRAINER_RANKS": list(range(0, local_size, 2)),
            "MPI_READER_RANKS": list(range(1, local_size, 2)),
            "MPI_TRAINER_RANK": local_rank * 2,
            "MPI_READER_RANK": local_rank * 2 + 1,
        }
    else:
        mpi_envs = {
            "MPI_TRAINER_RANKS": trainer_ranks,
            "MPI_READER_RANKS": reader_ranks,
            "MPI_TRAINER_RANK": trainer_rank,
            "MPI_READER_RANK": reader_rank,
        }

    ret = {
        "IS_TRAINER": is_trainer,
        "IS_READER": not is_trainer,
        "TRAINER_RANK": trainer_rank,
        "READER_RANK": reader_rank,
        "LOCAL_RANK": local_rank,
        "TRAINER_RANKS": trainer_ranks,
        "READER_RANKS": reader_ranks,
        "PADDLE_TRAINER_ID": trainer_id,
        "PADDLE_TRAINERS_NUM": trainer_num,
        "PADDLE_CURRENT_ENDPOINT": eps[trainer_id],
        ep_env_name: ep_env_value,
        "CUDA_VISIBLE_DEVICES": local_rank if is_trainer else "",
        "NVIDIA_VISIBLE_DEVICES": local_rank if is_trainer else "",
        "FLAGS_selected_gpus": 0
        if is_trainer else None,  # local_gpu_id if is_trainer else None,
        "ROLE": "trainer" if is_trainer else "reader",
        "NPROC_PER_NODE": len(local_gpus),
    }
    ret.update(mpi_envs)
    return ret


def print_env(global_rank, global_size):
    assert global_rank < global_size
    envs = prepare_envs(global_rank, global_size)
    for k, v in envs.items():
        if v is not None:
            print('export {}={}'.format(k, to_str(v)))


if __name__ == "__main__":
    assert len(sys.argv) == 3
    global_rank = int(sys.argv[1])
    global_size = int(sys.argv[2])
    print_env(global_rank, global_size)
