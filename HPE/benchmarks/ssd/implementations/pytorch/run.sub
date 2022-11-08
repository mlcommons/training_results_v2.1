#!/bin/bash
#SBATCH --job-name=ssd_mlpv21
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=4
#SBATCH --cpus-per-task=8
##
#SBATCH --partition=mlperf
#SBATCH --time=02:00:00
#SBATCH --exclusive

# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

set -eux

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
# Vars with defaults
: "${MLPERF_RULESET:=2.1.0}"
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${WORK_DIR:=/workspace/ssd}"
: "${CONT_NAME:=single_stage_detector}"
# ci automagically sets this correctly on Selene
: "${LOGDIR:=./results}"

# Scaleout brdige
: "${NVTX_FLAG:=0}"
: "${TIME_TAGS:=0}"
: "${NCCL_TEST:=0}"
: "${SYNTH_DATA:=0}"
: "${EPOCH_PROF:=0}"
: "${DISABLE_CG:=0}"

# API Logging defaults
: "${API_LOGGING:=0}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir

export WORLD_SIZE=${SLURM_NPROCS}
export RANK=$((${DGXNGPU}*${SLURM_NODEID}+${SLURM_LOCALID}))
export MASTER_ADDR=${SLURMD_NODENAME}
export MASTER_PORT=5678
echo "[HPE] setting distributed info WORLD_SIZE=$WORLD_SIZE, MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

LOGBASE="${DATESTAMP}"
SPREFIX="single_stage_detector_pytorch_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}"

if [ ${TIME_TAGS} -gt 0 ]; then
    LOGBASE="${SPREFIX}_mllog"
fi
if [ ${NVTX_FLAG} -gt 0 ]; then
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_nsys"
    else
        LOGBASE="${SPREFIX}_nsys"
    fi

    if [[ ! -d "${NVMLPERF_NSIGHT_LOCATION}" ]]; then
	echo "$NVMLPERF_NSIGHT_LOCATION doesn't exist on this system!" 1>&2
	exit 1
    fi
fi
if [ ${SYNTH_DATA} -gt 0 ]; then
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_synth"
    else
        LOGBASE="${SPREFIX}_synth"
    fi
fi
if [ ${EPOCH_PROF} -gt 0 ]; then
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_epoch"
    else
        LOGBASE="${SPREFIX}_epoch"
    fi
fi
if [ ${DISABLE_CG} -gt 0 ]; then
    EXTRA_PARAMS=$(echo $EXTRA_PARAMS | sed 's/--cuda-graphs//')
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_nocg"
    else
        LOGBASE="${SPREFIX}_nocg"
    fi
fi
export LOGDIR="${LOGDIR}/${SLURM_JOB_ID}"

# do we need to fetch the data from lustre to /raid/scratch?
if [[ "${LOCALDISK_FROM_SQUASHFS:-}" ]]; then
    # LOCALDISK_FROM_SQUASHFS should be the path/name of a squashfs file on /lustre
    echo "fetching ${LOCALDISK_FROM_SQUASHFS}"
    dd bs=4M if="${LOCALDISK_FROM_SQUASHFS}" of=/raid/scratch/tmp.sqsh oflag=direct
    echo "unsquashing /raid/scratch/tmp.sqsh"
    time unsquashfs -no-progress -dest /raid/scratch/local-root /raid/scratch/tmp.sqsh
fi

#######################################
# Print mount locations from file.
# Arguments:
#   arg1 - path to file containing volume mounting points
# Returns:
#   String containing comma-separated mount pairs list
#######################################
func_get_container_mounts() {
  declare -a mount_array
  readarray -t mount_array <<<$(egrep -v '^#' "${1}")
  local cont_mounts=$(envsubst <<< $(printf '%s,' "${mount_array[@]}" | sed 's/[,]*$//'))
  echo $cont_mounts
}

#######################################
# CI does not make the current directory the model directory. It is two levels up, which is different than a command line launch.
# This function looks in ${PWD} and then two levels down for a file, and updates the path if necessary.
# Arguments:
#   arg1 - expected path to file
#   arg2 - model path (e.g., language_model/pytorch/)
# Returns:
#   String containing updated (or original) path
#######################################
func_update_file_path_for_ci() {
  declare new_path
  if [ -f "${1}" ]; then
    new_path="${1}"
  else
    new_path="${2}/${1}"
  fi

  if [ ! -f "${new_path}" ]; then
    echo "File not found: ${1}"
    exit 1
  fi

  echo "${new_path}"
}
CONT_MOUNTS=$(func_get_container_mounts $(func_update_file_path_for_ci mounts.txt ${PWD}))

readonly LOG_FILE_BASE="${LOGDIR}/${LOGBASE}"
CONT_MOUNTS+=",${DATADIR}:/datasets/open-images-v6,${LOGDIR}:/results"
CONT_MOUNTS+=",${BACKBONE_DIR}:/root/.cache/torch"
CONT_MOUNTS+=",${BACKBONE_DIR}:/workspace/ssd/root/.cache/torch"
CONT_MOUNTS+=",${PWD}:${WORK_DIR}"

if [[ "${NVTX_FLAG}" -gt 0 ]]; then
    CONT_MOUNTS="${CONT_MOUNTS},${NVMLPERF_NSIGHT_LOCATION}:/nsight"
fi
# API Logging
if [ "${API_LOGGING}" -eq 1 ]; then
    CONT_MOUNTS="${CONT_MOUNTS},${API_LOG_DIR}:/logs"
fi

# # Setup directories
# ( umask 0002; mkdir -p "${LOGDIR}" )
# srun --ntasks="${SLURM_JOB_NUM_NODES}" mkdir -p "${LOGDIR}"

# Setup container
echo MELLANOX_VISIBLE_DEVICES="${MELLANOX_VISIBLE_DEVICES:-}"
# srun \
#     --ntasks="1" \
#     --container-image="${CONT}" \
#     --container-name="${CONT_NAME}" \
#     --container-mounts="${CONT_MOUNTS}" 'echo $MELLANOX_VISIBLE_DEVICES'
srun \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --container-image="${CONT}" \
    --container-name="${CONT_NAME}" \
    --container-mounts="${CONT_MOUNTS}" true
# srun -N1 -n1 --container-name="${CONT_NAME}" ibv_devinfo --list
# srun -N1 -n1 --container-name="${CONT_NAME}" nvidia-smi topo -m

# echo "NCCL_TEST = ${NCCL_TEST}"
# if [[ ${NCCL_TEST} -eq 1 ]]; then
#     (srun --mpi=pmix --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
#          --container-name="${CONT_NAME}" all_reduce_perf_mpi -b 33260119 -e 33260119 -d half -G 1    ) |& tee "${LOGDIR}/${SPREFIX}_nccl.log"

# fi

# Run experiments
for _experiment_index in $(seq -w 1 "${NEXP}"); do
    (

        echo "Beginning trial ${_experiment_index} of ${NEXP}"
	echo ":::DLPAL ${CONT} ${SLURM_JOB_ID} ${SLURM_JOB_NUM_NODES} ${SLURM_JOB_NODELIST}"

        # Print system info
        # srun -N1 -n1 --container-name="${CONT_NAME}" python -c ""

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
#             srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-name="${CONT_NAME}" python -c "
# from mlperf_logger import mllogger
# mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)"
            srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-mounts="${CONT_MOUNTS}" --container-name="${CONT_NAME}" python ${WORK_DIR}/drop.py 
        fi

        # Run experiment
        srun \
            --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" \
            --ntasks-per-node="${DGXNGPU}" \
            --container-name="${CONT_NAME}" \
            --container-mounts="${CONT_MOUNTS}" \
            --container-workdir=${WORK_DIR} \
            ./run_and_time.sh
    ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"
    # # compliance checker
    # srun --ntasks=1 --nodes=1 --container-name="${CONT_NAME}" \
    #      --container-mounts="$(realpath ${LOGDIR}):/results"   \
    #      --container-workdir="/results"                        \
    #      python3 -m mlperf_logging.compliance_checker --usage training \
    #      --ruleset "${MLPERF_RULESET}"                                 \
    #      --log_output "/results/compliance_${DATESTAMP}.out"           \
    #      "/results/${LOGBASE}_${_experiment_index}.log" \
	#  || true
done
