#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --job-name mlperf-training:rnnt
set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${METADATA_DIR:?METADATA_DIR not set}"
: "${SENTENCEPIECES_DIR:?SENTENCEPIECES_DIR not set}"

# Vars with defaults
: "${MLPERF_RULESET:=2.1.0}"
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${LOGDIR:=./results}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
: "${WALLTIME_DRYRUN:=$WALLTIME}"

# Other vars
LOGBASE="${DATESTAMP}"
SPREFIX="rnn_speech_recognition_pytorch_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}"
TIME_TAGS=${TIME_TAGS:-0}
NVTX_FLAG=${NVTX_FLAG:-0}
NCCL_TEST=${NCCL_TEST:-0}
SYNTH_DATA=${SYNTH_DATA:-0}
EPOCH_PROF=${EPOCH_PROF:-0}
DISABLE_CG=${DISABLE_CG:-0}

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
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_nocg"
    else
        LOGBASE="${SPREFIX}_nocg"
    fi
fi

readonly _logfile_base="${LOGDIR}/${LOGBASE}"
readonly _cont_name=rnn_speech_recognition
_cont_mounts="${DATADIR}:/datasets/,${LOGDIR}:/results"
_cont_mounts+=",${METADATA_DIR}:/metadata"
_cont_mounts+=",${SENTENCEPIECES_DIR}:/sentencepieces"

if [[ "${NVTX_FLAG}" -gt 0 ]]; then
    _cont_mounts+=",${NVMLPERF_NSIGHT_LOCATION}:/nsight"
fi
if [ "${API_LOGGING:-0}" -eq 1 ]; then
    _cont_mounts="${_cont_mounts},${API_LOG_DIR}:/logs"
fi
if [ "${REMOUNT_WORKDIR:-0}" -eq 1 ]; then
    _cont_mounts="${_cont_mounts},$(pwd):/workspace/rnnt"
fi


# MLPerf vars
MLPERF_HOST_OS=$(srun -N1 -n1 bash <<EOF
    source /etc/os-release
    source /etc/dgx-release || true
    echo "\${PRETTY_NAME} / \${DGX_PRETTY_NAME:-???} \${DGX_OTA_VERSION:-\${DGX_SWBUILD_VERSION:-???}}"
EOF
)
export MLPERF_HOST_OS

# Setup directories
( umask 0002; mkdir -p "${LOGDIR}" )
srun --ntasks="${SLURM_JOB_NUM_NODES}" mkdir -p "${LOGDIR}"

# Setup container
echo MELLANOX_VISIBLE_DEVICES="${MELLANOX_VISIBLE_DEVICES:-}"
srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONT}" --container-name="${_cont_name}" true
srun -N1 -n1 --container-name="${_cont_name}" ibv_devinfo --list
srun -N1 -n1 --container-name="${_cont_name}" nvidia-smi topo -m

echo "NCCL_TEST = ${NCCL_TEST}"
if [[ ${NCCL_TEST} -eq 1 ]]; then
    (srun --mpi=pmix --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
         --container-name="${_cont_name}" all_reduce_perf_mpi -b 98.1M -e 98.1M -d half
    ) |& tee "${LOGDIR}/${SPREFIX}_nccl.log"

fi

# Run experiments
for _experiment_index in $(seq -w 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"
	echo ":::DLPAL ${CONT} ${SLURM_JOB_ID} ${SLURM_JOB_NUM_NODES} ${SLURM_JOB_NODELIST}"

        # Print system info
        srun --nodes=1 --ntasks=1 --container-name="${_cont_name}" python -c ""

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
            srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-name="${_cont_name}" python -c "
from mlperf_logger import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)"
        fi

        # Run experiment
        srun --mpi=none --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
            --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" \
            ./run_and_time.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
    # compliance checker
    srun --ntasks=1 --nodes=1 --container-name="${_cont_name}" \
         --container-mounts="$(realpath ${LOGDIR}):/results"   \
         --container-workdir="/results"                        \
         python3 -m mlperf_logging.compliance_checker --usage training \
         --ruleset "${MLPERF_RULESET}"                                 \
         --log_output "/results/compliance_${DATESTAMP}.out"           \
         "/results/${LOGBASE}_${_experiment_index}.log" \
	 || true
done

