#!/bin/bash
#SBATCH --job-name mlperf-dlrm:hugectr
#SBATCH -t 00:30:00

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"

# Vars with defaults
: "${MLPERF_RULESET:=2.1.0}"
: "${NEXP:=10}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
: "${MOUNTS:=${DATADIR}:/data}"
: "${LOGDIR:=./results}"
if [ "${API_LOGGING:-0}" -eq 1 ]; then
    MOUNTS="${MOUNTS},${API_LOG_DIR}:/logs"
fi
REDACTED
REDACTED
REDACTED

# make sure the results directory exists on the host
( umask 0002; mkdir -p "${LOGDIR}" )

# Other vars
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=dlrm

# Setup container
echo MELLANOX_VISIBLE_DEVICES="${MELLANOX_VISIBLE_DEVICES:-}"
srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONT}" --container-name="${_cont_name}" true
srun -N1 -n1 --container-name="${_cont_name}" ibv_devinfo --list
srun -N1 -n1 --container-name="${_cont_name}" nvidia-smi topo -m

# Run experiments
for _experiment_index in $(seq -w 1 "${NEXP}"); do
    (
        echo ":::DLPAL ${CONT} ${SLURM_JOB_ID} ${SLURM_JOB_NUM_NODES} ${SLURM_JOB_NODELIST}"
        if [[ $CLEAR_CACHES == 1 ]]; then
            srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
            srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="${SLURM_JOB_NUM_NODES}" --container-name="${_cont_name}" python3 -c "
import mlperf_logging.mllog as mllog
mllogger = mllog.get_mllogger()
mllogger.event(key=mllog.constants.CACHE_CLEAR, value=True)"
        fi
        echo "Beginning trial ${_experiment_index} of ${NEXP}"
        srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
             --container-name="${_cont_name}" --container-mounts="${MOUNTS}" \
             ${LOGGER:-} ./run_and_time.sh
    ) |& tee "${_logfile_base}_raw_${_experiment_index}.log"

    # Sorting the MLPerf compliance logs by timestamps
    grep ":::.L..." "${_logfile_base}_raw_${_experiment_index}.log" | sort -k5 -n -s | tee "${_logfile_base}_${_experiment_index}.log"
    srun --ntasks=1 --nodes=1 --container-name="${_cont_name}" \
	 --container-mounts="$(realpath ${LOGDIR}):/results"   \
	 --container-workdir="/results"                        \
	 python3 -m mlperf_logging.compliance_checker --usage training \
	 --ruleset "${MLPERF_RULESET}"                                 \
	 --log_output "/results/compliance_${DATESTAMP}.out"           \
	 "/results/${DATESTAMP}_${_experiment_index}.log" \
	 || true
done
