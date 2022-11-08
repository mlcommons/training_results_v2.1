#!/bin/bash

set -uxo pipefail

: "${PADDLE_TRAINER_ENDPOINTS:?PADDLE_TRAINER_ENDPOINTS not set}"
: "${PADDLE_TRAINER_PORTS:?PADDLE_TRAINER_PORTS not set}"
: "${PADDLE_TRAINERS_NUM:?PADDLE_TRAINERS_NUM not set}"
: "${CONT_NAME:?CONT_NAME not set}"

# Vars with defaults
: "${CLEAR_CACHES:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${LOG_DIR:=$(pwd)/results}"
: "${NEXP:=5}"

: "${LOG_FILE_BASE:="${LOG_DIR}/${DATESTAMP}"}"
: "${CONT_NAME:=multinode_bert_2209}"

_config_env+=(SEED)
_config_env+=(PADDLE_TRAINER_ENDPOINTS)
_config_env+=(PADDLE_TRAINER_PORTS)
_config_env+=(PADDLE_TRAINERS_NUM)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

clear_cache() {
  for h in $(echo $PADDLE_TRAINER_ENDPOINTS | tr "," "\n");
  do
    ssh -t $h "sync && /sbin/sysctl vm.drop_caches=3"
  done
}

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            clear_cache
            echo 1
        fi

        # Run experiment
        docker exec  "${_config_env[@]}" "${CONT_NAME}" bash -c "bash ./run_and_time.sh"
    ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"
done
