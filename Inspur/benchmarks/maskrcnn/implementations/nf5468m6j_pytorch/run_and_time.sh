cd ../pytorch
source config_NF5468M6J.sh
DGXSYSTEM=NF5468M6J CONT=mlperf-inspur:maskrcnn DATADIR=/path/to/preprocessed/data LOGDIR=LOGDIR=/path/to/logfile ./run_with_docker.sh
