#!/bin/bash

cd ../pytorch
source config_NF5468M6J.sh
DGXSYSTEM="NF5468M6J" CONT=mlperf-inspur:ssd DATADIR=/path/to/preprocessed/data LOGDIR=/path/to/logfile ./run_with_docker.sh
