#!/bin/bash
cd ../hugectr
source config_G5500V6x8xA30.sh
export CONT=mlperf-xfusion:dlrm
export DATADIR=/path/to/preprocessed/data
export LOGDIR=/path/to/logfile
./run_with_docker.sh