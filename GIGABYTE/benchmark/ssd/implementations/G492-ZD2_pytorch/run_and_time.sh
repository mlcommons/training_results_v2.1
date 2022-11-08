#!/bin/bash
cd ../pytorch
source config_G492-ZD2_001x08x032.sh
export CONT=mlperfv2.1-gigabyte:ssd-20221012
export DATADIR=/path/to/dataset
export LOGDIR=/path/to/logdir
./run_with_docker.sh

