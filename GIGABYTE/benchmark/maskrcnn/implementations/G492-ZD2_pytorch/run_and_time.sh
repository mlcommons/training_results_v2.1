#!/bin/bash
cd ../pytorch
source config_G492-ZD2.sh
export CONT=mlperfv2.1-gigabyte:maskrcnn-20221004
export DATADIR=/path/to/preprocessed/data
export LOGDIR=/path/to/logfile
export PKLDIR=/path/to/folder
./run_with_docker.sh

