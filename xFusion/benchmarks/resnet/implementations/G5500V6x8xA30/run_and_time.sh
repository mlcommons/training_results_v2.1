#!/bin/bash
cd ../mxnet
source config_G5500V6x8xA30.sh
CONT=mlperf-xfusion:resnet DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR  ./run_with_docker.sh
