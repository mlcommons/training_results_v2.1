#!/bin/bash
set -x 



source config_1xXE8545x4A100-SXM4-40GB.sh


#CONT=740459dcf5ca PKLDIR=/mnt/data/coco2017  DATADIR=/mnt/data/ LOGDIR=`pwd`/maskrcnnresults ./run_with_docker.sh
CONT=038df5881ad9 PKLDIR=/mnt/data/coco2017  DATADIR=/mnt/data/ LOGDIR=`pwd`/maskrcnnresults ./run_with_docker.sh

