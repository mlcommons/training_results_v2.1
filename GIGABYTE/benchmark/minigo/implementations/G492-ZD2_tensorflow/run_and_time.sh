#!/bin//bash
cd ../tensorflow
source config_G492-ZD2.sh
export CONT=nvcr.io/nvdlfwea/mlperfv21/minigo:20221004.tensorflow
export DATADIR="</dataset/datasets/minigo/checkpoint>"
export LOGDIR="</path/to/logdir>"
./run_with_docker.sh

