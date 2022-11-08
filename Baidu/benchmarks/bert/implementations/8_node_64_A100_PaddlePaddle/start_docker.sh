set -ex

: "${CONT:?CONT not set}"
: "${BASE_DATA_DIR:?BASE_DATA_DIR not set}"

BASE_DATA_DIR=${BASE_DATA_DIR:-"/home/users/mlperf-workspace/bert_data"}
NVIDIA_SMI=`which nvidia-smi`

IB_DEVICES=$(find /dev/infiniband/* -maxdepth 1 -not -type d | xargs -I{} echo '--device {}:{}')

nvidia-docker run \
    ${IB_DEVICES} \
    -d -t \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --ulimit=nofile=1000000 \
    --name=$CONT_NAME \
    -e BASE_DATA_DIR=$BASE_DATA_DIR \
    -v $NVIDIA_SMI:$NVIDIA_SMI \
    -v $BASE_DATA_DIR:$BASE_DATA_DIR \
    -v $PWD:/workspace \
    -w /workspace \
    $CONT bash

docker exec $CONT_NAME ldconfig
docker start $CONT_NAME
