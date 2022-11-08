# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

NGC_VER=22.09
PADDLE_COMMIT=bd10211c786671d79fd764229aa81c3c9d78ba58

SRC_IMAGE="nvcr.io/nvidia/pytorch:${NGC_VER}-py3"
DST_IMAGE="nvcr.io/nvidia/pytorch:${NGC_VER}-py3-paddle-dev-test"

PYTHON_VER="3.8"

export https_proxy=http://172.19.56.199:3128
export http_proxy=http://172.19.56.199:3128
export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com

###################

TMP_IMAGE="nvcr.io/nvidia/pytorch:${NGC_VER}-py3-paddle-dev"
TMP_DOCKERFILE=Dockerfile.tmp
PADDLE_DIR=/workspace/Paddle_src
APEX_CLONE_DIR=/workspace/apex_dir
APEX_DIR=$APEX_CLONE_DIR/apex/build_scripts
FLASH_ATTN_CLONE_DIR=/workspace/flash_attn_dir
FLASH_ATTN_DIR=$FLASH_ATTN_CLONE_DIR/training_results_v2.0/HazyResearch/benchmarks/bert/implementations/pytorch/csrc/stream_attn
JEMALLOC_CLONE_DIR=/workspace/jemalloc_dir

PY_PACKAGE_DIR=/opt/conda/lib/python3.8/site-packages
PYTHON=python$PYTHON_VER


if [[ $SRC_IMAGE == $DST_IMAGE ]]; then
  echo "Error: SRC_IMAGE and DST_IMAGE cannot be the same!!!" 
  exit 1
fi

OLD_DIR=`pwd`
NEW_DIR=$(dirname `readlink -f "$0"`)
cd $NEW_DIR

docker build -t $TMP_IMAGE \
  --build-arg FROM_IMAGE_NAME=${SRC_IMAGE} \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  --build-arg no_proxy=$no_proxy \
  -f Dockerfile .

cd $NEW_DIR/..

cat <<EOF >$TMP_DOCKERFILE
FROM $TMP_IMAGE
RUN mkdir -p $APEX_CLONE_DIR \
        && cd $APEX_CLONE_DIR \
        && git clone -b new_fmhalib https://github.com/sneaxiy/apex \
        && cd $APEX_DIR \
        && bash build.sh 
ENV APEX_DIR $APEX_DIR
RUN mkdir -p $JEMALLOC_CLONE_DIR \
        && cd $JEMALLOC_CLONE_DIR \
        && git clone -b 5.3.0 https://github.com/jemalloc/jemalloc \
        && cd jemalloc \
        && ./autogen.sh && ./configure && make -j `nproc` && make install -j `nproc`
RUN mkdir -p $FLASH_ATTN_CLONE_DIR \
        && cd $FLASH_ATTN_CLONE_DIR \
        && git clone -b wrap_stream_attn_so https://github.com/sneaxiy/training_results_v2.0 \
        && cd $FLASH_ATTN_DIR \
        && bash build.sh 
RUN mkdir -p $PADDLE_DIR \
        && cd $PADDLE_DIR \
        && git clone https://github.com/PaddlePaddle/Paddle \
        && cd $PADDLE_DIR/Paddle \
        && git checkout ${PADDLE_COMMIT} 
ENV COMPILE_DIR $PADDLE_DIR/Paddle/build
COPY Dockerfiles/cub.cmake $PADDLE_DIR/Paddle/cmake/external
RUN mkdir -p $PADDLE_DIR/Paddle/build
RUN cd $PADDLE_DIR/Paddle/build && cmake .. \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH_NAME=Ampere \
  -DWITH_AVX=ON \
  -DWITH_MKL=ON \
  -DWITH_DISTRIBUTE=ON \
  -DWITH_BRPC_RDMA=OFF \
  -DWITH_LIBXSMM=OFF \
  -DWITH_PSLIB=OFF \
  -DWITH_BOX_PS=OFF \
  -DWITH_XBYAK=ON \
  -DWITH_PSCORE=ON \
  -DWITH_HETERPS=OFF \
  -DWITH_GLOO=ON \
  -DWITH_TESTING=OFF \
  -DPY_VERSION=$PYTHON_VER
RUN cd $PADDLE_DIR/Paddle/build && make -j `nproc` 
RUN $PYTHON -m pip install -U --force-reinstall $PADDLE_DIR/Paddle/build/python/dist/*.whl 
COPY external_ops external_ops 
ENV FLASH_ATTN_DIR $FLASH_ATTN_DIR
RUN cd external_ops && $PYTHON setup.py install --force && rm -rf external_ops
COPY pybind pybind 
RUN cd pybind && $PYTHON compile.py && mkdir -p $PY_PACKAGE_DIR/pybind && cp *.so $PY_PACKAGE_DIR/pybind       
RUN $PYTHON -m pip install -U --force-reinstall git+https://github.com/mlperf/logging.git@2.0.0-rc1 
RUN $PYTHON -m pip install -U --force-reinstall protobuf==3.20.1
RUN apt-get install -y openssh-server
EOF

docker build -t $DST_IMAGE \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  --build-arg no_proxy=$no_proxy \
  -f $TMP_DOCKERFILE .

rm -rf $TMP_DOCKERFILE
cd $OLD_DIR
