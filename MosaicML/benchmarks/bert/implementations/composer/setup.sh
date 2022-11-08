# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Print commands, stop on error
set -x -e

# Setup required environment variables

## System config
export SYS_ROOT_NAME="8xA100_80GB"
export SYS_GPU_NUM=8
export SYS_HOST_PROCESSORS_PER_NODE=2
export SYS_CACHE_CLEAR_CMD="/sbin/sysctl -w vm.drop_caches=3"

## Benchmark config
export BENCHMARK_NAME="bert"
export BENCHMARK_TOTAL_RUNS=10

# Install dependencies

## Run pip installer using the local requirements file
pip install -r ./requirements.txt

# Disable print command, stop on error
set +x +e
