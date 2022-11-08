#!/bin/bash
#please set your bert_env name first
source activate pt1120
torch_ccl_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))" 2> /dev/null)
if test -f $torch_ccl_path/env/setvars.sh ; then
	  source $torch_ccl_path/env/setvars.sh
fi

# env parameters
export KMP_AFFINITY=compact,1,granularity=fine
export KMP_BLOCKTIME=1
export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so

export CCL_MNIC_NAME=irdma-cvl01tf2,irdma-cvl11tf2
export CCL_MNIC=local
export CCL_MNIC_COUNT=2
export PSM3_NIC='+(irdma-cvl01tf2|irdma-cvl11tf2)'

export FI_PROVIDER=psm3
export CCL_ALLREDUCE=ring
export PSM3_IDENTIFY=1
export PSM3_ALLOW_ROUTERS=1
export PSM3_RDMA=1
export PSM3_RV_MR_CACHE_SIZE=8192
export FI_PROVIDER_PATH=/usr/lib64/libfabric
export I_MPI_OFI_PROVIDER=psm3
export CCL_ATL_TRANSPORT=mpi
export CCL_WORKER_COUNT=8
export I_MPI_DEBUG=5



#run
# CLEAR YOUR CACHE FIRST
python -c "
from mlperf_logging.mllog import constants
from mlperf_logger import log_event
log_event(key=constants.CACHE_CLEAR, value=True)"

export SEED=$(date +%s)
./run_dist_ht.sh -np 64 -ppn 4 -f hostfile bash cmd_gbs3072.sh --use_pcl --pcl_bf16 --unpad --dense_seq_out  --dist_lamb  --no_ddp --seed ${SEED} >& output.log
