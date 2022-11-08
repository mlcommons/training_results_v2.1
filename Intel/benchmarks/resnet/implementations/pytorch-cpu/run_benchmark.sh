#!/bin/bash

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
END_CORE=`expr $CORES - 1`

#Benchmark args
export BATCH_SIZE=3264
export DATAWORKERS=1
export DATAPATH=<IMAGENET PATH>
export seed_num=$(date +%s)


#OMP settings
export OMP_NUM_THREADS=$(( CORES - DATAWORKERS ))
export KMP_AFFINITY=granularity=fine,compact,1,0,verbose
export KMP_BLOCKTIME=1


#oneCCL settings
export CCL_WORKER_COUNT=8
export CCL_LOG_LEVEL=info
export CCL_BF16=avx512bf
export CCL_ATL_TRANSPORT=ofi
export CCL_MNIC_COUNT=2
export CCL_MNIC=local
export CCL_MNIC_NAME=irdma-cvl01tf2,irdma-cvl11tf2
export CCL_ALLREDUCE=ring
export CCL_WORKER_COUNT=8

for (( i = $SOCKETS; i < 2*$SOCKETS; i++ )); do  # pin CCL workers to HT
  START_CORE=$(( i * CORES ))
  for (( j = 0; j < $CCL_WORKER_COUNT; j++)); do
   CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY} $((START_CORE + j))"
  done
done

export CCL_WORKER_AFFINITY=`echo ${CCL_WORKER_AFFINITY} | tr " " ","`


#DDP settings
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export MASTER_ADDR=`head -1 hostfile`

# Fabric settings
export FI_PROVIDER=psm3
export PSM3_IDENTIFY=1
export PSM3_ALLOW_ROUTERS=1
export PSM3_RDMA=1
export PSM3_PRINT_STATS=1
export PSM3_RV_MR_CACHE_SIZE=8192
export PSM3_KASSIST_MODE=none

mpiexec.hydra -np 32 -ppn 2 -f hostfile -l -genv I_MPI_PIN_DOMAIN=socket \
	python -u train.py $DATAPATH -a resnet50 --bf16 --epochs 37 --warmup-epochs 2 \
      	--ipex -j $DATAWORKERS -b $BATCH_SIZE --seed $seed_num \
	--dist-backend=ccl --base-op=LARS --base-lr 10.5 --weight-decay 0.00005 2>&1 | tee training_rn50_bf16.log
