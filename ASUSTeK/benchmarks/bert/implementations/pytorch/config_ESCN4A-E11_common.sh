## System config params
export DGXNGPU=4
export DGXSOCKETCORES=64
export DGXNSOCKET=1
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
export CUDA_VISIBLE_DEVICES="0,1,2,3"
