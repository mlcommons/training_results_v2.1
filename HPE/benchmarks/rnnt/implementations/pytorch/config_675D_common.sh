## System config params
export DGXNGPU=8
export DGXSOCKETCORES=128
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

export export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
## Data Paths
#export DATADIR="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/2048_shards_uncompressed" #"/raid/datasets/bert/hdf5/4320_shards"
#export EVALDIR="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/2048_shards_uncompressed" #"/raid/datasets/bert/hdf5/eval_4320_shard"
#export DATADIR_PHASE2="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/2048_shards_uncompressed" #"/raid/datasets/bert/hdf5/4320_shards"
#export CHECKPOINTDIR="./ci_checkpoints"
#export RESULTSDIR="./results"
#using existing checkpoint_phase1 dir
#export CHECKPOINTDIR_PHASE1="/cstor/SHARED/datasets/MLPERF/training2.0/bert/phase1" # "/raid/datasets/bert/checkpoints/checkpoint_phase1"
#export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
