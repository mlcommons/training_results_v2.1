## System config params
export DGXNGPU=8
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/shared/mlperf/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
export DATADIR_PHASE2="/shared/mlperf/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="/shared/mlperf/bert_data/hdf5/eval_varlength"
export CHECKPOINTDIR="/shared/mlperf/checkpoints"
export CHECKPOINTDIR_PHASE1="/shared/mlperf/bert_data/phase1"
export UNITTESTDIR="/shared/mlperf/bert_unittest"
