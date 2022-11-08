#!/bin/bash

NBNODE=${1:-1}
echo "Running BERT on ${NBNODE} nodes"

export curDir=$PWD

#export TGT_DIR=/pfss/hddfs1/bruno/MLCOMMONS/training2.1/
export TGT_DIR=/pfss/nvmefs1/bruno/MLCOMMONS/training2.1/

export CONT=/pfss/hddfs1/bruno/MLCOMMONS/containers/enroot/mlperfv21.bert.sqsh


#export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
#mkdir -p ${LOGDIR}

export SCRIPTDIR=${curDir}

# 4320 SHARDS
export DATADIR="${TGT_DIR}/bert/hdf5/training-4320/hdf5_4320_shards_varlength" #<path/to/4320_shards_varlength/dir> 
export DATADIR_PHASE2="${TGT_DIR}/bert/hdf5/training-4320/hdf5_4320_shards_varlength" #<path/to/4320_shards_varlength/dir> 
export EVALDIR="${TGT_DIR}/bert/hdf5/eval_varlength" #<path/to/eval_varlength/dir> 
export CHECKPOINTDIR_PHASE1="${TGT_DIR}/bert/phase1/" #<path/to/pytorch/ckpt/dir> 
export NEXP=10 # 0

#cd pytorch
source ${curDir}/pytorch/config_675D_enroot_x${NBNODE}.sh
export MELLANOX_VISIBLE_DEVICES=all
sbatch --comment="turbo ; sysctl file=${PWD}/systcl-bert " --export=ALL  -N ${NBNODE} ${curDir}/pytorch/run.hpe.sub
