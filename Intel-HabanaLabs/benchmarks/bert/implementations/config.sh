#! /bin/bash

DATESTAMP=`date +'%y%m%d%H%M%S'`

export OUTPUT_DIR=./phase_2
export LOG_DIR=$OUTPUT_DIR
mkdir -p $OUTPUT_DIR

export INITIAL_CHECKPOINT=/home/mge/dataset/bert_data/download/MLPerf_BERT_checkpoint/tf2_ckpt/model.ckpt-28252
export BERT_CONFIG_DIR=/home/mge/dataset/bert_data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
export DATASET_PATH=/home/mge/dataset/bert_data/tfrecord/lower_case_1_seq_len_512_max_pred_76_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/
export INPUT_FILES_DIR_UNPACKED=$DATASET_PATH/training
export INPUT_FILES_DIR_PACKED=$DATASET_PATH/training_packed
export EVAL_FILES_DIR=$DATASET_PATH/test
export TRAIN_BATCH_SIZE=14 # OOM for 28
export EVAL_BATCH_SIZE=125
export MAX_EVAL_STEPS=10
export NUM_DIST_EVAL_WORKERS=8
export TRAIN_STEPS=6700
export WARMUP_STEPS=0
export LEARNING_RATE=0.000425
export LAMB_BETA_1=0.9
export LAMB_BETA_2=0.999
export EPSILON=1e-06
export LAMB_WEIGHT_DECAY_RATE=0.01
export LAMB_LEARNING_RATE_DECAY_POLY_POWER=1.0
export NUM_ACCUMULATION_STEPS=4 # 2
export SAMPLES_START_EVAL=0
export SAVE_CHECKPOINTS_STEPS=335
export PACKED_DATA=True
export USE_HOROVOD=True
export NUM_WORKERS_TOTAL=8
export USE_LIGHTWEIGHT_CHECKPOINT=False # to WA the error "TensorFlow device (XPU:0) is being mapped to multiple devices"
export DO_TRAIN=True
export DO_EVAL=True
export USE_ASYNC_CHECKPOINTING=True
export EXPERIMENTAL_SLACK=True

PHASE1_CKPT=$INITIAL_CHECKPOINT
BERT_CONFIG_FILE=$BERT_CONFIG_DIR/bert_config.json
MAX_SEQ_LENGTH=512
MAX_PRED_PER_SEQ=76
LIGHTWEIGHT_CHECKPOINT_IMPL=basic
OPTIMIZER=lamb
SAMPLES_BETWEEN_EVAL=$(($TRAIN_BATCH_SIZE*$NUM_WORKERS_TOTAL*$NUM_ACCUMULATION_STEPS*$SAVE_CHECKPOINTS_STEPS))
STOP_THRESHOLD=0.720

enable_device_warmup=True
precision="--noamp"
export TF_DISABLE_SCOPED_ALLOCATOR=True # To fix error "ScopedAllocatorMgr not supported on device"

IS_DIST_EVAL_ENABLED="False"
if [ $USE_HOROVOD == "True" ]; then
   horovod="--horovod --allreduce_post_accumulation=True"
#    IS_DIST_EVAL_ENABLED="True"
else
   horovod=""
fi

if [ $PACKED_DATA == "False" ]; then
   export INPUT_FILES_DIR=${__input_files_dir:-$INPUT_FILES_DIR_UNPACKED}
   packing_arg=""
else
   export INPUT_FILES_DIR=${__input_files_dir:-$INPUT_FILES_DIR_PACKED}
   packing_arg="--enable_packed_data_mode  --avg_seq_per_pack=2"
fi

# if [[ -n ${TF_PROFILE_STEPS} ]]; then
# 	TF_PROFILE_STEPS="--profile_steps=${TF_PROFILE_STEPS}"
# 	PROFILE=1
# else
# 	TF_PROFILE_STEPS=''
# fi
