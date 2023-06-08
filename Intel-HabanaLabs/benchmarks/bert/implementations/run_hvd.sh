#! /bin/bash

source ~/hvd.sh
export PYTHONPATH=/home/$user/workspace/training_results_v2.1/Intel-HabanaLabs/benchmarks/bert/implementations:$PYTHONPATH

export DATASET_PATH=/home/mge/dataset/bert_data/tfrecord/lower_case_1_seq_len_512_max_pred_76_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/

# from defaults.cfg
DATESTAMP=`date +'%y%m%d%H%M%S'`
export INPUT_FILES_DIR_UNPACKED=$DATASET_PATH/training
export INPUT_FILES_DIR_PACKED=$DATASET_PATH/training_packed
export EVAL_FILES_DIR=$DATASET_PATH/test
export OUTPUT_DIR=./phase_2
export LOG_DIR=./phase_2
export INITIAL_CHECKPOINT=/home/mge/dataset/bert_data/download/MLPerf_BERT_checkpoint/tf2_ckpt/model.ckpt-28252
export BERT_CONFIG_DIR=/home/mge/dataset/bert_data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
export TRAIN_BATCH_SIZE=14 # OOM for 28
export EVAL_BATCH_SIZE=125
export MAX_EVAL_STEPS=10
#export NUM_DIST_EVAL_WORKERS=8
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
#export HLS_TYPE="HLS1"
export NUM_WORKERS_TOTAL=8
export NUM_DIST_EVAL_WORKERS=$NUM_WORKERS_TOTAL # 8
#export RUN_TPC_FUSER=True
#export MPI_TCP_INCLUDE=enp24s0f0
#export TF_CPU_RUNTIME_FALLBACK=forbid
#export TF_HCCL_MEMORY_ALLOWANCE_MB=1536
#export HABANA_INITIAL_WORKSPACE_SIZE_MB=4600
#export CPU_BIND_TYPE=cpu
export USE_LIGHTWEIGHT_CHECKPOINT=False # to WA the error "TensorFlow device (XPU:0) is being mapped to multiple devices"
export DO_TRAIN=True
export DO_EVAL=True
export USE_ASYNC_CHECKPOINTING=True
export EXPERIMENTAL_SLACK=True

mkdir -p $OUTPUT_DIR

PHASE1_CKPT=$INITIAL_CHECKPOINT
BERT_CONFIG_FILE=$BERT_CONFIG_DIR/bert_config.json
MAX_SEQ_LENGTH=512
MAX_PRED_PER_SEQ=76
LIGHTWEIGHT_CHECKPOINT_IMPL=basic
OPTIMIZER=lamb
SAMPLES_BETWEEN_EVAL=$(($TRAIN_BATCH_SIZE*$NUM_WORKERS_TOTAL*$NUM_ACCUMULATION_STEPS*$SAVE_CHECKPOINTS_STEPS))
STOP_THRESHOLD=0.720
export TF_DISABLE_SCOPED_ALLOCATOR=True # To fix error "ScopedAllocatorMgr not supported on device"
enable_device_warmup=True
precision="--noamp"
export ITEX_AUTO_MIXED_PRECISION=1
export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ITEX_LIMIT_MEMORY_SIZE_IN_MB=4096
# export ITEX_ALLOC_MODE=2

if [ $USE_HOROVOD == "True" ]; then
   horovod="--horovod --allreduce_post_accumulation=True"
   IS_DIST_EVAL_ENABLED="True"
else
   horovod=""
   IS_DIST_EVAL_ENABLED="False"
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

mpirun -np $NUM_WORKERS_TOTAL \
python TensorFlow/nlp/bert/run_pretraining.py \
	--input_files_dir=$INPUT_FILES_DIR \
	--init_checkpoint=$PHASE1_CKPT \
	--eval_files_dir=$EVAL_FILES_DIR\
	--output_dir=$OUTPUT_DIR \
	--bert_config_file=$BERT_CONFIG_FILE \
	--do_train=$DO_TRAIN \
	--do_eval=$DO_EVAL \
	--experimental_slack=$EXPERIMENTAL_SLACK \
	--is_dist_eval_enabled=$IS_DIST_EVAL_ENABLED \
	--train_batch_size=$TRAIN_BATCH_SIZE \
	--eval_batch_size=$EVAL_BATCH_SIZE \
	--max_eval_steps=$MAX_EVAL_STEPS \
	--max_seq_length=$MAX_SEQ_LENGTH \
	--max_predictions_per_seq=$MAX_PRED_PER_SEQ \
	--num_train_steps=$TRAIN_STEPS \
	--num_accumulation_steps=$NUM_ACCUMULATION_STEPS \
	--num_warmup_steps=$WARMUP_STEPS \
	--save_checkpoints_steps=$SAVE_CHECKPOINTS_STEPS \
	--learning_rate=$LEARNING_RATE \
	$horovod \
	$precision \
	$packing_arg \
	--nouse_xla \
	--enable_device_warmup=$enable_device_warmup \
	--samples_between_eval=$SAMPLES_BETWEEN_EVAL \
	--stop_threshold=$STOP_THRESHOLD \
	--samples_start_eval=$SAMPLES_START_EVAL \
	--beta_1=$LAMB_BETA_1 \
	--beta_2=$LAMB_BETA_2 \
	--epsilon=$EPSILON \
	--weight_decay_rate=$LAMB_WEIGHT_DECAY_RATE \
	--power=$LAMB_LEARNING_RATE_DECAY_POLY_POWER \
	--noenable_habana_backend \
	--dllog_path=$LOG_DIR/bert_dllog.json \
	--use_lightweight_checkpoint=$USE_LIGHTWEIGHT_CHECKPOINT \
	--lightweight_checkpoint_impl=$LIGHTWEIGHT_CHECKPOINT_IMPL \
	--use_async_checkpointing=$USE_ASYNC_CHECKPOINTING \
	--num_dist_eval_workers=$NUM_DIST_EVAL_WORKERS \
	--optimizer_type=$OPTIMIZER # ${TF_PROFILE_STEPS} ${AUX_PARAMS}
