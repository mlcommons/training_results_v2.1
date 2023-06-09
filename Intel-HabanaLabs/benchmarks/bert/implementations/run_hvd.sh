#! /bin/bash

source ~/hvd.sh

source ./config.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

export CCL_ZE_CACHE=0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ITEX_AUTO_MIXED_PRECISION=1
export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16"
export ITEX_LIMIT_MEMORY_SIZE_IN_MB=1024
# export ITEX_ALLOC_MODE=2

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
	_--num_train_steps=$TRAIN_STEPS \
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
	--optimizer_type=$OPTIMIZER |& tee bert_hvd_$DATESTAMP.log # ${TF_PROFILE_STEPS} ${AUX_PARAMS}
