#! /bin/bash

source ~/hvd.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

export CCL_ZE_CACHE=0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ITEX_AUTO_MIXED_PRECISION=1
export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16"
export ITEX_LIMIT_MEMORY_SIZE_IN_MB=4096
# export ITEX_ALLOC_MODE=2

DATESTAMP=`date +'%y%m%d%H%M%S'`
source ./config.sh

export OUTPUT_DIR=./phase_2
export LOG_DIR=$OUTPUT_DIR
mkdir -p $OUTPUT_DIR

export DATASET_PATH=/home/mge/dataset/bert_data/tfrecord/lower_case_1_seq_len_512_max_pred_76_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/
export INPUT_FILES_DIR_UNPACKED=$DATASET_PATH/training
export INPUT_FILES_DIR_PACKED=$DATASET_PATH/training_packed
export EVAL_FILES_DIR=$DATASET_PATH/test
export INITIAL_CHECKPOINT=/home/mge/dataset/bert_data/download/MLPerf_BERT_checkpoint/tf2_ckpt/model.ckpt-28252
export BERT_CONFIG_FILE=/home/mge/dataset/bert_data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json

export PACKED_DATA=True
if [ $PACKED_DATA == "False" ]; then
   export INPUT_FILES_DIR=$INPUT_FILES_DIR_UNPACKED
   packing_arg=""
else
   export INPUT_FILES_DIR=$INPUT_FILES_DIR_PACKED
   packing_arg="--enable_packed_data_mode  --avg_seq_per_pack=2"
fi

mpirun -np $NUM_WORKERS_TOTAL \
python TensorFlow/nlp/bert/run_pretraining.py \
	--input_files_dir=$INPUT_FILES_DIR \
	--init_checkpoint=$INITIAL_CHECKPOINT \
	--eval_files_dir=$EVAL_FILES_DIR\
	--output_dir=$OUTPUT_DIR \
	--bert_config_file=$BERT_CONFIG_FILE \
	--do_train=$DO_TRAIN \
	--do_eval=$DO_EVAL \
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
	--samples_start_eval=$SAMPLES_START_EVAL \
	--noenable_habana_backend \
	--dllog_path=$LOG_DIR/bert_dllog.json \
	--use_lightweight_checkpoint=$USE_LIGHTWEIGHT_CHECKPOINT \
	--lightweight_checkpoint_impl=$LIGHTWEIGHT_CHECKPOINT_IMPL \
	--num_dist_eval_workers=$NUM_DIST_EVAL_WORKERS \
	--optimizer_type=$OPTIMIZER |& tee bert_hvd_$DATESTAMP.log # ${TF_PROFILE_STEPS} ${AUX_PARAMS}
