#! /bin/bash

export DO_TRAIN=True
export DO_EVAL=True
export EXPERIMENTAL_SLACK=True
export IS_DIST_EVAL_ENABLED=False
export TRAIN_BATCH_SIZE=14 # OOM for 28
export EVAL_BATCH_SIZE=125
export MAX_EVAL_STEPS=10
export MAX_SEQ_LENGTH=512
export MAX_PRED_PER_SEQ=76
export TRAIN_STEPS=6700
export NUM_ACCUMULATION_STEPS=4 # 2
export WARMUP_STEPS=0
export SAVE_CHECKPOINTS_STEPS=335
export LEARNING_RATE=0.000425
export NUM_WORKERS_TOTAL=8
export SAMPLES_BETWEEN_EVAL=$(($TRAIN_BATCH_SIZE*$NUM_WORKERS_TOTAL*$NUM_ACCUMULATION_STEPS*$SAVE_CHECKPOINTS_STEPS))
export STOP_THRESHOLD=0.720
export SAMPLES_START_EVAL=0
export LAMB_BETA_1=0.9
export LAMB_BETA_2=0.999
export EPSILON=1e-06
export LAMB_WEIGHT_DECAY_RATE=0.01
export LAMB_LEARNING_RATE_DECAY_POLY_POWER=1.0
export USE_LIGHTWEIGHT_CHECKPOINT=False # to WA the error "TensorFlow device (XPU:0) is being mapped to multiple devices"
export LIGHTWEIGHT_CHECKPOINT_IMPL=basic
export USE_ASYNC_CHECKPOINTING=True
export NUM_DIST_EVAL_WORKERS=8
export OPTIMIZER=lamb

export USE_HOROVOD=True
if [ $USE_HOROVOD == "True" ]; then
   horovod="--horovod --allreduce_post_accumulation=True"
#   export IS_DIST_EVAL_ENABLED=True # to WA error `RESOURCE_EXHAUSTED`?
else
   horovod=""
fi

export precision="--noamp"
export enable_device_warmup=True
export TF_DISABLE_SCOPED_ALLOCATOR=True # To fix error "ScopedAllocatorMgr not supported on device"