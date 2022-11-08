#!/bin/bash

NUMA_ARGS=""
if [ "x$MPI_LOCALRANKID" != "x" ] ; then
  REAL_NUM_NUMA_NODES=`lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
  PPNUMA=$(( MPI_LOCALNRANKS / REAL_NUM_NUMA_NODES ))
  if [ $PPNUMA -eq 0 ] ; then 
    if [ "x$SINGLE_SOCKET_ONLY" == "x1" ] ; then 
      if command -v numactl >& /dev/null ; then
        NUMA_ARGS="numactl -m 0 "
      fi
    fi
  else
    NUMARANK=$(( MPI_LOCALRANKID / PPNUMA ))
    NUMA_ARGS="$NUMA_ARGS $GDB_ARGS "
  fi
  NUM_RANKS=$PMI_SIZE
  echo "setting NUM_RANKS=$NUM_RANKS"
else
  if command -v numactl >& /dev/null ; then
    NUMA_ARGS="numactl -m 0 "
  fi
  NUM_RANKS=1
fi

echo "NUM_RANKS=$NUM_RANKS"

if [ "x$1" == "x-gdb" ] ; then
GDB_ARGS="gdb --args "
shift
else
GDB_ARGS=""
fi


# set dataset and model_path
if test -z $dataset || ! test -d $dataset ; then
  if test -d ./mlperf_dataset ; then
    dataset=./mlperf_dataset
  elif test -d ./mlperf_dataset_alt ; then
    dataset=./mlperf_dataset_alt
  else
    echo "Unable to find dataset path"
    exit 1
  fi
fi

if test -z $model_path || ! test -d $model_path ; then
  if test -d ./ref_checkpoint ; then
    model_path=./ref_checkpoint
  elif test -d ./ref_checkpoint_alt ; then
    model_path=./ref_checkpoint_alt
  else
    echo "Unable to find model path"
    exit 1
  fi
fi

GBS=2048
LBS=$(( GBS / NUM_RANKS ))

# --learning_rate=2.4e-3 --warmup_steps=0.0 --start_warmup_step=0
params="--train_batch_size=${LBS}    --learning_rate=2.0e-3     --opt_lamb_beta_1=0.66     --opt_lamb_beta_2=0.998     --warmup_proportion=0.0     --warmup_steps=0.0     --start_warmup_step=0     --max_steps=1200    --phase2    --max_predictions_per_seq=76      --do_train     --skip_checkpoint     --train_mlm_accuracy_window_size=0     --target_mlm_accuracy=0.720     --weight_decay_rate=0.01     --max_samples_termination=4500000     --eval_iter_start_samples=175000 --eval_iter_samples=175000     --eval_batch_size=16  --gradient_accumulation_steps=1     --log_freq=0 "

$NUMA_ARGS $GDB_ARGS python -u run_pretrain_mlperf.py \
    --input_dir ${dataset}/training-4320/hdf5_4320_shards_varlength/ \
    --eval_dir ${dataset}/eval_varlength/ \
    --model_type 'bert' \
    --model_name_or_path $model_path \
    --output_dir model_save \
    --fused_param_norm \
    $params \
    $@
