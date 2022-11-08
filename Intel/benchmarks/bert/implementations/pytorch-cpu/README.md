# Download and prepare data

## Dataset

Please download and prepare data as https://github.com/mlcommons/training_results_v2.0/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch#download-and-prepare-the-data. Don't use packed_data.

## Checkpoint
```
$python convert_checkpoint_tf2torch.py \
        --tf_checkpoint <path to model.ckpt-28252> \
        --bert_config_path <path to bert_config.json> \
        --output_checkpoint <path to pretraining pytorch_model.bin>
```

# Running the model

## Prepare enviroment

```
# Create new conda env 
# It creates an env.sh script for activating conda env
$bash setup_conda.sh [-p <conda_install_path>]
```

Download tpp-pytorch-extension to \<path-to-tpp\>.  

Install the tpp-pytorch-extension
```
$pushd <path-to-tpp>
$git submodule update --init
$python setup.py install
$popd
#install torch_ccl
$bash install_torch_ccl.sh
```

## Run pretraining

Install task specific requirements (one time):  
`$pip install -r requirements.txt`   

Create a link to path to MLPerf BERT pretraining dataset:  
`$ln -s <path to dataset> ./mlperf_dataset`  

Create a link to path to MLPerf BERT pretraining checkpoint:  
`$ln -s <path to checkpoint> ./ref_checkpoint` 

To run bert pretraining on multi-node run:  

```
#1) make sure system is under performance mode
$echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
$sudo echo 0 > /proc/sys/kernel/numa_balancing
$sudo cpupower frequency-set -g performance

#2) clean cache
$echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
$echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
$echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
$echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
$echo 1 > /proc/sys/vm/compact_memory; sleep 1
$echo 3 > /proc/sys/vm/drop_caches; sleep 1

#3) run the bash file 

#for closed division
#8 nodes
$bash run_8node.sh
#16 nodes
$bash run_16node.sh


#for open division
#8 nodes
$bash run_8node_open.sh
#16 nodes
$bash run_16node_open.sh
```
 

