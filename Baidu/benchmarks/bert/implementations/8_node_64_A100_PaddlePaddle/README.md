# Download and prepare the data

Please download and prepare the data as described [here](https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/README.md#download-and-prepare-the-data).

After preparation, you can see the directories are like:

```
<BASE_DATA_DIR>
                     |_ phase1                                   # checkpoint to start from tf1
                     |_ hdf5  
                           |_ eval                               # evaluation chunks in binary hdf5 format fixed length 
                           |_ eval_varlength                     # evaluation chunks in binary hdf5 format variable length
                           |_ training_4320                      # 
                              |_ hdf5_4320_shards_uncompressed   # sharded data in hdf5 format fixed length
                              |_ hdf5_4320_shards_varlength      # sharded data in hdf5 format variable length
```

# Build the docker image

You can run the following command to build the docker image. 

```
bash Dockerfiles/build_image_from_scratch.sh
```

This command would take a long time. It contains the following steps:

- Build the docker image which can compile the PaddlePaddle source code.
- Compile the PaddlePaddle source code.
- Compile the PaddlePaddle external operators.
- Compile the PaddlePaddle external pybind functions.

After the command finishes, you will get the docker image named `nvcr.io/nvidia/pytorch:22.09-py3-paddle-dev-test`.

# Prepare the checkpoint file

Originally, the checkpoint of the BERT model is generated from TensorFlow. We can convert the original TensorFlow checkpoint file to the Python dictionary like this and dump the dictionary using the Python pickle module.

```python
{
  "bert/encoder/layer_0/attention/self/query/kernel": numpy.ndarray(...),
  "bert/encoder/layer_0/attention/self/query/bias": numpy.ndarray(...),
  ...
}
```

In this way, we can run tests without installing TensorFlow again after conversion. You can use the following command to convert the original TensorFlow checkpoint file:

```python
python models/convert_tf_checkpoint.py \
    <BASE_DATA_DIR>/phase1/model.ckpt-28252 \
    <BASE_DATA_DIR>/phase1/model.ckpt-28252.tf_pickled
```

# Running the model using multiple nodes

1. Start containers

Start containers at all the nodes with the following command.

```
export BASE_DATA_DIR=<your_bert_data_dir>
export CONT_NAME=<your_container_name>
export CONT=nvcr.io/nvidia/pytorch:22.09-py3-paddle-dev-test # the docker image name
bash start_docker.sh
```

2. SSH/MPI configuration

* MPI/SSH should be configured in all the containers in order to use `mpirun` to launch job from one container.
* In order to clear cache for all the nodes from one node, we recommend to configure SSH authentication without password on the nodes, see `run_multi_node_with_docker.sh` for more.

3. Run benchmark

Run the following script in the first node, the training job will be launch in all the nodes by `mpirun`.

```
export PADDLE_TRAINER_ENDPOINTS=<node_ip_list> # all the ips of all nodes, separated by comma
export PADDLE_TRAINER_PORTS=60000,60001,60002,60003,60004,60005,60006,60007 # the ports used by PaddlePaddle, one port per gpu
export PADDLE_TRAINERS_NUM=8 # the number of nodes
export CONT_NAME=<your_container_name>
export NEXP=10 # the trial test number  

bash run_multi_node_with_docker.sh
```
