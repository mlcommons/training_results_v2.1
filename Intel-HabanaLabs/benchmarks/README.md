# Habana MLPerf training 2.1 submisssion

## Install firmware, driver, SynapseAI 1.7.98

Follow the steps in [Setup and Install](https://docs.habana.ai/en/v1.6.0/Installation_Guide/GAUDI_Installation_Guide.html) to setup each compute node in the cluster.


## Build and deploy HabanaLabs MLPERF training 2.1 container in the cluster

For each compute node, do
```
git clone HabanaLabs MLPERF training 2.1 code from public repo at https://github.com/mlcommons/
```

Pull HabanaLabs gaudi-docker-mlperf/ver2.1 release container from vault.

For Tensorflow:
```
docker pull vault.habana.ai/gaudi-docker-mlperf/ver2.1/tensorflow-installer-tf-cpu-2.8.3:1.7.98-27
```

For PyTorch:
```
docker pull vault.habana.ai/gaudi-docker-mlperf/ver2.1/pytorch-installer-1.12.0:1.7.98-27
```

Build MLPERF training 2.1 container by

1. Copying MLPERF training 2.1 code to /root/MLPERF
2. Copying ssh keys to enable passwordless ssh to /root/.ssh/
3. Creating hostfile that contains a list of hosts in the cluster. Store it in /root/share in the docker

    (e.g., for single node: ```echo your-machine-ip > /root/shared/hosts```)
4. Installing numactl package (required for large scale Gaudi)
5. Naming the container mlperf2.1_img

For Gaudi2 start MLPERF training 2.1 container by executing

```
docker run --privileged --security-opt seccomp=unconfined \
           --name mlperf2.1 -td                    \
           -v /dev:/dev                            \
           --device=/dev:/dev                      \
           -e LOG_LEVEL_ALL=6                      \
           -v /sys/kernel/debug:/sys/kernel/debug  \
           -v /tmp:/tmp                            \
           -v $LOG_DIR:/root/scratch               \
           -v $DATASET_DIR:/root/datasets/         \
           --cap-add=sys_nice --cap-add=SYS_PTRACE \
           --user root --workdir=/root --net=host  \
           --ulimit memlock=-1:-1 mlperf2.1_img

docker exec mlperf2.1 bash -c "service ssh start"
```

# Resnet50
## Prepare Imagenet dataset

 1. Sign up with [image-net.org](http://image-net.org/download-images) and acquire the rights to download original images
 2. Follow the link to the 2012 ILSVRC and download ILSVRC2012_img_val.tar and ILSVRC2012_img_train.tar
 3. Use the script below to unpact the dataset. Set IMAGENET_HOME to a folder where dataset should be placed

```
export IMAGENET_HOME=/path/to/imagenet
mkdir -p $IMAGENET_HOME/val
mkdir -p $IMAGENET_HOME/train
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_HOME/val
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_HOME/train
cd $IMAGENET_HOME/train
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
done
rm $IMAGENET_HOME/train/*.tar
cd $IMAGENET_HOME/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

## Run and time TensorFlow Resnet50
Inside docker install additional packages required for Resnet50:
```
export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
pip install -r $RESNET_IMPLEMENTATIONS/TensorFlow/computer_vision/Resnets/resnet_keras/requirements.txt
```
Execute the script
```
cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-TF
./launch_keras_resnet_hvd.sh --config batch_256.cfg --cpu-pin cpu --jpeg-data-dir /path/to/imagenet --log_dir /root/scratch
```
for a cluster run based on hostfile.
Use the ```$IMAGENET_HOME``` folder from [prepare imagenet section](#prepare-imagenet-dataset) for ```--jpeg-data-dir```.
Results of the run will be placed on the host, in folder specified by ```--log_dir``` parameter.

## Run and time PyTorch Resnet50
Inside docker install additional packages required for Resnet50:
```
export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
pip install -r $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-PT/PyTorch/requirements.txt
```
Execute the script
```
cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-PT
./launch_resnet.sh --config batch_256.cfg --data-dir /path/to/imagenet --log-dir /root/scratch
```
for a cluster run based on hostfile.
Use the ```$IMAGENET_HOME``` folder from [prepare imagenet section](#prepare-imagenet-dataset) for ```--data-dir```.
Results of the run will be placed on the host, in folder specified by ```--log-dir``` parameter.

# Bert TF

## Prepare packed wiki dataset

**Location to download Dataset and Checkpoint:** [Dataset and Checkpoint download location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT)

**Dataset Preparation:** In order to use dataset one needs to preprocess it similarly as described in [Bert dataset preparation](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets).

Each of the 500 dataset files can be converted in the following way:
```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/TensorFlow/nlp/bert
pip3 install -r requirements.txt
python3 pretraining/create_pretraining_data.py \
    --input_file=<path to downloaded and unzipped dataset>/part-00XXX-of-00500 \
    --output_file=<output dir for tfrecord files>/part-00XXX-of-00500 \
    --vocab_file=<path to downloaded vocab.txt> \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=10
```


After tfrecord files are ready we pack them using similar code as suggested by [GraphCore for v1.0 submission](https://github.com/mlcommons/training_results_v1.0/tree/master/Graphcore/benchmarks/bert/implementations/popart/bert_data)

```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/TensorFlow/nlp/bert
pip3 install -r requirements.txt
python3 pack_pretraining_data_tfrec.py \
    --input-dir /path-to-tfrecords-dir \
    --output-dir /path-to-tfrecords-packed-dir \
    --max-files 500
```

For additional details please refer to [Packing: Towards 2x NLP BERT Acceleration](https://arxiv.org/abs/2107.02027).

## Run and time

Log into one of the mlperf2.1 containers

Given a runtime configuration, for instance, 8 Gaudi2 run
```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/HLS-Gaudi2-TF
```
Edit defaults.cfg with the right location of your packed dataset tf records inside the container

for example, ```INPUT_FILES_DIR_PACKED=/root/datasets/bert_pretraining/packed```
execute the script ```launch_bert_hvd.sh --config defaults.cfg``` for a cluster run based on hostfile
It will place the results of the run at $LOG_DIR on the host.

# Bert PT

## Dataset Preparation

**Location to download Dataset and Checkpoint:** [Dataset and Checkpoint download location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT)

One should use the steps described in https://github.com/mlcommons/training_results_v2.0/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch#download-and-prepare-the-data to download and preprocess data. At this stage checkpoint ```/workspace/bert_data/phase1``` and evaluation data  ```/workspace/bert_data/hdf5/eval_varlength``` are ready. Training data ```/workspace/bert_data/hdf5/training_4320/hdf5_4320_shards_uncompressed``` need to undergo packing process described below.

## Training data packing

After train data are ready, we pack them using a similar code as suggested by [GraphCore for v1.0 submission](https://github.com/mlcommons/training_results_v1.0/tree/master/Graphcore/benchmarks/bert/implementations/popart/bert_data)

```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/PyTorch
pip3 install -r requirements.txt
python3 pack_pretraining_data_pytorch.py \
    --input_dir=/workspace/bert_data/hdf5/training_4320/hdf5_4320_shards_uncompressed \
    --output_dir=/workspace/bert_data/packed \
    --max_predictions_per_seq=76
```

Please refer to [Packing: Towards 2x NLP BERT Acceleration](https://arxiv.org/abs/2107.02027) for additional details.
## Run and time

Log into one of the mlperf2.1 containers

Given a runtime configuration, for instance, 8 Gaudi2 run
```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/HLS-Gaudi2-PT
```
Run the following code, adjusting parameters to locations of processed datasets and checkpoints.

```
bash launch_bert_pytorch.sh --input_dir /workspace/bert_data/packed
--phase1-ckpt /workspace/bert_data/phase1/model.ckpt-28252.pt --eval-dir /workspace/bert_data/hdf5/eval_varlength
```
, where:

```/workspace/bert_data/phase1/model.ckpt-28252.pt``` - checkpoint from phase1 prepared by prepare_data.sh as described in [Dataset preparation](#dataset-preparation)

```/workspace/bert_data/hdf5/eval_varlength``` - evaluation dataset prepared by prepare_data.sh as described in [Dataset preparation](#dataset-preparation)

```/workspace/bert_data/packed``` - training dataset generated as described in [Training dataset packing](#training-data-packing)

By default, results of the training will be placed under ```/tmp/BERT_PRETRAINING```