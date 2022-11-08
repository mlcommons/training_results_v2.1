# Benchmark

Train [bert large](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert) using [wiki dataset](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets).

# Software Requirements

* [MosaicML's Composer Library mlperf-v2.1](https://github.com/mosaicml/composer/tree/mlperf-v2.1)
* [MosaicML's PyTorch Docker Image](https://hub.docker.com/r/mosaicml/pytorch/tags)
   * Tag: `1.12.1_cu116-python3.9-ubuntu20.04`
   * PyTorch Version: 1.12.1
   * CUDA Version: 11.6
   * Python Version: 3.9
   * Ubuntu Version: 20.04
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

# Running Benchmark Configs

1. Launch a Docker container using the `pytorch` Docker Image on your training system.
   
   ```
   docker pull mosaicml/pytorch:1.12.1_cu116-python3.9-ubuntu20.04
   docker run -it mosaicml/pytorch:1.12.1_cu116-python3.9-ubuntu20.04
   ``` 
   **Note:** The `mosaicml/pytorch` Docker image can also be used with your container orchestration framework of choice.

1. Checkpoint creation: All submitted results (`baseline` and `methods`) start from the MLPerf provided checkpoint. We
   use NVidia's [convert_tf_checkpoint.py](https://github.com/mlcommons/training_results_v2.0/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch) script to 
   convert the TensorFlow checkpoint to PyTorch checkpoint and then use this PyTorch checkpoint to create a 
   Composer checkpoint. See the [scripts](scripts) folder for converting a PyTorch checkpoint to Composer Checkpoint.

1. Pre-process the wiki dataset using [the provided scripts](https://github.com/mosaicml/streaming/tree/main/streaming/text/convert/enwiki/mds) in our streaming format.

1. Pick which config you would like to run, te following configs are currently supported:
   
   | `$BENCHMARK_CONFIG` | Relative path to config file |
   | --- | --- |
   | [baseline](../8xA100_80GB-baseline/config.yaml) | `../8xA100_80GB-baseline/config.yaml` |
   | [methods](../8xA100_80GB-methods/config.yaml) | `../8xA100_80GB-methods/config.yaml` |

   **Note**: We use a few extra algorithms not specified in  `../8xA100_80GB-methods/config.yaml`. These algorithms aren't
   publicly available so the results you will see with config will be ~15% slower than our submitted results.


1. Run the benchmark, you will need to specify a supported `$BENCHMARK_CONFIG` value from the previous step and the path to the dataset. 
   The dataset can be in local dir or in a remote s3 bucket. Specify the path of the Composer checkpoint using `load_path`.

   ```
   # for dataset available locally in a folder
   bash ./run_and_time.sh --config $BENCHMARK_CONFIG --train_dataset.streaming_enwiki.local=<path_to_dataset_dir> --eval_dataset.streaming_enwiki.local=<path_to_dataset_dir> --load_object_store=None --load_path=/tmp/bert_large_<>.pt
   ```

   ```
   # for dataset stored in an object storage
   bash ./run_and_time.sh --config $BENCHMARK_CONFIG --train_dataset.streaming_enwiki.remote=<s3://path_to_dataset_dir> --eval_dataset.streaming_enwiki.remote=<s3://path_to_dataset_dir> --load_object_store=None --load_path=/tmp/bert_large_<>.pt
   ```
   
   **Note:** The `run_and_time.sh` script sources the `setup.sh` script to setup the environment.
