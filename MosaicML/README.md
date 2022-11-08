# MLPerf Training v2.1 MosaicML Submission

Repository for [MosaicML](http://www.mosaicml.com)'s submission to the MLperf Training v2.1 Open Division benchmark. This submission has several goals:
* Submit a PyTorch-based bert-large using an easy-to-use trainer, our open-source [Composer](https://github.com/mosaicml/composer) library.
* Highlight the gains from our recipes of methods that speed up the training of deep learning models by system-level optimizations and changes to the training algorithm.

# Submitted Benchmarks

In this round, we submit [bert](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert) model
trained on the provided [wiki dataset](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets) benchmark using [PyTorch](http://pytorch.org) with two configurations:
* **baseline**: A bert-large baseline that uses off-the-shelf [HuggingFace's bert](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert) model.
* **methods**: We modify the training with several speed up metohds from our [Composer](https://github.com/mosaicml/composer) libary and our private algorithms respository.

**Note** Please note that we use the same optimizers and hyperparameters between the `baseline` and `methods` runs.

Each configuration is trained using our open source library [Composer](https://github.com/mosaicml/composer) on 8x NVIDIA A100-80GB GPUs on a single server. The library includes an `MLPerfCallback` that generates the results logs; this will make it easier for the broader community to submit to the MLPerf Open Division in the future.

In the previous submission round, we submitted
[ResNet-50](https://github.com/mlcommons/training/tree/master/image_classification) and the results for the same are
available [here](https://github.com/mlcommons/training_results_v2.0/tree/main/MosaicML). Please checkout [our blog on ResNet-50](https://www.mosaicml.com/blog/mlperf-2022) results for more details.

# Configuration details

Each configuration is defined with a YAML file, linked in the below table.

| Benchmark | Benchmark Config | Description | Speedup Methods |
| --- | --- | --- | --- |
| bert | [8xA100_80GB-baseline](benchmarks/bert/implementations/8xA100_80GB-baseline/config.yaml) | Base training recipe | 20% masking |
| bert | [8xA100_80GB-methods](benchmarks/bert/implementations/8xA100_80GB-methods/config.yaml) | Optimized training recipe | 20% masking, Unpadded model, Flash Attention, Fusions, Fused LayerNorm |

For details on our speed up methods, please see our [Methods Overview](https://docs.mosaicml.com/en/stable/method_cards/methods_overview.html) documentation.

# System Overview 

These benchmarks have been tested with the following machine configuration:

* 2x AMD EPYC 7513 32-Core Processor
* 8x NVIDIA A100-SXM4-80GB GPUs

# Reproducing Results

To exactly reproduce our results, following the instructions in the [implementations](benchmarks/bert/implementations/composer) folder to setup and run the `run_and_time.sh` script. We use YAML along with a config manager [yahp](https://github.com/mosaicml/yahp).

## Using algorithms in your own code

One of our goals is enabling the community to access these optimizations outside of MLPerf benchmarks, and to easily submit your own configs to MLPerf. While this submission code uses YAML and scripting, these speed-up methods can also be applied directly to your own python code. For example, this submission's speed-up algorithms can be applied with:

```python
from composer import Trainer, algorithms
from composer.callbacks import MLPerfCallback

trainer = Trainer(
    model=your__model,
    train_dataloader=your_dataloader,
    optimizers=your_optimizer,
    schedulers=your_scheduler,
    
    # speed-up algorithms below
    algorithms=[
        algorithms.FusedLayerNorm(),
    ],
    
    # optional: MLperf logging
    callbacks=[
        MLPerfCallback('/results/', index=0)
    ],
    ...
)

trainer.fit()

```

For more details, see the Composer [documentation](https://docs.mosaicml.com/en/stable/) and the [MLPerfCallback](https://docs.mosaicml.com/en/stable/api_reference/composer.callbacks.mlperf.html#composer.callbacks.mlperf.MLPerfCallback)

## Software Packages

* [Composer](https://github.com/mosaicml/composer)
* MosaicML's [PyTorch](https://hub.docker.com/r/mosaicml/pytorch/tags) Docker Image
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

# Repository Contents

The repository is submitted with the following:

* `benchmarks`: Configuration files and Composer entrypoint code for running submitted benchmarks
* `results`: Run results for each benchmark
* `systems`: System configuration for each benchmark
