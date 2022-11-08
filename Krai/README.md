# MLPerf v2.1 Krai Submission

This is a repository of Krai's submission to the MLPerf v2.1 benchmark. It
includes optimized implementations of the benchmark code. The reference
implementations can be found elsewhere:
https://github.com/mlcommons/training.git

# v2.1 release

This readme was updated in Oct 2022, for the v2.1 round of MLPerf.

# Contents

The implementation(s) in the `benchmarks` subdirectory provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to run the benchmark in a container.
* Documentation on the dataset, model, and machine setup.

# Running the Benchmark

These benchmarks have been tested on the following machine configuration:

* A server with 2x NVIDIA RTX A5000s (2x24GB gpus) using MxNet 22.04.
* A server with 2x NVIDIA RTX A5000s (2x24GB gpus) using MxNet 22.08.

Please see [here](./benchmarks/resnet/implementations/mxnet/README.md) for the detail instructions in running the benchmark. 
