## Steps to launch training

### NVIDIA DGX H100

Launch configuration and system-specific hyperparameters for the NVIDIA DGX
H100 submission are in the `../<implementation>/config_DGXH100_1x8x192x1.sh` script.

Steps required to launch training on NVIDIA DGX H100.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
cd ../pytorch
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_DGXH100_1x8x192x1.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
