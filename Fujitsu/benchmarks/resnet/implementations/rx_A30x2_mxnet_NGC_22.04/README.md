### Steps to download data
```
./init_datasets.sh
```

## Steps to launch training

### FUJITSU PRIMERGY RX2540 M6 (single node)
Launch configuration and system-specific hyperparameters for the FUJITSU PRIMERGY RX2540 M6
single node submission are in the `config_RX2540M6_A30.sh` script.

Steps required to launch single node training on FUJITSU PRIMERGY RX2540 M6:

```
cd ../mxnet-fujitsu
docker build --pull -t mlperf-fujitsu:image_classification .
source config_RX2540M6_A30.sh
CONT=mlperf-fujitsu:image_classification DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
