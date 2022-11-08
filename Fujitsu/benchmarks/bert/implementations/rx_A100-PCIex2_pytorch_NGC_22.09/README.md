## Steps to launch training on a single node

### FUJITSU PRIMERGY RX2540M6 (single node)
Launch configuration and system-specific hyperparameters for the PRIMERGY RX2540M6
single node submission are in the following scripts:
* for the 1-node PRIMERGY RX2540M6 submission: `config_RX2540M6.sh`

Steps required to launch training on PRIMERGY RX2540M6:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-fujitsu:language_model .
docker push <docker/registry>/mlperf-fujitsu:language_model
```

2. Launch the training:

1-node PRIMERGY RX2540M6 training:

```
cd ../pytorch-fujitsu
bash bert_RX.sh
```
