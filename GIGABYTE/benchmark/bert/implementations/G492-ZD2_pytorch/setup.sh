#!/bin/bash
cd ../pytorch
docker build --pull -t mlperfv2.1-gigabyte:bert-20221004 .
