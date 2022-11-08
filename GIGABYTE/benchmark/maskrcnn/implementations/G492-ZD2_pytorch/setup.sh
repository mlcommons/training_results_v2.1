#!/bin/bash
cd ../pytorch
docker build --pull -t mlperfv2.1-gigabyte:maskrcnn-20221004 .
