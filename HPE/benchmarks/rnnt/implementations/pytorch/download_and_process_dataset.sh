#!/bin/bash

echo "--------------"
echo "--------------"
echo "DOWNLOAD START"
echo "--------------"
echo "--------------"
bash ./scripts/download_librispeech.sh
echo "--------------"
echo "--------------"
echo "DOWNLOAD ENDED"
echo "--------------"
echo "--------------"
echo "----------------"
echo "----------------"
echo "PREPROCESS START"
echo "----------------"
echo "----------------"
bash ./scripts/preprocess_librispeech.sh
echo "----------------"
echo "----------------"
echo "PREPROCESS ENDED"
echo "----------------"
echo "----------------"
