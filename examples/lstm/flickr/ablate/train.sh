#!/bin/bash

SOLVER=$1
GPU=$2

if [ -z "$GPU" ]; then
  echo "Usage:"
  echo "\ttrain.sh <solver> <GPU ID>"
  exit 1
fi

WEIGHTS=\
./bvlc_reference_caffenet.caffemodel

LOG=\
./logs/ablate/$(basename ${SOLVER}).train_log.txt

echo "Training, logging to: ${LOG}"

caffe train \
 -weights ${WEIGHTS} \
 -solver ${SOLVER} \
 -gpu ${GPU} \
 > ${LOG} 2>&1

echo "Done"
