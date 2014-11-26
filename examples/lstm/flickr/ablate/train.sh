#!/bin/bash

SOLVER=$1
GPU=$2
SNAPSHOT=$3

if [ -z "$GPU" ]; then
  echo "Usage:"
  echo "train.sh <solver> <GPU ID>"
  exit 1
fi

WEIGHTS=\
./bvlc_reference_caffenet.caffemodel

LOG=./logs/ablate/$(basename ${SOLVER})
if [ -z "$SNAPSHOT" ]; then
  INIT_ARG="--weights=${WEIGHTS}"
else
  LOG=${LOG}.continue.$(basename ${SNAPSHOT})
  INIT_ARG="--snapshot=${SNAPSHOT}"
fi
LOG=${LOG}.train_log.txt

echo "Training, logging to: ${LOG}"

caffe train \
 ${INIT_ARG} \
 -solver ${SOLVER} \
 -gpu ${GPU} \
 > ${LOG} 2>&1

echo "Done"
