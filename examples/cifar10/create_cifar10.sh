#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=/home/lisa/caffe-LSTM-video/examples/cifar10
DATA=/home/lisa/caffe-LSTM-video/data/cifar10
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

/home/lisa/caffe-LSTM-video/build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

/home/lisa/caffe-LSTM-video/build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
