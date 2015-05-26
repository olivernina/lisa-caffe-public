#!/usr/bin/env sh

../../build/tools/caffe train \
    --solver=imagenet_new_normalization.prototxt --model=/x/data/Caffenets/alexnet_seed1/caffe_alexnet_train_iter_300000.caffemodel > imagenet_new_normalization_lr0p1_msra.out 2>&1 &
