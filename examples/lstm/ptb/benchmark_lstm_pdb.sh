#!/bin/sh

../../../build/tools/caffe time -gpu 2 -model ./ptb_lstm_benchmark_net.prototxt \
  > benchmark_`git rev-parse HEAD`.txt 2>&1
