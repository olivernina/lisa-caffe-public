#!/usr/bin/env sh

TOOLS=../../build/tools

#GLOG_logtostderr=1 /usr/bin/gdb --args $TOOLS/caffe train -solver LCRN_activity_RGB_solver.prototxt -weights LCRN_activity_RGB_ucf101_split1_iter_30000.caffemodel

GLOG_logtostderr=1 $TOOLS/caffe train -solver LCRN_activity_RGB_solver.prototxt -weights LCRN_activity_RGB_ucf101_split1_iter_30000.caffemodel > test_RGB_video.out 2>&1

echo "Done."
