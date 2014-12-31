#!/usr/bin/env sh
# Create the hmdb video leveldb inputs
# usage: [split_id] is optional; should be 1, 2, or 3 if used.
#

SPLIT_ID=1
RGB_OR_FLOW='RGB'
TOOLS=../../../build/tools

echo "Creating train leveldb..."

GLOG_logtostderr=1 $TOOLS/convert_videoset.bin \
    PATH_TO_FRAMES \
    ucf101_${RGB_OR_FLOW}_test_split_${SPLIT_ID}.txt \
    PATH_TO_TEST_LEVELDB

echo "Creating test leveldb..."

GLOG_logtostderr=1 $TOOLS/convert_videoset.bin \
    PATH_TO_FRAMES \
    ucf101_${RGB_OR_FLOW}_train_split_${SPLIT_ID}.txt \
    PATH_TO_TRAIN_LEVELDB

echo "Done."
