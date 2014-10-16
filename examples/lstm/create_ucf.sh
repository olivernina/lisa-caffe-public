#!/usr/bin/env sh
# Create the hmdb video leveldb inputs
# N.B. set the path to the video train + val data dirs
# usage: [split_id] is optional; should be 1, 2, or 3 if used.
#
#   ./create_hmdb.sh [split_id]

SPLIT_ID=$1
if [ -z "$SPLIT_ID" ]; then
    SPLIT_ID=1
fi
TOOLS=../../build/tools

echo "Creating train leveldb..."

#~jdonahue/gdb/gdb-7.7/gdb/gdb --args

#GLOG_logtostderr=1 ~jdonahue/gdb/gdb-7.7/gdb/gdb --args $TOOLS/convert_videoset.bin \
#    /x/data/ucf101/frames/ \
#    ucf101_train_split_${SPLIT_ID}.txt \
#    /x/data/ucf101/ucf_train_split_all_${SPLIT_ID}_leveldb_test 0

#echo "Creating test leveldb..."

GLOG_logtostderr=1 $TOOLS/convert_videoset.bin \
    /x/data/ucf101/frames/ \
    ucf101_test_split_${SPLIT_ID}.txt \
    /x/data/ucf101/ucf_test_split_all_${SPLIT_ID}_leveldb_test 0

echo "Done."
