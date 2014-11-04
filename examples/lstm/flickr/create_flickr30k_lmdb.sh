#!/usr/bin/env bash
ROOT=\
./
TRAIN_LISTFILE=\
./cocoflickr/flickr30k_hdf5/buffer_100_maxwords_20/train_batches/image_list.with_dummy_labels.txt
TRAIN_OUT_DB_NAME=\
./cocoflickr/flickr30k_lmdb/train
VAL_LISTFILE=\
./cocoflickr/flickr30k_hdf5/buffer_100_maxwords_20/valid_batches/image_list.with_dummy_labels.txt
VAL_OUT_DB_NAME=\
./cocoflickr/flickr30k_lmdb/valid

RESIZE_HEIGHT=256
RESIZE_WIDTH=256
BACKEND=lmdb

../../../build/tools/convert_imageset $ROOT $TRAIN_LISTFILE $TRAIN_OUT_DB_NAME \
  -resize_height $RESIZE_HEIGHT \
  -resize_width $RESIZE_WIDTH \
  -backend $BACKEND

../../../build/tools/convert_imageset $ROOT $VAL_LISTFILE $VAL_OUT_DB_NAME \
  -resize_height $RESIZE_HEIGHT \
  -resize_width $RESIZE_WIDTH \
  -backend $BACKEND
