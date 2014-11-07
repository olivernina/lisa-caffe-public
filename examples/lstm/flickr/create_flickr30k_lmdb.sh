#!/usr/bin/env bash
IM_ROOT=./
ROOT=\
./cocoflickr/coco_flickr30k_hdf5/buffer_100_maxwords_20
TRAIN_LISTFILE=\
$ROOT/train_batches/image_list.with_dummy_labels.txt
OUT_DB_ROOT=\
$ROOT/lmdb
TRAIN_OUT_DB_NAME=\
$OUT_DB_ROOT/train
VAL_LISTFILE=\
$ROOT/valid_batches/image_list.with_dummy_labels.txt
VAL_OUT_DB_NAME=\
$OUT_DB_ROOT/valid

RESIZE_HEIGHT=256
RESIZE_WIDTH=256
BACKEND=lmdb

if [ -e "$OUT_DB_ROOT" ]; then
  echo "Error: output dir exists: $OUT_DB_ROOT"
  exit 1
else
  mkdir -p "$OUT_DB_ROOT"
  ../../../build/tools/convert_imageset $IM_ROOT $TRAIN_LISTFILE $TRAIN_OUT_DB_NAME \
    -resize_height $RESIZE_HEIGHT \
    -resize_width $RESIZE_WIDTH \
    -backend $BACKEND
  ../../../build/tools/convert_imageset $IM_ROOT $VAL_LISTFILE $VAL_OUT_DB_NAME \
    -resize_height $RESIZE_HEIGHT \
    -resize_width $RESIZE_WIDTH \
    -backend $BACKEND
fi
