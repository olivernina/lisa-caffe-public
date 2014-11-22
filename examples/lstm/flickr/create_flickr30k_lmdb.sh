#!/usr/bin/env bash
SET_NAME=train_flickr
IM_ROOT=./
ROOT=\
./cocoflickr/coco_flickr30k_hdf5/buffer_100_maxwords_20
LISTFILE=\
$ROOT/${SET_NAME}_batches/image_list.with_dummy_labels.txt
OUT_DB_ROOT=\
$ROOT/lmdb/${SET_NAME}
OUT_DB_NAME=\
${OUT_DB_ROOT}/${SET_NAME}

RESIZE_HEIGHT=256
RESIZE_WIDTH=256
BACKEND=lmdb

if [ -e "$OUT_DB_NAME" ]; then
  echo "Error: output dir exists: $OUT_DB_NAME"
  exit 1
else
  mkdir -p "$OUT_DB_ROOT"
  ../../../build/tools/convert_imageset \
    $IM_ROOT $LISTFILE $OUT_DB_NAME \
    -resize_height $RESIZE_HEIGHT \
    -resize_width $RESIZE_WIDTH \
    -backend $BACKEND
fi
