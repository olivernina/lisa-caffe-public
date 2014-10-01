#!/bin/sh

TRAIN_DIR=./train_batches
VAL_DIR=./val_batches

rm -rf $TRAIN_DIR
./generate_hdf5_data.py $TRAIN_DIR

rm -rf $VAL_DIR
./generate_hdf5_data.py $VAL_DIR
