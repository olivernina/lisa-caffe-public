#!/usr/bin/env sh

TOOLS=../../build/tools

#TO COME: determine which gpu has more memory

####SET UP PROTOTXT FILES

device_id=0
data_set='ucf'
source_train=ucf_train_split_all_1_leveldb
#source_train=hmdb_train_split_1_Get_leveldb

clip=16
sub=1
hidden=256
batch=384
wd='0.005'
lr='0.0001'
snap='SNAP6'
solver=$(printf "solver_sub%d_clip%d_h%s_batch%s_wd_%s.prototxt" "$sub" "$clip" "$hidden" "$batch" "$wd")
net_proto=$(printf "expLR_net_sub%d_clip%d_h%s_batch%s_wd_%s_lr_%s_%s_drop8.prototxt" "$sub" "$clip" "$hidden" "$batch" "$wd" "$lr" "$data_set")
weight_loss='false'
lstm_weight_filler_max='0.08'
lstm_weight_filler_min='-0.08'
test_max_vid='3783'
train_max_vid='9537'
gamma='0.5'

#variables for solver_prototxt
SOLVER_TEMPLATE='template:solver_hmdb_lstm.prototxt'
NET_TEMPLATE='template:hmdb_recurrent_train_ARCH2.prototxt'
OUTPUTFILE_SOLVER=$(printf "output_file:%s" "$solver")
OUTPUTFILE_NET=$(printf "output_file:%s" "$net_proto")
NET=$(printf 'net:"%s"' "$net_proto")
WEIGHT_DECAY=$(printf "weight_decay:%s" "$wd")
DEVICE_ID=$(printf "device_id:%d" "$device_id")
RANDOM_SEED=random_seed:1701
SNAPSHOT_PREFIX=$(printf 'snapshot_prefix:"expLR_snapshots/%s_imagenet_finetune_recur_clip%d_sub%d_h%d_batch%d_wd%s_lr%sdrop8_%s"' "$data_set" "$clip" "$sub" "$hidden" "$batch" "$wd" "$lr" "$snap")
BASE_LR=$(printf "base_lr:%s" "$lr")
GAMMA=$(printf "gamma:%s" "$gamma")
#variables for train/test prototxt
s=$(($batch / $clip))
CLIP_LENGTH=$(printf "data:data_param:LSTM_clip_length:%d" "$clip")
SUB_SAMPLE=$(printf "data:data_param:clip_sub_sample:%d" "$sub")
WEIGHT_LOSS=$(printf "data:data_param:weight_loss:%s" "$weight_loss")
BATCH_SIZE=$(printf "data:data_param:batch_size:%d" "$batch")
SOURCE=$(printf "data:data_param:source:%s" "$source_train")
HIDDEN=$(printf "lstm1:lstm_param:hidden_dim:%d" "$hidden")
sLSTM=$(printf "data:data_param:slstm:%s" "$s")
BUFFER_SIZE=$(printf "lstm1:lstm_param:buffer_size:%s" "$s")
LSTM_WEIGHT_FILLER_MIN=$(printf "lstm1:lstm_param:weight_filler:min:%s" "$lstm_weight_filler_min")
LSTM_WEIGHT_FILLER_MAX=$(printf "lstm1:lstm_param:weight_filler:max:%s" "$lstm_weight_filler_max")
MAX_TEST_ITEM=$(printf "data:data_param:max_test_item:%s" "$test_max_vid")
MAX_TRAIN_ITEM=$(printf "data:data_param:max_train_item:%s" "$train_max_vid")

SAVE_OUT=$(printf "expLR_%s_sub%d_clip%d_h%s_batch%s_wd%s_lr%sdrop8_%s.out" "$data_set" "$sub" "$clip" "$hidden" "$batch" "$wd" "$lr" "$snap")

echo $(printf "Data will be saved to file: %s" "$SAVE_OUT") > outfile.out 2>&1


#####

./create_solver_prototxt.py $OUTPUTFILE_SOLVER $SOLVER_TEMPLATE $NET $WEIGHT_DECAY $DEVICE_ID $RANDOM_SEED $SNAPSHOT_PREFIX $BASE_LR $GAMMA> solver.out 2>&1
./create_train_test_proto.py $NET_TEMPLATE $OUTPUTFILE_NET $CLIP_LENGTH $SUB_SAMPLE $BATCH_SIZE $SOURCE $HIDDEN $sLSTM $BUFFER_SIZE $WEIGHT_LOSS $LSTM_WEIGHT_FILLER_MIN $LSTM_WEIGHT_FILLER_MAX $MAX_TEST_ITEM $MAX_TRAIN_ITEM > train_test.out 2>&1


#increase number of hidden units
SOLVER=$solver
#GLOG_logtostderr=1 valgrind $TOOLS/caffe train -solver $SOLVER -weights caffe_imagenet_train_iter_310000 > valgrind_debugging.out 2>&1
#GLOG_logtostderr=1 ~jdonahue/gdb/gdb-7.7/gdb/gdb --args $TOOLS/caffe train -solver $SOLVER -weights caffe_imagenet_train_iter_310000  
#GLOG_logtostderr=1 $TOOLS/caffe train -solver $SOLVER -weights expLR_snapshots/ucf_imagenet_finetune_recur_clip16_sub1_h256_batch384_wd0.00375_lr0.075drop8_SNAP3_iter_4000.caffemodel> $SAVE_OUT 2>&1
GLOG_logtostderr=1 $TOOLS/caffe train -solver $SOLVER -weights expLR_snapshots/ucf_imagenet_finetune_recur_clip16_sub1_h256_batch384_wd0.005_lr0.0001drop8_SNAP5_iter_2000.caffemodel> $SAVE_OUT 2>&1
echo "Done."
