// Copyright 2014 BVLC and contributors.

#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using std::string;
using std::vector;

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() > 1) {
    this->output_labels_ = true;
  } else {
    this->output_labels_ = false;
  }
  if (top.size() > 2) {
    this->output_clip_markers_ = true;
  } else {
    this->output_clip_markers_ = false;
  }
  if (top.size() > 3) {
    this->weight_loss_ = true;
  } else {
    this->weight_loss_ = false;
  }
  this->video_id_ = 0;

  this->batch_videos_ = this->layer_param_.data_param().batch_videos(); 
  CHECK_EQ(0,this->layer_param_.data_param().batch_size() % this->batch_videos_) <<  "Batch size must be divisible by batch_videos";
  
  this->batch_frames_ = this->layer_param_.data_param().batch_size() / this->batch_videos_; 
  // Initialize the DB and rand_skip.
  Reset();


  if (this->layer_param_.data_param().backend() == DataParameter_DB_LEVELDB) {
    iter_[0]->SeekToLast();
    string tmp_key = iter_[0]->key().ToString(); 
    tmp_key = tmp_key.substr(0,8);
    this->max_video_ = atoi(tmp_key.c_str());
    iter_[0]->SeekToFirst();
  } else {
    this->max_video_ = 1;
  }

  // Read a data point, and use it to initialize the top blob.
  Datum datum = load_datum(0, 0) ;

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    top[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
    this->transformed_data_.Reshape(1, datum.channels(), crop_size, crop_size);
  } else {
    top[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
    this->transformed_data_.Reshape(1, datum.channels(),
      datum.height(), datum.width());
  }

  //video

  this->clip_length_ = this->layer_param_.data_param().lstm_clip_length();

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    top[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(
        this->layer_param_.data_param().batch_size(), 1, 1, 1);
  }
  if (this->output_clip_markers_) {
    top[2]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_clip_markers_.Reshape(
        this->layer_param_.data_param().batch_size(), 1, 1, 1);
    const int count = this->prefetch_clip_markers_.count();
    Dtype* prefetch_clip_markers = this->prefetch_clip_markers_.mutable_cpu_data();
  }
  if (this->weight_loss_) {
    top[3]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_weight_loss_.Reshape(
        this->layer_param_.data_param().batch_size(), 1, 1, 1);
    const int count = this->prefetch_weight_loss_.count();
    Dtype* prefetch_weight_loss = this->prefetch_weight_loss_.mutable_cpu_data();
    caffe_set(count, Dtype(1), prefetch_weight_loss);
  }

}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry() {
  int count = 0;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_clip_markers = NULL;
  Dtype* top_weight_loss = NULL;
  const Dtype pad_value = this->layer_param_.data_param().clip_pad_value();
  const int sub_sample = this->layer_param_.data_param().clip_sub_sample();
  const bool clip_collapse_labels = this->layer_param_.data_param().clip_collapse_labels();
  int collapsed_label_id = 0;
  const int label_num = this->prefetch_label_.num();

  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  if (this->output_clip_markers_) {
    top_clip_markers = this->prefetch_clip_markers_.mutable_cpu_data();
  }
  if (this->weight_loss_) {
    top_weight_loss = this->prefetch_weight_loss_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();
  int max_video = this->max_video_;
  CHECK_GE(max_video,0)  << "Need to have more videos than 0.";

  int iter_index = 0;
  int frame_id;
  for (int item_id = 0; item_id < batch_size; ) {
    bool first_video = true;
    // get a blob
    bool continuing_video = false;
    if (this->transfer_frame_ids_[iter_index] > 0) {
      continuing_video = true;
      first_video = false;  
    }

    // Read in first blob from video to initialize everything.
    if (this->video_id_ > max_video){
      this->video_id_ = 0;
    }
    int first_frame = 0;
    int current_video = this->video_id_;
    if (continuing_video){
      first_frame = this->transfer_frame_ids_[iter_index];
      current_video = this->transfer_video_ids_[iter_index];
    }

    Datum datum = load_datum(current_video, first_frame);

    //If frame is too small, just skip over it
    while ((datum.height() < this->layer_param_.transform_param().crop_size()) | (datum.width() < this->layer_param_.transform_param().crop_size())){
      ++this->video_id_;
      current_video = this->video_id_;
      datum = load_datum(current_video, first_frame);
    }
    

    const int num_frames = datum.frames();
    int current_frame = datum.current_frame();
    CHECK_GT(num_frames, 0) << "Input had no frames.";
    int output_length;
    int input_offset = 0;
    int output_offset = 0;
    int video_length = 0;

    const int out_frame_size =
        datum.channels()* (this->layer_param_.transform_param().crop_size() ? pow(this->layer_param_.transform_param().crop_size(), 2) : (datum.height() * datum.width()));

    int remaining_items = this->batch_frames_ - ((item_id + this->batch_frames_) % this->batch_frames_);
    if (this->layer_param_.data_param().lstm_clip()) {
      CHECK_GE(num_frames, this->layer_param_.data_param().lstm_clip());
      video_length = this->layer_param_.data_param().lstm_clip_length()*sub_sample;
      if (num_frames > this->clip_length_){
        output_length = this->clip_length_*sub_sample;
        if (!continuing_video) {
          input_offset = this->input_offset(num_frames, sub_sample);
          output_offset = this->output_offset(num_frames, sub_sample);
        }
      } else {
        output_length = (num_frames - current_frame) *sub_sample;
      }  
    } else {
      video_length = num_frames; 
      output_length = (num_frames-current_frame)*sub_sample;
    }

    output_length = std::min(remaining_items*sub_sample, output_length);
    //frame loop
    int out_frame_id = current_frame;
    for (out_frame_id; out_frame_id < output_length + first_frame;
        out_frame_id += sub_sample, ++item_id) {

      frame_id = out_frame_id - output_offset + input_offset;
      //CHECK_LT(frame_id, num_frames);

      int frame_major_id = item_id/this->batch_frames_ + (item_id % this->batch_frames_) * this->batch_videos_;

      if (frame_id != 0){ //if frame_id is zero than the frame loaded currently is the frame we want
        datum = load_datum(current_video, frame_id);
      }
      current_frame = datum.current_frame();
      if (DataParameter_DB_LEVELDB){
        CHECK_EQ(frame_id, current_frame) << "LMDB? " << DataParameter_DB_LMDB;
        CHECK_GE(frame_id, 0);
        CHECK_LT(frame_id, num_frames);
      }
      // Apply data transformations (mirror, scale, crop...).  Use predetermined h_off and w_off.  
      // False indicates that we will not recalculate these values.
      CHECK_LT(item_id, batch_size);
      int offset = this->prefetch_data_.offset(frame_major_id);
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_.Transform(datum, &(this->transformed_data_), first_video,iter_index);
      first_video = false;

      if (this->output_labels_) {
          top_label[frame_major_id] = datum.label();
      }
      if (this->output_clip_markers_) {
        top_clip_markers[frame_major_id] = (out_frame_id == output_offset) ?
            DataLayer<Dtype>::CLIP_BEGIN : DataLayer<Dtype>::CLIP_CONTINUE;
      } 
      if (this->weight_loss_) {
        top_weight_loss[frame_major_id] = (out_frame_id == output_length-1) ?
            1 : 0;
      } 
    } //for (out_frame_id = 0; out_frame_id < output_length) 
    // go to the next iter
    if (item_id > this->batch_frames_*(iter_index+1) - 1) {
      //keep track of which frame we want to start on at iter_index with next batch
      if (out_frame_id + sub_sample >= video_length) {
        this->transfer_frame_ids_[iter_index] = 0;
        this->transfer_video_ids_[iter_index] = this->video_id_+1; 
      } else {
        this->transfer_frame_ids_[iter_index] = frame_id + sub_sample;
        this->transfer_video_ids_[iter_index] = current_video; 
      }
      iter_index += 1;
    } else {
      this->transfer_frame_ids_[iter_index] = 0;
    }    
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      break;
    case DataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
    if (!continuing_video) {
      ++this->video_id_;
    }      
  }  // while (item_id < batch_size)
} 

template <typename Dtype>
void DataLayer<Dtype>::Reset() {
  // Initialize the DB.
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);


   //adding a few extra variables for debugging


   iter_.resize(this->batch_videos_);

    for (int i = 0; i < this->batch_videos_; i++){
      iter_[i].reset(db_->NewIterator(leveldb::ReadOptions()));
      iter_[i]->SeekToFirst();
    }


//OLD CODE:
//    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
//    iter_->SeekToFirst();


    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  this->transfer_frame_ids_.resize(this->batch_videos_);
  this->transfer_video_ids_.resize(this->batch_videos_);

  for (int i = 0; i < this->batch_videos_; i++){
    this->transfer_frame_ids_[i] = 0;  //all frames equal zero....
    this->transfer_video_ids_[i] = 0;  //all frames equal zero....
  }

  // Check if we need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    if (!skip_initialized_) {
      skip_ = caffe_rng_rand() % this->layer_param_.data_param().rand_skip();
    }
    LOG(INFO) << "Skipping first " << skip_ << " data points.";
    unsigned int skip = skip_;
    while (skip-- > 0) {
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_[0]->Next();
        if (!iter_[0]->Valid()) {
          iter_[0]->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
    skip_initialized_ = true;
  }
}

template <typename Dtype>
int DataLayer<Dtype>::input_offset(const int num_frames,
    const int sub_sample) {
  const int crop_needed =
      num_frames - ((this->layer_param_.data_param().lstm_clip_length() * sub_sample) - sub_sample + 1);
  if (crop_needed <= 0) { return 0; }
  if (!this->layer_param_.data_param().clip_allow_crop()) {
    LOG(FATAL) << "Clip (length " << num_frames << ") is longer than "
        "fixed clip length (" << this->clip_length_ << ") "
        "and clip_allow_crop was not set.";
  }
  switch (this->layer_param_.data_param().clip_crop_mode()) {
  case DataParameter_ClipPadCropMode_RANDOM:
    return this->data_transformer_.Rand(crop_needed+1);
  case DataParameter_ClipPadCropMode_START:
    return 0;
  case DataParameter_ClipPadCropMode_CENTER:
    return crop_needed / 2;
  case DataParameter_ClipPadCropMode_END:
    return crop_needed;
  default:
    LOG(FATAL) << "Unknown clip crop mode: " <<
        DataParameter_ClipPadCropMode_Name(
            this->layer_param_.data_param().clip_crop_mode());
  }
  LOG(FATAL) << "Shouldn't reach this line; switch returns or LOG(FATAL)s.";
  return 0;
}

template <typename Dtype>
int DataLayer<Dtype>::output_offset(const int num_frames,
    const int sub_sample) {
  const int pad_needed =
      (this->layer_param_.data_param().lstm_clip_length() * sub_sample) - sub_sample + 1 - num_frames;
  if (pad_needed <= 0) { return 0; }
  if (!this->layer_param_.data_param().clip_allow_pad()) {
    LOG(FATAL) << "Clip (length " << num_frames << ") is shorter than "
        "fixed clip length (" << this->clip_length_ <<
        ") and clip_allow_pad was not set.";
  }
  switch (this->layer_param_.data_param().clip_pad_mode()) {
  case DataParameter_ClipPadCropMode_RANDOM:
    return this->data_transformer_.Rand(pad_needed+1);
  case DataParameter_ClipPadCropMode_START:
    return 0;
  case DataParameter_ClipPadCropMode_CENTER:
    return pad_needed / 2;
  case DataParameter_ClipPadCropMode_END:
    return pad_needed;
  default:
    LOG(FATAL) << "Unknown clip pad mode: " <<
        DataParameter_ClipPadCropMode_Name(
            this->layer_param_.data_param().clip_pad_mode());
  }
  LOG(FATAL) << "Shouldn't reach this line; switch returns or LOG(FATAL)s.";
  return 0;
}

template <typename Dtype>
Datum DataLayer<Dtype>::load_datum(const int current_video, const int frame_id) {
   Datum datum;
  int length_key;
  char my_key[17];
  std::string value;

   switch (this->layer_param_.data_param().backend()) {
     case DataParameter_DB_LEVELDB:
     length_key = snprintf(my_key, 17, "%08d%08d", current_video, frame_id); 
     db_->Get(leveldb::ReadOptions(), my_key, &value);
     datum.ParseFromString(value);
     break;
   case DataParameter_DB_LMDB:
     CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
             &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
     datum.ParseFromArray(mdb_value_.mv_data,
         mdb_value_.mv_size);
     break;
   default:
     LOG(FATAL) << "Unknown database backend";
   }
   return datum;
}

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
