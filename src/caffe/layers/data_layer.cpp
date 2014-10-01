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
  this->video_id_ = 0;
  // Initialize the DB and rand_skip.
  Reset();

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

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
  int clip_length_ = this->layer_param_.data_param().clip_length();
  if (this->layer_param_.data_param().clip_mode() == DataParameter_ClipMode_FIXED_LENGTH) {
    CHECK_EQ(0, this->layer_param_.data_param().batch_size() % clip_length_)
        << "If using fixed length clips, the batch size must be an exact "
        << "multiple of the clip length to avoid adding unnecessary padding. "
        << "Consider setting batch_size = " << clip_length_ *
          (this->layer_param_.data_param().batch_size() / clip_length_);
  }


  if (this->layer_param_.data_param().clip_order() == DataParameter_ClipOrder_FRAME_MAJOR) {
    top[0]->set_frame_major_clip_length(clip_length_);
    CHECK_EQ(this->layer_param_.data_param().clip_mode(), DataParameter_ClipMode_FIXED_LENGTH)
        << "FRAME_MAJOR clip_order requires FIXED_LENGTH clip_mode.";
  } else {
    top[0]->set_frame_major_clip_length(0);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    if (this->layer_param_.data_param().clip_collapse_labels()) {  //this->clip_collapse_labels_
      CHECK_EQ(this->layer_param_.data_param().clip_mode(), DataParameter_ClipMode_FIXED_LENGTH)
          << "clip_collapse_labels requires fixed_length clip_mode.";
      const int collapsed_label_num = this->layer_param_.data_param().batch_size() / clip_length_;
      top[1]->Reshape(collapsed_label_num, 1, 1, 1);
      this->prefetch_label_.Reshape(collapsed_label_num, 1, 1, 1);
    } else {
      top[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
      this->prefetch_label_.Reshape(
          this->layer_param_.data_param().batch_size(), 1, 1, 1);
    }
  }
  if (this->output_clip_markers_) {
    top[2]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_clip_markers_.Reshape(
        this->layer_param_.data_param().batch_size(), 1, 1, 1);
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();

  if (this->output_clip_markers_) {
    const int count = this->prefetch_clip_markers_.count();
    Dtype* prefetch_clip_markers = this->prefetch_clip_markers_.mutable_cpu_data();
    if (this->layer_param_.data_param().clip_mode() == DataParameter_ClipMode_FIXED_LENGTH) {
    // Prefill markers for fixed length batch size, as they'll never change.
      for (int i = 0; i < count; ++i) {
        prefetch_clip_markers[i] =
            (i % clip_length_ > 0) ? CLIP_CONTINUE : CLIP_BEGIN;
      }
    } else {
      caffe_set(count, Dtype(PADDING), prefetch_clip_markers);
    }
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry() {
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_clip_markers = NULL;
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  if (this->output_clip_markers_) {
    top_clip_markers = this->prefetch_clip_markers_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    bool first_video = true;
    // get a blob
    Datum datum;
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datum.ParseFromString(iter_->value().ToString());
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
    // Apply data transformations (mirror, scale, crop...)
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_.Transform(datum, &(this->transformed_data_), first_video);
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }

    // go to the next iter
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
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
  }
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
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
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
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
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

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
