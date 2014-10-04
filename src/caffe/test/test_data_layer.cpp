// Copyright 2014 BVLC and contributors.

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "leveldb/db.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class DataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DataLayerTest()
      : backend_(DataParameter_DB_LEVELDB),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        blob_top_clip_markers_(new Blob<Dtype>()),
        seed_(1701) {}
  virtual void SetUp() {
    filename_.reset(new string());
    MakeTempDir(filename_.get());
    *filename_ += "/db";
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  void FillLevelDB(const bool unique_pixels, const int clip_length = 1,
                   const int num_clips = 5) {
    backend_ = DataParameter_DB_LEVELDB;
    LOG(INFO) << "Using temporary leveldb " << *filename_;
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    leveldb::Status status =
        leveldb::DB::Open(options, filename_->c_str(), &db);
    CHECK(status.ok());
    int count = 0;
    char key_cstr[17];
    for (int i = 0; i < num_clips; ++i) {
      for (int frame_id = 0; frame_id < clip_length; ++frame_id) {
        Datum datum;
        datum.set_label(i);
        datum.set_frames(clip_length);
        datum.set_channels(2);
        datum.set_height(3);
        datum.set_width(4);
        std::string* data = datum.mutable_data();
        datum.set_current_frame(frame_id);  //need to check with jeff
        for (int j = 0; j < 24; ++j) {
          int datum = (unique_pixels ? j : i) + (10 * frame_id);
          data->push_back(static_cast<uint8_t>(datum));
        }
        int n = sprintf(key_cstr, "%08d%08d", i, frame_id);
        db->Put(leveldb::WriteOptions(), string(key_cstr), datum.SerializeAsString());
        ++count;
      }
    }
    delete db;
  }

  // Fill the LMDB with data: unique_pixels has same meaning as in FillLevelDB.
  void FillLMDB(const bool unique_pixels) {
    backend_ = DataParameter_DB_LMDB;
    LOG(INFO) << "Using temporary lmdb " << *filename_;
    CHECK_EQ(mkdir(filename_->c_str(), 0744), 0) << "mkdir " << filename_
                                                 << "failed";
    MDB_env *env;
    MDB_dbi dbi;
    MDB_val mdbkey, mdbdata;
    MDB_txn *txn;
    CHECK_EQ(mdb_env_create(&env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(env, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(env, filename_->c_str(), 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(env, NULL, 0, &txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(txn, NULL, 0, &dbi), MDB_SUCCESS) << "mdb_open failed";

    for (int i = 0; i < 5; ++i) {
      Datum datum;
      datum.set_label(i);
      datum.set_channels(2);
      datum.set_height(3);
      datum.set_width(4);
      std::string* data = datum.mutable_data();
      for (int j = 0; j < 24; ++j) {
        int datum = unique_pixels ? j : i;
        data->push_back(static_cast<uint8_t>(datum));
      }
      stringstream ss;
      ss << i;

      string value;
      datum.SerializeToString(&value);
      mdbdata.mv_size = value.size();
      mdbdata.mv_data = reinterpret_cast<void*>(&value[0]);
      string keystr = ss.str();
      mdbkey.mv_size = keystr.size();
      mdbkey.mv_data = reinterpret_cast<void*>(&keystr[0]);
      CHECK_EQ(mdb_put(txn, dbi, &mdbkey, &mdbdata, 0), MDB_SUCCESS)
          << "mdb_put failed";
    }
    CHECK_EQ(mdb_txn_commit(txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    mdb_close(env, dbi);
    mdb_env_close(env);
  }

  void TestRead() {
    const Dtype scale = 3;
    LayerParameter param;
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(5);
    data_param->set_max_train_item(5);
    data_param->set_max_test_item(5);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);

    DataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 5);
    EXPECT_EQ(blob_top_data_->channels(), 2);
    EXPECT_EQ(blob_top_data_->height(), 3);
    EXPECT_EQ(blob_top_data_->width(), 4);
    EXPECT_EQ(blob_top_label_->num(), 5);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 1);
    EXPECT_EQ(blob_top_label_->width(), 1);

    for (int iter = 0; iter < 100; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(scale * i, blob_top_data_->cpu_data()[i * 24 + j])
              << "debug: iter " << iter << " i " << i << " j " << j;
        }
      }
    }
  }

  void TestReadCrop() {
    const Dtype scale = 3;
    LayerParameter param;
    Caffe::set_random_seed(1701);

    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(5);
    data_param->set_max_train_item(5);
    data_param->set_max_test_item(5);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);
    transform_param->set_crop_size(1);

    DataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 5);
    EXPECT_EQ(blob_top_data_->channels(), 2);
    EXPECT_EQ(blob_top_data_->height(), 1);
    EXPECT_EQ(blob_top_data_->width(), 1);
    EXPECT_EQ(blob_top_label_->num(), 5);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 1);
    EXPECT_EQ(blob_top_label_->width(), 1);

    for (int iter = 0; iter < 2; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      int num_with_center_value = 0;
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 2; ++j) {
          const Dtype center_value = scale * (j ? 17 : 5);
          num_with_center_value +=
              (center_value == blob_top_data_->cpu_data()[i * 2 + j]);
          // At TEST time, check that we always get center value.
          if (Caffe::phase() == Caffe::TEST) {
            EXPECT_EQ(center_value, this->blob_top_data_->cpu_data()[i * 2 + j])
                << "debug: iter " << iter << " i " << i << " j " << j;
          }
        }
      }
      // At TRAIN time, check that we did not get the center crop all 10 times.
      // (This check fails with probability 1-1/12^10 in a correct
      // implementation, so we call set_random_seed.)
      if (Caffe::phase() == Caffe::TRAIN) {
        EXPECT_LT(num_with_center_value, 10);
      }
    }
  }

  void TestReadCropTrainSequenceSeeded() {
    LayerParameter param;
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(5);
    data_param->set_max_train_item(5);
    data_param->set_max_test_item(5);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_crop_size(1);
    transform_param->set_mirror(true);

    // Get crop sequence with Caffe seed 1701.
    Caffe::set_random_seed(seed_);
    vector<vector<Dtype> > crop_sequence;
    {
      DataLayer<Dtype> layer1(param);
      layer1.SetUp(blob_bottom_vec_, blob_top_vec_);
      for (int iter = 0; iter < 2; ++iter) {
        layer1.Forward(blob_bottom_vec_, blob_top_vec_);
        for (int i = 0; i < 5; ++i) {
          EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
        }
        vector<Dtype> iter_crop_sequence;
        for (int i = 0; i < 5; ++i) {
          for (int j = 0; j < 2; ++j) {
            iter_crop_sequence.push_back(
                blob_top_data_->cpu_data()[i * 2 + j]);
          }
        }
        crop_sequence.push_back(iter_crop_sequence);
      }
    }  // destroy 1st data layer and unlock the leveldb

    // Get crop sequence after reseeding Caffe with 1701.
    // Check that the sequence is the same as the original.
    Caffe::set_random_seed(seed_);
    DataLayer<Dtype> layer2(param);
    layer2.SetUp(blob_bottom_vec_, blob_top_vec_);
    for (int iter = 0; iter < 2; ++iter) {
      layer2.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 2; ++j) {
          EXPECT_EQ(crop_sequence[iter][i * 2 + j],
                    blob_top_data_->cpu_data()[i * 2 + j])
              << "debug: iter " << iter << " i " << i << " j " << j;
        }
      }
    }
  }

  void TestReadCropTrainSequenceUnseeded() {
    LayerParameter param;
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(5);
    data_param->set_max_train_item(5);
    data_param->set_max_test_item(5);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_crop_size(1);
    transform_param->set_mirror(true);

    // Get crop sequence with Caffe seed 1701, srand seed 1701.
    Caffe::set_random_seed(seed_);
    srand(seed_);
    vector<vector<Dtype> > crop_sequence;
    {
      DataLayer<Dtype> layer1(param);
      layer1.SetUp(blob_bottom_vec_, blob_top_vec_);
      for (int iter = 0; iter < 2; ++iter) {
        layer1.Forward(blob_bottom_vec_, blob_top_vec_);
        for (int i = 0; i < 5; ++i) {
          EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
        }
        vector<Dtype> iter_crop_sequence;
        for (int i = 0; i < 5; ++i) {
          for (int j = 0; j < 2; ++j) {
            iter_crop_sequence.push_back(
                blob_top_data_->cpu_data()[i * 2 + j]);
          }
        }
        crop_sequence.push_back(iter_crop_sequence);
      }
    }  // destroy 1st data layer and unlock the leveldb

    // Get crop sequence continuing from previous Caffe RNG state; reseed
    // srand with 1701. Check that the sequence differs from the original.
    srand(seed_);
    DataLayer<Dtype> layer2(param);
    layer2.SetUp(blob_bottom_vec_, blob_top_vec_);
    for (int iter = 0; iter < 2; ++iter) {
      layer2.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      int num_sequence_matches = 0;
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 2; ++j) {
          num_sequence_matches += (crop_sequence[iter][i * 2 + j] ==
                                   blob_top_data_->cpu_data()[i * 2 + j]);
        }
      }
      EXPECT_LT(num_sequence_matches, 10);
    }
  }

  void TestReadFixedLengthClipsFrameMajor(const int clip_length,
                                          const int batch_size, const int sub_sample = 1) {
    LayerParameter param;
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(batch_size);
    data_param->set_clip_length(clip_length);
    data_param->set_clip_mode(DataParameter_ClipMode_FIXED_LENGTH);
    data_param->set_clip_order(DataParameter_ClipOrder_FRAME_MAJOR);
    data_param->set_max_train_item(batch_size);
    data_param->set_max_test_item(batch_size);
    data_param->set_clip_sub_sample(sub_sample);
    const Dtype scale = 1;
    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);
    data_param->set_scale(scale);
    data_param->set_source(this->filename_->c_str());
    DataLayer<Dtype> layer(param);
    this->blob_top_vec_.push_back(this->blob_top_clip_markers_);
    DataLayer<Dtype> layer1(param);
    layer1.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(this->blob_top_data_->num(), 6);
    EXPECT_EQ(this->blob_top_data_->channels(), 2);
    EXPECT_EQ(this->blob_top_data_->height(), 3);
    EXPECT_EQ(this->blob_top_data_->width(), 4);
    EXPECT_EQ(this->blob_top_label_->num(), 6);
    EXPECT_EQ(this->blob_top_label_->channels(), 1);
    EXPECT_EQ(this->blob_top_label_->height(), 1);
    EXPECT_EQ(this->blob_top_label_->width(), 1);

    for (int iter = 0; iter < 100; ++iter) {
      layer1.Forward(blob_bottom_vec_, blob_top_vec_);
      EXPECT_EQ(0, this->blob_top_label_->cpu_data()[0]);
      EXPECT_EQ(1, this->blob_top_label_->cpu_data()[1]);
      EXPECT_EQ(0, this->blob_top_label_->cpu_data()[2]);
      EXPECT_EQ(1, this->blob_top_label_->cpu_data()[3]);
      EXPECT_EQ(0, this->blob_top_label_->cpu_data()[4]);
      EXPECT_EQ(1, this->blob_top_label_->cpu_data()[5]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_BEGIN,
                this->blob_top_clip_markers_->cpu_data()[0]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_BEGIN,
                this->blob_top_clip_markers_->cpu_data()[1]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[2]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[3]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[4]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[5]);
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
          const Dtype expected = 10 * sub_sample * (i / 2) + i % 2;
          EXPECT_EQ(expected,
                    this->blob_top_data_->cpu_data()[i * 24 + j])
              << " i = " << i << "; j = " << j;
        }
      }

      layer1.Forward(blob_bottom_vec_, blob_top_vec_);
      EXPECT_EQ(2, this->blob_top_label_->cpu_data()[0]);
      EXPECT_EQ(3, this->blob_top_label_->cpu_data()[1]);
      EXPECT_EQ(2, this->blob_top_label_->cpu_data()[2]);
      EXPECT_EQ(3, this->blob_top_label_->cpu_data()[3]);
      EXPECT_EQ(2, this->blob_top_label_->cpu_data()[4]);
      EXPECT_EQ(3, this->blob_top_label_->cpu_data()[5]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_BEGIN,
                this->blob_top_clip_markers_->cpu_data()[0]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_BEGIN,
                this->blob_top_clip_markers_->cpu_data()[1]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[2]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[3]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[4]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[5]);
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
          const Dtype expected = 10 * sub_sample * (i / 2) + i % 2 + 2;
          EXPECT_EQ(expected,
                    this->blob_top_data_->cpu_data()[i * 24 + j])
              << " i = " << i << "; j = " << j;
        }
      }

      layer1.Forward(blob_bottom_vec_, blob_top_vec_);
      EXPECT_EQ(4, this->blob_top_label_->cpu_data()[0]);
      EXPECT_EQ(5, this->blob_top_label_->cpu_data()[1]);
      EXPECT_EQ(4, this->blob_top_label_->cpu_data()[2]);
      EXPECT_EQ(5, this->blob_top_label_->cpu_data()[3]);
      EXPECT_EQ(4, this->blob_top_label_->cpu_data()[4]);
      EXPECT_EQ(5, this->blob_top_label_->cpu_data()[5]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_BEGIN,
                this->blob_top_clip_markers_->cpu_data()[0]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_BEGIN,
                this->blob_top_clip_markers_->cpu_data()[1]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[2]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[3]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[4]);
      EXPECT_EQ(DataLayer<Dtype>::CLIP_CONTINUE,
                this->blob_top_clip_markers_->cpu_data()[5]);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 24; ++j) {
        const Dtype expected = 10 * sub_sample * (i / 2) + i % 2 + 4;
          EXPECT_EQ(expected,
                    this->blob_top_data_->cpu_data()[i * 24 + j])
              << " i = " << i << "; j = " << j;
        }
      }
    }
  }

  void TestReadFixedLengthClipsCollapsedLabels(const int clip_length,
                                               const int batch_size, const int sub_sample = 1) {
    LayerParameter param;
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(batch_size);
    data_param->set_clip_length(clip_length);
    data_param->set_clip_mode(DataParameter_ClipMode_FIXED_LENGTH);
    data_param->set_clip_collapse_labels(true);
    data_param->set_max_train_item(batch_size);
    data_param->set_max_test_item(batch_size);
    data_param->set_clip_sub_sample(sub_sample);
    const Dtype scale = 3;
    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);
     
    data_param->set_scale(scale);
    data_param->set_source(this->filename_->c_str());
    DataLayer<Dtype> layer(param);
    this->blob_top_vec_.push_back(this->blob_top_clip_markers_);
    DataLayer<Dtype> layer1(param);
    layer1.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(this->blob_top_data_->num(), 6);
    EXPECT_EQ(this->blob_top_data_->channels(), 2);
    EXPECT_EQ(this->blob_top_data_->height(), 3);
    EXPECT_EQ(this->blob_top_data_->width(), 4);
    EXPECT_EQ(this->blob_top_label_->num(), 2);
    EXPECT_EQ(this->blob_top_label_->channels(), 1);
    EXPECT_EQ(this->blob_top_label_->height(), 1);
    EXPECT_EQ(this->blob_top_label_->width(), 1);
    EXPECT_EQ(this->blob_top_clip_markers_->num(), 6);
    EXPECT_EQ(this->blob_top_clip_markers_->channels(), 1);
    EXPECT_EQ(this->blob_top_clip_markers_->height(), 1);
    EXPECT_EQ(this->blob_top_clip_markers_->width(), 1);

    int clip_index = 0;
    const int pad_start = (batch_size / clip_length) * clip_length;
    for (int iter = 0; iter < 10; ++iter) {
      layer1.Forward(blob_bottom_vec_, blob_top_vec_);
      const int clip_start_index = clip_index;
      for (int i = 0; i < batch_size / clip_length; i++) {
        const Dtype expected_value = clip_index % batch_size + i;  //labels should not depend on sub sampling
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i])
            << "debug: iter " << iter << " i " << i;
        if (i % clip_length == clip_length - 1) { ++clip_index; }
      }
      clip_index = clip_start_index;
      for (int i = 0; i < batch_size; ++i) {
        Dtype expected_value = (i < pad_start) ?
            (scale * (clip_index % batch_size + (i % clip_length) * 10*sub_sample)) : 0;
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j])
              << "debug: iter " << iter << " i " << i << " j " << j;
        }
        if (i % clip_length == clip_length - 1) { ++clip_index; }
      }
      for (int i = 0; i < batch_size; ++i) {
        const Dtype expected_value = (i < pad_start) ?
           ((i % clip_length) ? DataLayer<Dtype>::CLIP_CONTINUE :
                                DataLayer<Dtype>::CLIP_BEGIN) :
           DataLayer<Dtype>::PADDING;
        EXPECT_EQ(expected_value, this->blob_top_clip_markers_->cpu_data()[i])
            << "debug: iter " << iter << " i " << i;
      }
    }
  }

  void TestReadFixedLengthClips(const int clip_length, const int batch_size) {
    LayerParameter param;
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(batch_size);
    data_param->set_clip_length(clip_length);
    data_param->set_clip_mode(DataParameter_ClipMode_VARIABLE);
    data_param->set_max_train_item(batch_size);
    data_param->set_max_test_item(batch_size);
    const Dtype pad_value = 27281;
    data_param->set_clip_pad_value(pad_value);
    const Dtype scale = 3;
    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);
    data_param->set_source(this->filename_->c_str());
    DataLayer<Dtype> layer(param);
    this->blob_top_vec_.push_back(this->blob_top_clip_markers_);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(this->blob_top_data_->num(), batch_size);
    EXPECT_EQ(this->blob_top_data_->channels(), 2);
    EXPECT_EQ(this->blob_top_data_->height(), 3);
    EXPECT_EQ(this->blob_top_data_->width(), 4);
    EXPECT_EQ(this->blob_top_label_->num(), batch_size);
    EXPECT_EQ(this->blob_top_label_->channels(), 1);
    EXPECT_EQ(this->blob_top_label_->height(), 1);
    EXPECT_EQ(this->blob_top_label_->width(), 1);

    int clip_index = 0;
    const int pad_start = (batch_size / clip_length) * clip_length;
    for (int iter = 0; iter < 6; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      const int clip_start_index = clip_index;
      for (int i = 0; i < batch_size; ++i) {
        const Dtype expected_value =
            (i < pad_start) ? (clip_index % batch_size) : pad_value;
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i])
            << "debug: iter " << iter << " i " << i;
        if (i % clip_length == clip_length - 1) { ++clip_index; }
      }
      clip_index = clip_start_index;
      for (int i = 0; i < batch_size; ++i) {
        Dtype expected_value = (i < pad_start) ?
            (scale * (clip_index % batch_size + (i % clip_length) * 10)) :
            pad_value;
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j])
              << "debug: iter " << iter << " i " << i << " j " << j;
        }
        if (i % clip_length == clip_length - 1) { ++clip_index; }
      }
      for (int i = 0; i < batch_size; ++i) {
        const Dtype expected_value = (i < pad_start) ?
           ((i % clip_length) ? DataLayer<Dtype>::CLIP_CONTINUE :
                                DataLayer<Dtype>::CLIP_BEGIN) :
           DataLayer<Dtype>::PADDING;
        EXPECT_EQ(expected_value, this->blob_top_clip_markers_->cpu_data()[i])
            << "debug: iter " << iter << " i " << i;
      }
    }
  }

  void TestReadActuallyVariableLengthClips(int num_test_sample) {
    LayerParameter param;
    DataParameter* data_param = param.mutable_data_param();
    const int batch_size = 10;
    data_param->set_batch_size(batch_size);
    data_param->set_clip_mode(DataParameter_ClipMode_VARIABLE);
    data_param->set_max_train_item(num_test_sample);
    data_param->set_max_test_item(num_test_sample);
    const Dtype pad_value = 27281;
    data_param->set_clip_pad_value(pad_value);
    const Dtype scale = 1;
    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);
    data_param->set_source(this->filename_->c_str());
    DataLayer<Dtype> layer(param);
    this->blob_top_vec_.push_back(this->blob_top_clip_markers_);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(this->blob_top_data_->num(), 10);
    EXPECT_EQ(this->blob_top_data_->channels(), 2);
    EXPECT_EQ(this->blob_top_data_->height(), 3);
    EXPECT_EQ(this->blob_top_data_->width(), 4);
    EXPECT_EQ(this->blob_top_label_->num(), 10);
    EXPECT_EQ(this->blob_top_label_->channels(), 1);
    EXPECT_EQ(this->blob_top_label_->height(), 1);
    EXPECT_EQ(this->blob_top_label_->width(), 1);
    EXPECT_EQ(this->blob_top_clip_markers_->num(), 10);
    EXPECT_EQ(this->blob_top_clip_markers_->channels(), 1);
    EXPECT_EQ(this->blob_top_clip_markers_->height(), 1);
    EXPECT_EQ(this->blob_top_clip_markers_->width(), 1);

    Dtype expected_value;
    for (int iter = 0; iter < 10; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      expected_value = 5;
      for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i]);
        EXPECT_EQ((i == 0) ? DataLayer<Dtype>::CLIP_BEGIN :
                             DataLayer<Dtype>::CLIP_CONTINUE,
                  this->blob_top_clip_markers_->cpu_data()[i]);
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j]);
        }
      }
      expected_value = 8;
      for (int i = 4; i < 5; ++i) {
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i]);
        EXPECT_EQ((i == 4) ? DataLayer<Dtype>::CLIP_BEGIN :
                             DataLayer<Dtype>::CLIP_CONTINUE,
                  this->blob_top_clip_markers_->cpu_data()[i]);
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j]);
        }
      }
      expected_value = 4;
      for (int i = 5; i < 7; ++i) {
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i]);
        EXPECT_EQ((i == 5) ? DataLayer<Dtype>::CLIP_BEGIN :
                             DataLayer<Dtype>::CLIP_CONTINUE,
                  this->blob_top_clip_markers_->cpu_data()[i]);
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j]);
        }
      }
      expected_value = pad_value;
      for (int i = 7; i < 10; ++i) {
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i]);
        EXPECT_EQ(DataLayer<Dtype>::PADDING,
                  this->blob_top_clip_markers_->cpu_data()[i]);
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j]);
        }
      }

      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      expected_value = 1;
      for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i]);
        EXPECT_EQ((i == 0) ? DataLayer<Dtype>::CLIP_BEGIN :
                             DataLayer<Dtype>::CLIP_CONTINUE,
                  this->blob_top_clip_markers_->cpu_data()[i]);
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j]);
        }
      }
      expected_value = pad_value;
      for (int i = 9; i < 10; ++i) {
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i]);
        EXPECT_EQ(DataLayer<Dtype>::PADDING,
                  this->blob_top_clip_markers_->cpu_data()[i]);
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j]);
        }
      }

      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      expected_value = 10;
      for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i]);
        EXPECT_EQ((i == 0) ? DataLayer<Dtype>::CLIP_BEGIN :
                             DataLayer<Dtype>::CLIP_CONTINUE,
                  this->blob_top_clip_markers_->cpu_data()[i]);
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j]);
        }
      }
      expected_value = 7;
      for (int i = 4; i < 7; ++i) {
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i]);
        EXPECT_EQ((i == 4) ? DataLayer<Dtype>::CLIP_BEGIN :
                             DataLayer<Dtype>::CLIP_CONTINUE,
                  this->blob_top_clip_markers_->cpu_data()[i]);
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j]);
        }
      }
      expected_value = pad_value;
      for (int i = 7; i < 10; ++i) {
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i]);
        EXPECT_EQ(DataLayer<Dtype>::PADDING,
                  this->blob_top_clip_markers_->cpu_data()[i]);
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j]);
        }
      }
    }
  }

  void TestReadFixedLengthPaddedClips(const int clip_length,
                                      const int batch_size) {
    LayerParameter param;
    DataParameter* data_param = param.mutable_data_param();
    data_param->set_batch_size(batch_size);
    data_param->set_clip_length(clip_length);
    data_param->set_clip_mode(DataParameter_ClipMode_FIXED_LENGTH);
    data_param->set_clip_allow_pad(true);
    data_param->set_clip_pad_mode(DataParameter_ClipPadCropMode_END);
    data_param->set_max_train_item(batch_size);
    data_param->set_max_test_item(batch_size);
    const Dtype pad_value = 27281;
    data_param->set_clip_pad_value(pad_value);
    const Dtype scale = 3;
    TransformationParameter* transform_param =
        param.mutable_transform_param();
    transform_param->set_scale(scale);
    data_param->set_source(this->filename_->c_str());
    DataLayer<Dtype> layer(param);
    this->blob_top_vec_.push_back(this->blob_top_clip_markers_);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(this->blob_top_data_->num(), 6);
    EXPECT_EQ(this->blob_top_data_->channels(), 2);
    EXPECT_EQ(this->blob_top_data_->height(), 3);
    EXPECT_EQ(this->blob_top_data_->width(), 4);
    EXPECT_EQ(this->blob_top_label_->num(), 6);
    EXPECT_EQ(this->blob_top_label_->channels(), 1);
    EXPECT_EQ(this->blob_top_label_->height(), 1);
    EXPECT_EQ(this->blob_top_label_->width(), 1);
    EXPECT_EQ(this->blob_top_clip_markers_->num(), 6);
    EXPECT_EQ(this->blob_top_clip_markers_->channels(), 1);
    EXPECT_EQ(this->blob_top_clip_markers_->height(), 1);
    EXPECT_EQ(this->blob_top_clip_markers_->width(), 1);

    int clip_index = 0;
    for (int iter = 0; iter < 10; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      const int prepad_end = 0;
      const int postpad_start = 3;
      const int clip_start_index = clip_index;
      for (int i = 0; i < batch_size; ++i) {
        const Dtype expected_value = (clip_index % batch_size);
        EXPECT_EQ(expected_value, this->blob_top_label_->cpu_data()[i])
            << "debug: iter " << iter << " i " << i;
        if (i % clip_length == clip_length - 1) { ++clip_index; }
      }
      clip_index = clip_start_index;
      for (int i = 0; i < batch_size; ++i) {
        const int frame_id = i % clip_length;
        const Dtype expected_value =
            (frame_id > prepad_end && frame_id < postpad_start) ?
            (scale * (clip_index % batch_size + ((i - 1) % clip_length) * 10)) :
            pad_value;
        for (int j = 0; j < 24; ++j) {
          EXPECT_EQ(expected_value,
              this->blob_top_data_->cpu_data()[i * 24 + j])
              << "debug: iter " << iter << " i " << i << " j " << j;
        }
        if (i % clip_length == clip_length - 1) { ++clip_index; }
      }
      for (int i = 0; i < batch_size; ++i) {
        const int frame_id = i % clip_length;
        const typename DataLayer<Dtype>::ClipMarker& expected_value =
           (frame_id > prepad_end && frame_id < postpad_start) ?
           (((i - 1) % clip_length == 0) ? DataLayer<Dtype>::CLIP_BEGIN :
                                           DataLayer<Dtype>::CLIP_CONTINUE) :
           DataLayer<Dtype>::PADDING;
        EXPECT_EQ(expected_value, this->blob_top_clip_markers_->cpu_data()[i])
            << "debug: iter " << iter << " i " << i;
      }
    }
  }

  void LevelDBAppendVariableLengthClip(const int id, int clip_length, int value,
      leveldb::DB* db) {
    for (int frame_id = 0; frame_id < clip_length; ++frame_id) {
      Datum datum;
      char key_cstr[17];
      datum.set_label(value);  
      datum.set_frames(clip_length);
      datum.set_channels(2);
      datum.set_height(3);
      datum.set_width(4);
      datum.set_current_frame(frame_id);
      std::string* data = datum.mutable_data();
      for (int j = 0; j < 24; ++j) {
        uint8_t item = static_cast<uint8_t>(value);
        data->push_back(item);
      }
      int n = sprintf(key_cstr, "%08d%08d",id,frame_id);
      db->Put(leveldb::WriteOptions(), string(key_cstr), datum.SerializeAsString());
    }
  }

  void FillLevelDBVariableLengthClips() {
    LOG(INFO) << "Using temporary leveldb " << *filename_;
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    leveldb::Status status =
        leveldb::DB::Open(options, filename_->c_str(), &db);
    CHECK(status.ok());
    int id = 0;
    LevelDBAppendVariableLengthClip(id++, 4, 5, db);
    LevelDBAppendVariableLengthClip(id++, 1, 8, db);
    LevelDBAppendVariableLengthClip(id++, 2, 4, db);
    LevelDBAppendVariableLengthClip(id++, 9, 1, db);
    LevelDBAppendVariableLengthClip(id++, 4, 10, db);
    LevelDBAppendVariableLengthClip(id++, 3, 7, db);
    delete db;
  }

  virtual ~DataLayerTest() { delete blob_top_data_; delete blob_top_label_; }

  DataParameter_DB backend_;
  shared_ptr<string> filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  Blob<Dtype>* const blob_top_clip_markers_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
};

TYPED_TEST_CASE(DataLayerTest, TestDtypesAndDevices);

TYPED_TEST(DataLayerTest, TestReadLevelDB) {
  const bool unique_pixels = false;  // all pixels the same; images different
  this->FillLevelDB(unique_pixels);
  this->TestRead();
}

TYPED_TEST(DataLayerTest, TestReadCropTrainLevelDB) {
  Caffe::set_phase(Caffe::TRAIN);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  this->TestReadCrop();
}

TYPED_TEST(DataLayerTest, TestReadFixedLengthClipsFrameMajor) {
  const bool unique_pixels = false;  // all pixels the same; images different
  const int clip_length = 3;
  const int batch_size = 6;
  this->FillLevelDB(unique_pixels, clip_length, batch_size);
  this->TestReadFixedLengthClipsFrameMajor(clip_length, batch_size);
}

TYPED_TEST(DataLayerTest, TestReadFixedLengthClipsFrameMajorSubSample) {
  const bool unique_pixels = false;  // all pixels the same; images different
  const int batch_size = 6;
  const int sub_sample = 2;
  int clip_length = 5;
  this->FillLevelDB(unique_pixels, clip_length, batch_size);
  clip_length = 3;
  this->TestReadFixedLengthClipsFrameMajor(clip_length, batch_size,sub_sample);
}

TYPED_TEST(DataLayerTest, TestReadFixedLengthClipsCollapsedLabels) {
  const bool unique_pixels = false;  // all pixels the same; images different
  const int clip_length = 3;
  const int batch_size = 6;
  this->FillLevelDB(unique_pixels, clip_length, batch_size);
  this->TestReadFixedLengthClipsCollapsedLabels(clip_length, batch_size);
}

TYPED_TEST(DataLayerTest, TestReadFixedLengthClipsCollapsedLabelsSubSample) {
  const bool unique_pixels = false;  // all pixels the same; images different
  int clip_length = 5;
  const int batch_size = 6;
  this->FillLevelDB(unique_pixels, clip_length, batch_size);
  clip_length = 3;
  int sub_sample = 2;
  this->TestReadFixedLengthClipsCollapsedLabels(clip_length, batch_size, sub_sample);
}

TYPED_TEST(DataLayerTest, TestReadFixedLengthClips) {
  Caffe::set_phase(Caffe::TRAIN);
  const bool unique_pixels = false;  // all pixels the same; images different
  const int clip_length = 3;
  const int batch_size = 5;
  this->FillLevelDB(unique_pixels, clip_length, batch_size);
  this->TestReadFixedLengthClips(clip_length, batch_size);
}

TYPED_TEST(DataLayerTest, TestReadActuallyVariableLengthClips) {
  this->FillLevelDBVariableLengthClips();
  const int sub_sample = 6;
  this->TestReadActuallyVariableLengthClips(sub_sample);
}

TYPED_TEST(DataLayerTest, TestReadFixedLengthPaddedClips) {
  Caffe::set_phase(Caffe::TRAIN);
  const bool unique_pixels = false;  // all pixels the same; images different
  const int clip_length = 3;
  const int input_clip_length = 2;
  const int batch_size = 6;
  this->FillLevelDB(unique_pixels, input_clip_length, batch_size);
  this->TestReadFixedLengthPaddedClips(clip_length, batch_size);
}

// Test that the sequence of random crops is consistent when using
// Caffe::set_random_seed.
TYPED_TEST(DataLayerTest, TestReadCropTrainSequenceSeededLevelDB) {
  Caffe::set_phase(Caffe::TRAIN);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  this->TestReadCropTrainSequenceSeeded();
}

// Test that the sequence of random crops differs across iterations when
// Caffe::set_random_seed isn't called (and seeds from srand are ignored).
TYPED_TEST(DataLayerTest, TestReadCropTrainSequenceUnseededLevelDB) {
  Caffe::set_phase(Caffe::TRAIN);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  this->TestReadCropTrainSequenceUnseeded();
}

TYPED_TEST(DataLayerTest, TestReadCropTestLevelDB) {
  Caffe::set_phase(Caffe::TEST);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLevelDB(unique_pixels);
  this->TestReadCrop();
}

TYPED_TEST(DataLayerTest, TestReadLMDB) {
  const bool unique_pixels = false;  // all pixels the same; images different
  this->FillLMDB(unique_pixels);
  this->TestRead();
}

TYPED_TEST(DataLayerTest, TestReadCropTrainLMDB) {
  Caffe::set_phase(Caffe::TRAIN);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLMDB(unique_pixels);
  this->TestReadCrop();
}

// Test that the sequence of random crops is consistent when using
// Caffe::set_random_seed.
TYPED_TEST(DataLayerTest, TestReadCropTrainSequenceSeededLMDB) {
  Caffe::set_phase(Caffe::TRAIN);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLMDB(unique_pixels);
  this->TestReadCropTrainSequenceSeeded();
}

// Test that the sequence of random crops differs across iterations when
// Caffe::set_random_seed isn't called (and seeds from srand are ignored).
TYPED_TEST(DataLayerTest, TestReadCropTrainSequenceUnseededLMDB) {
  Caffe::set_phase(Caffe::TRAIN);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLMDB(unique_pixels);
  this->TestReadCropTrainSequenceUnseeded();
}

TYPED_TEST(DataLayerTest, TestReadCropTestLMDB) {
  Caffe::set_phase(Caffe::TEST);
  const bool unique_pixels = true;  // all images the same; pixels different
  this->FillLMDB(unique_pixels);
  this->TestReadCrop();
}

}  // namespace caffe
