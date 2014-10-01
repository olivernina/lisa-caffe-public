#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class LSTMLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LSTMLayerTest() : hidden_dim_(4) {
    blob_bottom_.Reshape(3, 4, 3, 2);
    blob_bottom_flush_.Reshape(3, 1, 1, 1);
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_bottom_);
    blob_bottom_vec_.push_back(&blob_bottom_);
    blob_bottom_vec_.push_back(&blob_bottom_flush_);
    blob_top_vec_.push_back(&blob_top_);
    layer_param_.mutable_lstm_param()->set_hidden_dim(hidden_dim_);
    FillerParameter* weight_filler =
        layer_param_.mutable_lstm_param()->mutable_weight_filler();
    weight_filler->set_type("gaussian");
    weight_filler->set_std(0.01);
  }

  LayerParameter layer_param_;
  int hidden_dim_;
  Blob<Dtype> blob_bottom_;
  Blob<Dtype> blob_bottom_flush_;
  Blob<Dtype> blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LSTMLayerTest, TestDtypesAndDevices);

TYPED_TEST(LSTMLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LSTMLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_.num(), this->blob_bottom_.num());
  EXPECT_EQ(this->blob_top_.channels(), this->hidden_dim_);
  EXPECT_EQ(this->blob_top_.height(), 1);
  EXPECT_EQ(this->blob_top_.width(), 1);
}

TYPED_TEST(LSTMLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LSTMLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(LSTMLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LSTMLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(LSTMLayerTest, TestGradientNonZeroFlush) {
  typedef typename TypeParam::Dtype Dtype;
  LSTMLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  for (int i = 0; i < this->blob_bottom_flush_.num(); ++i) {
    this->blob_bottom_flush_.mutable_cpu_data()[i] = 1;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(LSTMLayerTest, TestGradientNonZeroFlushBufferSize2) {
  typedef typename TypeParam::Dtype Dtype;
  this->layer_param_.mutable_lstm_param()->set_buffer_size(2);
  this->blob_bottom_.Reshape(4, 4, 3, 2);
  this->blob_bottom_flush_.Reshape(4, 1, 1, 1);
  // fill the values
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&this->blob_bottom_);
  LSTMLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  for (int i = 0; i < this->blob_bottom_flush_.count(); ++i) {
    this->blob_bottom_flush_.mutable_cpu_data()[i] = 1;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
