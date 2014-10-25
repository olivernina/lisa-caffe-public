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
  LSTMLayerTest() : hidden_dim_(3) {
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
    weight_filler->set_std(0.2);
    FillerParameter* bias_filler =
        layer_param_.mutable_lstm_param()->mutable_bias_filler();
    bias_filler->set_type("gaussian");
    bias_filler->set_std(0.1);
  }

  void TestForward(const bool use_diag_cell_gates) {
    const int kNumTimesteps = 3;
    const int num = this->blob_bottom_.num();

    vector<Blob<Dtype>*> full_bottom_vec;
    Blob<Dtype> input_sequence(kNumTimesteps * num, 4, 3, 2);
    Blob<Dtype> flush(kNumTimesteps * num, 1, 1, 1);
    full_bottom_vec.push_back(&input_sequence);
    full_bottom_vec.push_back(&flush);

    vector<Blob<Dtype>*> full_top_vec;
    Blob<Dtype> output_sequence;
    full_top_vec.push_back(&output_sequence);

    // Fill the flush blob with <0, 1, 1, ..., 1>,
    // indicating a sequence that begins at the first timestep
    // then continues for the rest of the sequence.
    FillerParameter filler_param;
    filler_param.set_value(1);
    ConstantFiller<Dtype> one_filler(filler_param);
    one_filler.Fill(&flush);
    for (int n = 0; n < num; ++n) {
      flush.mutable_cpu_data()[n] = 0;
    }

    // Process the full sequence in a single batch.
    filler_param.set_mean(0);
    filler_param.set_std(1);
    GaussianFiller<Dtype> sequence_filler(filler_param);
    sequence_filler.Fill(&input_sequence);
    this->layer_param_.mutable_lstm_param()->set_buffer_size(num);
    this->layer_param_.mutable_lstm_param()->
        set_diagonal_cell_gates(use_diag_cell_gates);
    Caffe::set_random_seed(1701);
    shared_ptr<LSTMLayer<Dtype> > layer(
        new LSTMLayer<Dtype>(this->layer_param_));
    layer->SetUp(full_bottom_vec, full_top_vec);
    LOG(INFO) << "Calling forward for full sequence LSTM";
    layer->Forward(full_bottom_vec, full_top_vec);

    // Process the batch one timestep at a time;
    // check that we get the same result.
    Caffe::set_random_seed(1701);
    layer.reset(new LSTMLayer<Dtype>(this->layer_param_));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const int bottom_count = this->blob_bottom_.count();
    const int top_count = this->blob_top_.count();
    const Dtype kEpsilon = 1e-5;
    for (int t = 0; t < kNumTimesteps; ++t) {
      caffe_copy(bottom_count,
                 input_sequence.cpu_data() + t * bottom_count,
                 this->blob_bottom_.mutable_cpu_data());
      for (int n = 0; n < num; ++n) {
        this->blob_bottom_flush_.mutable_cpu_data()[n] = t > 0;
      }
      LOG(INFO) << "Calling forward for LSTM timestep " << t;
      layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      for (int i = 0; i < bottom_count; ++i) {
        ASSERT_LT(t * bottom_count + i, input_sequence.count());
        EXPECT_EQ(this->blob_bottom_.cpu_data()[i],
                  input_sequence.cpu_data()[t * bottom_count + i])
           << "t = " << t << "; i = " << i;
      }
      for (int i = 0; i < top_count; ++i) {
        ASSERT_LT(t * top_count + i, output_sequence.count());
        EXPECT_NEAR(this->blob_top_.cpu_data()[i],
                    output_sequence.cpu_data()[t * top_count + i], kEpsilon)
           << "t = " << t << "; i = " << i;
      }
    }

    // Process the batch one timestep at a time with all flush blobs set to 0.
    // Check that we get a different result, except in the first timestep.
    Caffe::set_random_seed(1701);
    layer.reset(new LSTMLayer<Dtype>(this->layer_param_));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int t = 0; t < kNumTimesteps; ++t) {
      caffe_copy(bottom_count,
                 input_sequence.cpu_data() + t * bottom_count,
                 this->blob_bottom_.mutable_cpu_data());
      for (int n = 0; n < num; ++n) {
        this->blob_bottom_flush_.mutable_cpu_data()[n] = 0;
      }
      LOG(INFO) << "Calling forward for LSTM timestep " << t;
      layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      for (int i = 0; i < bottom_count; ++i) {
        ASSERT_LT(t * bottom_count + i, input_sequence.count());
        EXPECT_EQ(this->blob_bottom_.cpu_data()[i],
                  input_sequence.cpu_data()[t * bottom_count + i])
           << "t = " << t << "; i = " << i;
      }
      for (int i = 0; i < top_count; ++i) {
        ASSERT_LT(t * top_count + i, output_sequence.count());
        if (t == 0) {
          EXPECT_NEAR(this->blob_top_.cpu_data()[i],
                      output_sequence.cpu_data()[t * top_count + i], kEpsilon)
             << "t = " << t << "; i = " << i;
        } else {
          EXPECT_NE(this->blob_top_.cpu_data()[i],
                    output_sequence.cpu_data()[t * top_count + i])
             << "t = " << t << "; i = " << i;
        }
      }
    }
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
  const bool kUseDiagCellGates = false;
  this->TestForward(kUseDiagCellGates);
}

TYPED_TEST(LSTMLayerTest, TestForwardDiagCellGates) {
  const bool kUseDiagCellGates = true;
  this->TestForward(kUseDiagCellGates);
}

TYPED_TEST(LSTMLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LSTMLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(LSTMLayerTest, TestGradientNonZeroFlush) {
  Caffe::set_phase(Caffe::TEST);
  typedef typename TypeParam::Dtype Dtype;
  LSTMLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  for (int i = 0; i < this->blob_bottom_flush_.count(); ++i) {
    this->blob_bottom_flush_.mutable_cpu_data()[i] =
        i >= this->layer_param_.lstm_param().buffer_size();
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(LSTMLayerTest, TestGradientNonZeroFlushBufferSize2) {
  Caffe::set_phase(Caffe::TEST);
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
    this->blob_bottom_flush_.mutable_cpu_data()[i] =
        i >= this->layer_param_.lstm_param().buffer_size();
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(LSTMLayerTest, TestGradientNonZeroFlushBufferSize2DiagCellGates) {
  Caffe::set_phase(Caffe::TEST);
  typedef typename TypeParam::Dtype Dtype;
  this->layer_param_.mutable_lstm_param()->set_buffer_size(2);
  this->layer_param_.mutable_lstm_param()->set_diagonal_cell_gates(true);
  this->blob_bottom_.Reshape(4, 4, 3, 2);
  this->blob_bottom_flush_.Reshape(4, 1, 1, 1);
  // fill the values
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&this->blob_bottom_);
  LSTMLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  for (int i = 0; i < this->blob_bottom_flush_.count(); ++i) {
    this->blob_bottom_flush_.mutable_cpu_data()[i] =
        i >= this->layer_param_.lstm_param().buffer_size();
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
