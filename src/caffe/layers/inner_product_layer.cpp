#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  N_ = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  index_input_dim_ = this->layer_param_.inner_product_param().index_input_dim();
  if (index_input_dim_) {
    K_ = index_input_dim_;
  } else {
    K_ = bottom[0]->count() / bottom[0]->num();
  }
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    if (index_input_dim_) {
      // Use transposed weights for one-hot inputs so that the column copy is
      // direct, instead of strided over K_.
      this->blobs_[0].reset(new Blob<Dtype>(1, 1, K_, N_));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
    }
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  backward_update_lr_ =
      this->layer_param_.inner_product_param().backward_update_lr();
  if (backward_update_lr_ != Dtype(0)) {
    CHECK(this->layer_param_.blobs_lr_size() == 0 ||
          this->layer_param_.blobs_lr(0) == 0)
        << "backward_update_lr only supported with blobs_lr(0) == 0";
    CHECK_NE(index_input_dim_, 0)
        << "backward_update_lr only supported for one-hot inputs.";
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  M_ = bottom[0]->num();
  if (index_input_dim_) {
    CHECK_EQ(bottom[0]->count(), bottom[0]->num())
        << "Indexed inputs must be 1 dimensional.";
  } else {
    CHECK_EQ(bottom[0]->count() / bottom[0]->num(), K_) << "Input size "
      "incompatible with inner product parameters.";
  }
  top[0]->Reshape(bottom[0]->num(), N_, 1, 1);
  // Set up the bias multiplier
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, M_);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (index_input_dim_) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < bottom[0]->num(); ++n) {
      const int index = static_cast<int>(bottom_data[n]);
      DCHECK_EQ(static_cast<Dtype>(index), bottom_data[n])
          << "index_input_dim_ used with non-integer inputs.";
      const Dtype* weight_offset = weight + index * N_;
      Dtype* top_data_offset = top_data + n * N_;
      caffe_copy(N_, weight_offset, top_data_offset);
    }
  } else {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
        bottom_data, weight, (Dtype)0., top_data);
  }
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (index_input_dim_) {
      Dtype* weight_diff = (backward_update_lr_ == Dtype(0)) ?
          this->blobs_[0]->mutable_cpu_diff() :
          this->blobs_[0]->mutable_cpu_data();
      const Dtype alpha = (backward_update_lr_ == Dtype(0)) ?
          Dtype(1) : -backward_update_lr_;
      for (int n = 0; n < bottom[0]->num(); ++n) {
        const int index = static_cast<int>(bottom_data[n]);
        DCHECK_EQ(static_cast<Dtype>(index), bottom_data[n])
            << "index_input_dim_ used with non-integer inputs.";
        const Dtype* top_diff_offset = top_diff + n * N_;
        Dtype* weight_diff_offset = weight_diff + index * N_;
        caffe_axpy(N_, alpha, top_diff_offset, weight_diff_offset);
      }
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
          top_diff, bottom_data, Dtype(1),
          this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), Dtype(1),
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    CHECK_EQ(0, index_input_dim_) << "Can't propagate down to indexed inputs.";
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);

}  // namespace caffe
