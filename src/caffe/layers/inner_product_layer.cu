#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void IndexInputForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* weight, const int num, const int dim, const int input_dim,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;
    const int one_hot_index = static_cast<int>(bottom_data[n]);
    const int weight_index = d * input_dim + one_hot_index;
    top_data[index] = weight[weight_index];
  }
}

template <typename Dtype>
__global__ void IndexInputBackward(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_diff, const int num, const int dim, const int input_dim,
    Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int output_index = index / input_dim;
    const int one_hot_index = index % input_dim;
    for (int n = 0; n < num; ++n) {
      if (bottom_data[n] == one_hot_index) {
        weight_diff[index] += top_diff[n * dim + output_index];
      }
    }
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (index_input_dim_) {
    const int count = top[0]->count();
    const int num = top[0]->num();
    IndexInputForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, weight, num, N_, index_input_dim_, top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
        bottom_data, weight, (Dtype)0., top_data);
  }
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(),
        this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    // Gradient with respect to weight
    if (index_input_dim_) {
      const int count = this->blobs_[0]->count();
      const int num = top[0]->num();
      IndexInputBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, top_diff, num, N_, index_input_dim_, weight_diff);
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
          top_diff, bottom_data, Dtype(1), weight_diff);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, Dtype(1), top_diff,
        bias_multiplier_.gpu_data(), Dtype(1),
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->gpu_data(), Dtype(0),
        bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_CLASS(InnerProductLayer);

}  // namespace caffe
