#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LSTMUnitForward(const int nthreads, const int dim,
    const Dtype* C_prev, const Dtype* X, Dtype* C, Dtype* H) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;
    const Dtype* X_offset = X + 4 * dim * n;
    const Dtype i = X_offset[d];
    const Dtype f = X_offset[1 * dim + d];
    const Dtype o = X_offset[2 * dim + d];
    const Dtype g_sigmoid = X_offset[3 * dim + d];
    const Dtype g = Dtype(2) * g_sigmoid - 1;
    const Dtype c_prev = C_prev[index];
    const Dtype c = f * c_prev + i * g;
    C[index] = c;
    const Dtype tanh_c =
        Dtype(2) / (Dtype(1) + exp(-2 * c)) - Dtype(1);
    H[index] = o * tanh_c;
  }
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = top[1]->count();
  const Dtype* C_prev = bottom[0]->gpu_data();
  const Dtype* X = bottom[1]->gpu_data();
  Dtype* C = top[0]->mutable_gpu_data();
  Dtype* H = top[1]->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LSTMUnitForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, hidden_dim_, C_prev, X, C, H);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void LSTMUnitBackward(const int nthreads, const int dim,
    const Dtype* C_prev, const Dtype* X, const Dtype* C, const Dtype* H,
    const Dtype* C_diff, const Dtype* H_diff,
    Dtype* C_prev_diff, Dtype* X_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;
    const Dtype* X_offset = X + 4 * dim * n;
    const Dtype i = X_offset[d];
    const Dtype f = X_offset[1 * dim + d];
    const Dtype o = X_offset[2 * dim + d];
    const Dtype g_sigmoid = X_offset[3 * dim + d];
    const Dtype g = Dtype(2) * g_sigmoid - 1;
    const Dtype c_prev = C_prev[index];
    const Dtype c = C[index];
    const Dtype tanh_c =
        Dtype(2) / (Dtype(1) + exp(-2 * c)) - Dtype(1);
    Dtype* c_prev_diff = C_prev_diff + index;
    Dtype* X_diff_offset = X_diff + 4 * dim * n;
    Dtype* i_diff = X_diff_offset + d;
    Dtype* f_diff = X_diff_offset + 1 * dim + d;
    Dtype* o_diff = X_diff_offset + 2 * dim + d;
    Dtype* g_diff = X_diff_offset + 3 * dim + d;
    const Dtype c_term_diff =
        C_diff[index] + H_diff[index] * o * (1 - tanh_c * tanh_c);
    *c_prev_diff = c_term_diff * f;
    *i_diff = c_term_diff * g;
    *f_diff = c_term_diff * c_prev;
    *o_diff = H_diff[index] * tanh_c;
    *g_diff = 2 * c_term_diff * i;
  }
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] && !propagate_down[1]) { return; }
  const int count = top[1]->count();
  const Dtype* C_prev = bottom[0]->gpu_data();
  const Dtype* X = bottom[1]->gpu_data();
  const Dtype* C = top[0]->gpu_data();
  const Dtype* H = top[1]->gpu_data();
  const Dtype* C_diff = top[0]->gpu_diff();
  const Dtype* H_diff = top[1]->gpu_diff();
  Dtype* C_prev_diff = bottom[0]->mutable_gpu_diff();
  Dtype* X_diff = bottom[1]->mutable_gpu_diff();
  LSTMUnitBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, hidden_dim_, C_prev, X, C, H, C_diff, H_diff, C_prev_diff, X_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_CLASS(LSTMUnitLayer);


}  // namespace caffe
