#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxInstanceLoss(const int nthreads,
    const int num, const int channels, const int spatial_dim,
    const Dtype* prob_data, const Dtype* label_data, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    const int label = static_cast<int>(label_data[index]);
    const Dtype prob = prob_data[(n * channels + label) * spatial_dim + s];
    loss[index] = -log(max(prob, Dtype(FLT_MIN)));
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  // Hack: store loss per instance in loss_multiplier_ diff.
  Dtype* instance_loss = loss_multiplier_.mutable_gpu_diff();
  const int kernel_count = bottom[1]->count();
  SoftmaxInstanceLoss<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(kernel_count), CAFFE_CUDA_NUM_THREADS>>>(
      kernel_count, num, channels, spatial_dim, prob_data, label,
      instance_loss);
  const Dtype* weights = NULL;
  if (bottom.size() > 2) {
    weights = bottom[2]->gpu_data();
    DCHECK_EQ(kernel_count, bottom[2]->count());
  } else {
    weights = loss_multiplier_.gpu_data();
    DCHECK_EQ(kernel_count, loss_multiplier_.count());
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  caffe_gpu_dot(kernel_count, instance_loss, weights, loss);
  *loss /= kernel_count;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackward(const int nthreads,
    const int num, const int channels, const int spatial_dim,
    const Dtype loss_weight, const Dtype* prob_data, const Dtype* label_data,
    const Dtype* weight_data, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels / spatial_dim;
    int c = index / spatial_dim % channels;
    int s = index % spatial_dim;
    const int label_index = n * spatial_dim + s;
    const int label = static_cast<int>(label_data[label_index]);
    const Dtype weight = weight_data ? weight_data[label_index] : Dtype(1);
    bottom_diff[index] = prob_data[index] - (c == label);
    bottom_diff[index] *= loss_weight * weight / (num * spatial_dim);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (!propagate_down[0]) { return; }
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  const int kernel_count = prob_.count();
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* weights = (bottom.size() > 2) ? bottom[2]->gpu_data(): NULL;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  SoftmaxLossBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(kernel_count), CAFFE_CUDA_NUM_THREADS>>>(
      kernel_count, num, channels, spatial_dim, loss_weight,
      prob_data, label, weights, bottom_diff);
}

INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
