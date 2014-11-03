#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReductionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data;
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case ReductionParameter_ReductionOp_SUM:
    bottom_data = bottom[0]->gpu_data();
    caffe_gpu_dot(count, sum_multiplier_.gpu_data(), bottom_data, top_data);
    break;
  case ReductionParameter_ReductionOp_ASUM:
    *top_data = bottom[0]->asum_data();
    break;
  case ReductionParameter_ReductionOp_SUM_OF_SQUARES:
    *top_data = bottom[0]->sumsq_data();
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
  }
  *top_data *= coeff_;
}

template <typename Dtype>
void ReductionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  const Dtype top_diff = top[0]->cpu_diff()[0];
  const Dtype bottom_coeff = top_diff * coeff_;
  const Dtype* bottom_data;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  switch (op_) {
  case ReductionParameter_ReductionOp_SUM:
    caffe_gpu_set(count, bottom_coeff, bottom_diff);
    break;
  case ReductionParameter_ReductionOp_ASUM:
    bottom_data = bottom[0]->gpu_data();
    caffe_gpu_sign(count, bottom_data, bottom_diff);
    caffe_gpu_scal(count, bottom_coeff, bottom_diff);
    break;
  case ReductionParameter_ReductionOp_SUM_OF_SQUARES:
    bottom_data = bottom[0]->gpu_data();
    caffe_gpu_scale(count, 2 * bottom_coeff, bottom_data, bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
  }
}

INSTANTIATE_CLASS(ReductionLayer);

}  // namespace caffe
