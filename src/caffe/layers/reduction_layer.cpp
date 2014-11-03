#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReductionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  op_ = this->layer_param_.reduction_param().operation();
  coeff_ = this->layer_param().reduction_param().coeff();
}

template <typename Dtype>
void ReductionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(1, 1, 1, 1);
  if (op_ == ReductionParameter_ReductionOp_SUM) {
    sum_multiplier_.ReshapeLike(*bottom[0]);
    caffe_set(bottom[0]->count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ReductionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data;
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case ReductionParameter_ReductionOp_SUM:
    bottom_data = bottom[0]->cpu_data();
    *top_data = caffe_cpu_dot(count, sum_multiplier_.cpu_data(), bottom_data);
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
void ReductionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  const Dtype top_diff = top[0]->cpu_diff()[0];
  const Dtype bottom_coeff = top_diff * coeff_;
  const Dtype* bottom_data;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  switch (op_) {
  case ReductionParameter_ReductionOp_SUM:
    caffe_set(count, bottom_coeff, bottom_diff);
    break;
  case ReductionParameter_ReductionOp_ASUM:
    bottom_data = bottom[0]->cpu_data();
    caffe_cpu_sign(count, bottom_data, bottom_diff);
    caffe_scal(count, bottom_coeff, bottom_diff);
    break;
  case ReductionParameter_ReductionOp_SUM_OF_SQUARES:
    bottom_data = bottom[0]->cpu_data();
    caffe_cpu_scale(count, 2 * bottom_coeff, bottom_data, bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ReductionLayer);
#endif

INSTANTIATE_CLASS(ReductionLayer);

}  // namespace caffe
