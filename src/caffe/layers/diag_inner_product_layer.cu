#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DiagInnerProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    caffe_gpu_mul(dim_, weight, bottom_data + i * dim_, top_data + i * dim_);
  }
}

template <typename Dtype>
void DiagInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_diff_buffer = weight_diff_buffer_.mutable_gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    for (int i = 0; i < top[0]->num(); ++i) {
      caffe_gpu_mul(dim_, bottom_data + i * dim_,
                top_diff + i * dim_, weight_diff_buffer);
      caffe_gpu_axpy(dim_, Dtype(1), weight_diff_buffer, weight_diff);
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    for (int i = 0; i < top[0]->num(); ++i) {
      caffe_gpu_mul(dim_, weight, top_diff + i * dim_, bottom_diff + i * dim_);
    }
  }
}

#ifdef gpu_ONLY
STUB_GPU(DiagInnerProductLayer);
#endif

INSTANTIATE_CLASS(DiagInnerProductLayer);

}  // namespace caffe
