#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DiagInnerProductLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  dim_ = bottom[0]->count() / bottom[0]->num();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, dim_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.diag_inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  weight_diff_buffer_.ReshapeLike(*this->blobs_[0]);
}

template <typename Dtype>
void DiagInnerProductLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void DiagInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    caffe_mul(dim_, weight, bottom_data + i * dim_, top_data + i * dim_);
  }
}

template <typename Dtype>
void DiagInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* weight_diff_buffer = weight_diff_buffer_.mutable_cpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int i = 0; i < top[0]->num(); ++i) {
      caffe_mul(dim_, bottom_data + i * dim_,
                top_diff + i * dim_, weight_diff_buffer);
      caffe_axpy(dim_, Dtype(1), weight_diff_buffer, weight_diff);
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < top[0]->num(); ++i) {
      caffe_mul(dim_, weight, top_diff + i * dim_, bottom_diff + i * dim_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DiagInnerProductLayer);
#endif

INSTANTIATE_CLASS(DiagInnerProductLayer);

}  // namespace caffe
