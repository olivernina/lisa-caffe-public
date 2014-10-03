#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//  // First, join the thread
//  JoinPrefetchThread();
//  // Copy the data
//  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
//      top[0]->mutable_gpu_data());
//  if (this->output_labels_) {
//    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
//        top[1]->mutable_gpu_data());
//  }
  // Start a new prefetch thread
//  CreatePrefetchThread();
  switch (this->layer_param_.data_param().clip_order()) {
  case DataParameter_ClipOrder_CLIP_MAJOR:
    // First, join the thread
    JoinPrefetchThread();
    // Copy the data
    caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
        top[0]->mutable_gpu_data());
    if (this->output_labels_) {
      caffe_copy(prefetch_label_.num(), prefetch_label_.cpu_data(),
          top[1]->mutable_gpu_data());
    }
    if (this->output_clip_markers_) {
      caffe_copy(prefetch_clip_markers_.count(),
          prefetch_clip_markers_.cpu_data(), top[2]->mutable_gpu_data());
    }
    // Start a new prefetch thread
    CreatePrefetchThread();
    break;
  case DataParameter_ClipOrder_FRAME_MAJOR:
    // TODO(jdonahue): GPU implementation.
    Forward_cpu(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown clip order: " << this->layer_param_.data_param().clip_order();
  }
}

INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
