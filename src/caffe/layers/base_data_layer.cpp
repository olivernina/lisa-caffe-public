#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  data_transformer_.InitRand();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Copy the data

  int dim;
  int num_clips;
  const Dtype* prefetch_data = prefetch_data_.cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_label = NULL;
  const Dtype* prefetch_label = NULL;
  if (this->output_labels_) {
    prefetch_label = prefetch_label_.cpu_data();
    top_label = top[1]->mutable_cpu_data();
  }
  Dtype* top_clip_markers = NULL;
  const Dtype* prefetch_clip_markers = NULL;
  if (this->output_clip_markers_) {
    prefetch_clip_markers = prefetch_clip_markers_.cpu_data();
    top_clip_markers = top[2]->mutable_cpu_data();
  }
  
  Dtype* top_weight_loss = NULL;
  const Dtype* prefetch_weight_loss = NULL;
  if (this->layer_param_.data_param().weight_loss()) {
    prefetch_weight_loss = prefetch_weight_loss_.cpu_data();
    top_weight_loss = top[3]->mutable_cpu_data();
  }
  
  int length_row;
  int num_rows;
  int clip_length;

  switch(this->layer_param_.data_param().clip_order()) {
  case DataParameter_ClipOrder_CLIP_MAJOR:
    caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
               top[0]->mutable_cpu_data());
    if (this->output_labels_) {
      caffe_copy(prefetch_label_.num(), prefetch_label_.cpu_data(),
                 top[1]->mutable_cpu_data());
    }
    if (this->output_clip_markers_) {
      //caffe_copy(prefetch_clip_markers_.count(), prefetch_label_.cpu_data(),
      //           top[2]->mutable_cpu_data());
      caffe_copy(prefetch_data_.num(), prefetch_clip_markers_.cpu_data(),
	         top[2]->mutable_cpu_data());
    }
    break;
  case DataParameter_ClipOrder_FRAME_MAJOR:
    dim = prefetch_data_.count() / prefetch_data_.num();
    CHECK_GT(this->layer_param_.data_param().clip_length(), 0);
    num_clips = prefetch_data_.num() / this->layer_param_.data_param().clip_length();
//    caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
//               top[0]->mutable_cpu_data());
    if (this->layer_param_.data_param().clip_mode() == DataParameter_ClipMode_LSTM) {
      length_row = this->tLSTM_;
      num_rows = this->sLSTM_;
      clip_length = this->layer_param_.data_param().lstm_clip_length();
    } else {
      length_row = this->layer_param_.data_param().clip_length();
      num_rows = num_clips;
      clip_length = this->layer_param_.data_param().clip_length();
    }
    for (int frame_id = 0; frame_id < length_row; ++frame_id) {
      for (int clip_id = 0; clip_id < num_rows; ++clip_id) {
        // prefetch_id: the usual clip-major index.
        const int prefetch_id = clip_id * length_row + frame_id;
        // top_id: the frame-major index.
        const int top_id = frame_id * num_rows + clip_id;
        caffe_copy(dim, &prefetch_data[prefetch_id * dim],
                   &top_data[top_id * dim]);
        if (this->output_labels_) {
        if (this->layer_param_.data_param().clip_collapse_labels() && frame_id == 0) {
            top_label[top_id] = prefetch_label[top_id];
          } else if (!this->layer_param_.data_param().clip_collapse_labels()) {
            top_label[top_id] = prefetch_label[prefetch_id];
          }
        }
        if (this->output_clip_markers_) {
          top_clip_markers[top_id] = prefetch_clip_markers[prefetch_id];
        }
        if (this->layer_param_.data_param().weight_loss()) {
          top_weight_loss[top_id] =  prefetch_weight_loss[prefetch_id];
        }
      }
    }
    break;
//    if (this->output_labels_) {
//      caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
//                 top[1]->mutable_cpu_data());
//    }
//    if (this->output_clip_markers_) {
//      caffe_copy(prefetch_clip_markers_.count(), prefetch_label_.cpu_data(),
//                 top[2]->mutable_cpu_data());
//    }
  default:
    LOG(FATAL) << "Unknown clip order: " << this->layer_param_.data_param().clip_order();

  }


  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
