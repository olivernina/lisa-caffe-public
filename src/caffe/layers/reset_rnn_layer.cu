#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ResetRNNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
  if (Caffe::phase() == Caffe::TEST) {
    rnn_->ShareWeightData();
  }

  const int timestep_dim = buffer_size_ * hidden_dim_;

  DCHECK_EQ(timestep_dim, h_input_blob_->count());
  DCHECK_EQ(timestep_dim, h_output_blob_->count());
  const Dtype* hidden_output_data = h_output_blob_->gpu_data();
  Dtype* hidden_input_data = h_input_blob_->mutable_gpu_data();
  caffe_copy(timestep_dim, hidden_output_data, hidden_input_data);

  // Run the LSTM in forward mode.
  rnn_->ForwardPrefilled();
}

template <typename Dtype>
void ResetRNNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence index inputs.";

  rnn_->Backward();
}

INSTANTIATE_CLASS(ResetRNNLayer);

}  // namespace caffe
