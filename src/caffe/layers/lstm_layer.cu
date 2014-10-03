#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LSTMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
  if (Caffe::phase() == Caffe::TEST) {
    lstm_->ShareWeightData();
  }

  // Run the LSTM in forward mode.
  lstm_->ForwardPrefilled();

  // Copy the LSTM outputs.
  const int output_timestep_dim = buffer_size_ * hidden_dim_;
  const Dtype* output_data;
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int t = 0; t < T_; ++t) {
    output_data = output_blobs_[t]->gpu_data();
    caffe_copy(output_timestep_dim, output_data,
               top_data + t * output_timestep_dim);
  }
}

template <typename Dtype>
void LSTMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence index inputs.";
  const int output_timestep_dim = buffer_size_ * hidden_dim_;
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* output_diff;
  for (int t = 0; t < T_; ++t) {
    CHECK_EQ(output_timestep_dim, output_blobs_[t]->count());
    output_diff = output_blobs_[t]->mutable_gpu_diff();
    caffe_copy(output_timestep_dim, top_diff + t * output_timestep_dim,
               output_diff);
  }

  lstm_->Backward();
  lstm_->AccumulateSharedWeightDiffs();
}

INSTANTIATE_CLASS(LSTMLayer);

}  // namespace caffe
