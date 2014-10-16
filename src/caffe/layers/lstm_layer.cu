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

  const int timestep_dim = buffer_size_ * hidden_dim_;

  CHECK_EQ(timestep_dim, h_input_blob_->count());
  CHECK_EQ(timestep_dim, h_output_blob_->count());
  const Dtype* hidden_output_data = h_output_blob_->gpu_data();
  Dtype* hidden_input_data = h_input_blob_->mutable_gpu_data();
  caffe_copy(timestep_dim, hidden_output_data, hidden_input_data);

  CHECK_EQ(timestep_dim, c_input_blob_->count());
  CHECK_EQ(timestep_dim, c_output_blob_->count());
  const Dtype* cell_output_data = c_output_blob_->gpu_data();
  Dtype* cell_input_data = c_input_blob_->mutable_gpu_data();
  caffe_copy(timestep_dim, cell_output_data, cell_input_data);

  CHECK_EQ(bottom[1]->count(), flush_input_blob_->count());
  caffe_copy(bottom[1]->count(), bottom[1]->cpu_data(),
             flush_input_blob_->mutable_cpu_data());

  // Run the LSTM in forward mode.
  lstm_->ForwardPrefilled();
}

template <typename Dtype>
void LSTMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence index inputs.";

  lstm_->Backward();

  if (!propagate_down[0]) { return; }
  const int count = x_input_blob_->count();
  caffe_copy(count, x_input_blob_->gpu_diff(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_CLASS(LSTMLayer);

}  // namespace caffe
