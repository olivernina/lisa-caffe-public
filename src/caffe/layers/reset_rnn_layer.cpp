#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
string ResetRNNLayer<Dtype>::int_to_str(const int t) {
  ostringstream num;
  num << t;
  return num.str();
}

template <typename Dtype>
void ResetRNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  hidden_dim_ = this->layer_param_.lstm_param().hidden_dim();
  CHECK_GT(hidden_dim_, 0) << "hidden_dim must be positive.";
  buffer_size_ = this->layer_param_.lstm_param().buffer_size();
  CHECK_GT(buffer_size_, 0) << "buffer_size must be positive.";
  const FillerParameter& weight_filler =
      this->layer_param_.lstm_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.lstm_param().bias_filler();

  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_LE(buffer_size_, bottom[0]->num())
      << "buffer_size must be at most the number of inputs";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter hidden_param;
  hidden_param.set_type(LayerParameter_LayerType_INNER_PRODUCT);
  hidden_param.mutable_inner_product_param()->set_num_output(hidden_dim_);
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  hidden_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
  biased_hidden_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter diag_hidden_param;
  diag_hidden_param.set_type(LayerParameter_LayerType_DIAG_INNER_PRODUCT);
  diag_hidden_param.mutable_diag_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  LayerParameter sum_param;
  sum_param.set_type(LayerParameter_LayerType_ELTWISE);
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  LayerParameter prod_param(sum_param);
  prod_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_PROD);

  LayerParameter sigmoid_param;
  sigmoid_param.set_type(LayerParameter_LayerType_SIGMOID);

  LayerParameter tanh_param;
  tanh_param.set_type(LayerParameter_LayerType_TANH);

  LayerParameter slice_param;
  slice_param.set_type(LayerParameter_LayerType_SLICE);
  slice_param.mutable_slice_param()->set_slice_dim(0);

  LayerParameter split_param;
  split_param.set_type(LayerParameter_LayerType_SPLIT);

  // Maximum number of timesteps is the batch size.
  CHECK_EQ(0, bottom[0]->num() % buffer_size_)
      << "bottom num must be divisible by buffer_size (T = num / buffer_size)";
  T_ = bottom[0]->num() / buffer_size_;
  LOG(INFO) << "Initializing LSTM: assuming input batch contains "
            << T_ << " timesteps of " << buffer_size_ << " streams.";
  NetParameter net_param;
  net_param.set_force_backward(true);
  net_param.add_input("x");
  net_param.add_input_dim(bottom[0]->num());
  net_param.add_input_dim(bottom[0]->channels());
  net_param.add_input_dim(bottom[0]->height());
  net_param.add_input_dim(bottom[0]->width());
  net_param.add_input("flush");
  net_param.add_input_dim(bottom[0]->num());
  net_param.add_input_dim(1);
  net_param.add_input_dim(1);
  net_param.add_input_dim(1);
  string ts = int_to_str(0);
  net_param.add_input("h_" + ts);
  net_param.add_input_dim(buffer_size_);
  net_param.add_input_dim(hidden_dim_);
  net_param.add_input_dim(1);
  net_param.add_input_dim(1);

  LayerParameter* flush_slice_param = net_param.add_layers();
  flush_slice_param->CopyFrom(slice_param);
  flush_slice_param->add_bottom("flush");
  flush_slice_param->set_name("flush slice");

  {
    LayerParameter* w_xh_param = net_param.add_layers();
    w_xh_param->CopyFrom(biased_hidden_param);
    w_xh_param->add_bottom("x");
    w_xh_param->add_param("W_{xh}");
    w_xh_param->add_param("b_h");
    w_xh_param->add_top("W_{xh} x + b_h");
    w_xh_param->set_name("W_{xh} x + b_h");
  }
  LayerParameter* w_xh_slice_param = net_param.add_layers();
  w_xh_slice_param->CopyFrom(slice_param);
  w_xh_slice_param->add_bottom("W_{xh} x + b_h");
  w_xh_slice_param->set_name("W_{xh} x + b_h slice");

  {
    LayerParameter* w_xf_param = net_param.add_layers();
    w_xf_param->CopyFrom(biased_hidden_param);
    w_xf_param->add_bottom("x");
    w_xf_param->add_param("W_{xf}");
    w_xf_param->add_param("b_f");
    w_xf_param->add_top("W_{xf} x + b_f");
    w_xf_param->set_name("W_{xf} x + b_f");
  }
  LayerParameter* w_xf_slice_param = net_param.add_layers();
  w_xf_slice_param->CopyFrom(slice_param);
  w_xf_slice_param->add_bottom("W_{xf} x + b_f");
  w_xf_slice_param->set_name("W_{xf} x + b_f slice");

  {
    LayerParameter* w_xz_param = net_param.add_layers();
    w_xz_param->CopyFrom(biased_hidden_param);
    w_xz_param->add_bottom("x");
    w_xz_param->add_param("W_{xz}");
    w_xz_param->add_param("b_z");
    w_xz_param->add_top("W_{xz} x + b_z");
    w_xz_param->set_name("W_{xz} x + b_z");
  }
  LayerParameter* w_xz_slice_param = net_param.add_layers();
  w_xz_slice_param->CopyFrom(slice_param);
  w_xz_slice_param->add_bottom("W_{xz} x + b_z");
  w_xz_slice_param->set_name("W_{xz} x + b_z slice");

  LayerParameter output_concat_layer;
  output_concat_layer.set_name("h_concat");
  output_concat_layer.set_type(LayerParameter_LayerType_CONCAT);
  output_concat_layer.add_top("h");
  output_concat_layer.set_has_external_diff(true);
  output_concat_layer.mutable_concat_param()->set_concat_dim(0);

  string tm1s;
  for (int t = 1; t <= T_; ++t) {
    string tm1s = int_to_str(t - 1);
    string ts = int_to_str(t);

    flush_slice_param->add_top("flush_" + tm1s);

    w_xh_slice_param->add_top("W_{xh} x_" + ts + " + b_h");
    w_xf_slice_param->add_top("W_{xf} x_" + ts + " + b_f");
    w_xz_slice_param->add_top("W_{xz} x_" + ts + " + b_z");

    // Add layers to flush the hidden and cell state, when beginning a new clip.
    {
      LayerParameter* flush_h_param = net_param.add_layers();
      flush_h_param->CopyFrom(sum_param);
      flush_h_param->mutable_eltwise_param()->set_coeff_blob(true);
      flush_h_param->add_bottom("h_" + tm1s);
      flush_h_param->add_bottom("flush_" + tm1s);
      flush_h_param->add_top("h_" + tm1s + "_flushed");
      flush_h_param->set_name("h_" + tm1s + " flush");
    }

    // Add layers to compute the forgetting vector f.
    // f_t = \sigmoid[
    //   W_{xf} x_t + W_{hf} h_{t-1} + b_f
    // ]
    //
    // (b_f computed in the W_{xf} InnerProductLayer, W_{cf} diagonal)
    {
      LayerParameter* w_hf_param = net_param.add_layers();
      w_hf_param->CopyFrom(hidden_param);
      w_hf_param->add_bottom("h_" + tm1s + "_flushed");
      w_hf_param->add_param("W_{hf}");
      w_hf_param->add_top("W_{hf} h_" + tm1s);
      w_hf_param->set_name("W_{hf} h_" + tm1s);
    }
    {
      LayerParameter* f_input_param = net_param.add_layers();
      f_input_param->CopyFrom(sum_param);
      f_input_param->add_bottom("W_{xf} x_" + ts + " + b_f");
      f_input_param->add_bottom("W_{hf} h_" + tm1s);
      f_input_param->add_top("f_" + ts + "_input");
      f_input_param->set_name("f_" + ts + "_input");
    }
    {
      LayerParameter* f_param = net_param.add_layers();
      f_param->CopyFrom(sigmoid_param);
      f_param->add_bottom("f_" + ts + "_input");
      f_param->add_top("f_" + ts);
      f_param->set_name("f_" + ts);
    }

    // Add layers to compute z.
    // z_t = \sigmoid[
    //   W_{xz} x_t + W_{hz} h_{t-1} + b_z
    // ]
    {
      LayerParameter* w_hz_param = net_param.add_layers();
      w_hz_param->CopyFrom(hidden_param);
      w_hz_param->add_param("W_{hz}");
      w_hz_param->add_bottom("h_" + tm1s + "_flushed");
      w_hz_param->add_top("W_{hz} h_" + tm1s);
      w_hz_param->set_name("W_{hz} h_" + tm1s);
    }
    {
      LayerParameter* z_input_param = net_param.add_layers();
      z_input_param->CopyFrom(sum_param);
      z_input_param->add_bottom("W_{hz} h_" + tm1s);
      z_input_param->add_bottom("W_{xz} x_" + ts + " + b_z");
      z_input_param->add_top("z_" + ts + "_input");
      z_input_param->set_name("z_" + ts + "_input");
    }
    {
      LayerParameter* z_param = net_param.add_layers();
      z_param->CopyFrom(sigmoid_param);
      z_param->add_bottom("z_" + ts + "_input");
      z_param->add_top("z_" + ts);
      z_param->set_name("z_" + ts);
    }

    // Add layers to compute h_tilde.
    // h_tilde_t = \sigmoid[
    //   W_{xh} x_t + W_{hh} (h_{t-1} .* f_t) + b_h
    // ]
    {
      LayerParameter* h_t_term_1_param = net_param.add_layers();
      h_t_term_1_param->CopyFrom(prod_param);
      h_t_term_1_param->add_bottom("f_" + ts);
      h_t_term_1_param->add_bottom("h_" + tm1s + "_flushed");
      h_t_term_1_param->add_top("h_" + ts + "_forgotten");
      h_t_term_1_param->set_name("h_" + ts + "_forgotten");
    }
    {
      LayerParameter* w_hh_param = net_param.add_layers();
      w_hh_param->CopyFrom(hidden_param);
      w_hh_param->add_bottom("h_" + ts + "_forgotten");
      w_hh_param->add_param("W_{hh}");
      w_hh_param->add_top("W_{hh} h_" + tm1s);
      w_hh_param->set_name("W_{hh} h_" + tm1s);
    }
    {
      LayerParameter* h_input_param = net_param.add_layers();
      h_input_param->CopyFrom(sum_param);
      h_input_param->add_bottom("W_{hh} h_" + tm1s);
      h_input_param->add_bottom("W_{xh} x_" + ts + " + b_h");
      h_input_param->add_top("h_tilde_" + ts + "_input");
      h_input_param->set_name("h_tilde_" + ts + "_input");
    }
    {
      LayerParameter* h_tilde_param = net_param.add_layers();
      h_tilde_param->CopyFrom(sigmoid_param);
      h_tilde_param->add_bottom("h_tilde_" + ts + "_input");
      h_tilde_param->add_top("h_tilde_" + ts);
      h_tilde_param->set_name("h_tilde_" + ts);
    }

    // Add layers to compute the hidden vector h.
    // h_t = z_t .* h_{t-1} + (1 - z_t) .* h_tilde_t
    {
      LayerParameter* h_t_term_1_param = net_param.add_layers();
      h_t_term_1_param->CopyFrom(prod_param);
      h_t_term_1_param->add_bottom("z_" + ts);
      h_t_term_1_param->add_bottom("h_" + tm1s + "_flushed");
      h_t_term_1_param->add_top("h_" + ts + "_term_1");
      h_t_term_1_param->set_name("h_" + ts + "_term_1");
    }
    {
      LayerParameter* one_minus_z_t = net_param.add_layers();
      one_minus_z_t->set_type(LayerParameter_LayerType_POWER);
      one_minus_z_t->mutable_power_param()->set_scale(-1);
      one_minus_z_t->mutable_power_param()->set_shift(1);
      one_minus_z_t->add_bottom("z_" + ts);
      one_minus_z_t->add_top("one_minus_z_" + ts);
      one_minus_z_t->set_name("one_minus_z_" + ts);
    }
    {
      LayerParameter* h_t_term_2_param = net_param.add_layers();
      h_t_term_2_param->CopyFrom(prod_param);
      h_t_term_2_param->add_bottom("one_minus_z_" + ts);
      h_t_term_2_param->add_bottom("h_tilde_" + ts);
      h_t_term_2_param->add_top("h_" + ts + "_term_2");
      h_t_term_2_param->set_name("h_" + ts + "_term_2");
    }
    {
      LayerParameter* h_param = net_param.add_layers();
      h_param->CopyFrom(sum_param);
      h_param->add_bottom("h_" + ts + "_term_1");
      h_param->add_bottom("h_" + ts + "_term_2");
      h_param->add_top("h_" + ts);
      h_param->set_name("h_" + ts);
      output_concat_layer.add_bottom("h_" + ts);
    }
  }
  net_param.add_layers()->CopyFrom(output_concat_layer);

  const string& layer_name = this->layer_param_.name();
  for (int i = 0; i < net_param.layers_size(); ++i) {
    LayerParameter* layer = net_param.mutable_layers(i);
    layer->set_name(layer_name + "_" + layer->name());
  }

  rnn_.reset(new Net<Dtype>(net_param));

  rnn_->set_debug_info(this->layer_param_.lstm_param().lstm_debug_info());

  x_input_blob_ = CHECK_NOTNULL(rnn_->blob_by_name("x").get());
  flush_input_blob_ = CHECK_NOTNULL(rnn_->blob_by_name("flush").get());
  h_input_blob_ = CHECK_NOTNULL(rnn_->blob_by_name("h_0").get());
  output_blob_ = CHECK_NOTNULL(rnn_->blob_by_name("h").get());
  ts = int_to_str(T_);
  h_output_blob_ = CHECK_NOTNULL(rnn_->blob_by_name("h_" + ts).get());

  // 3 inputs: x, flush, h_0.
  CHECK_EQ(3, rnn_->input_blobs().size());
  // 2 outputs: main output (h) plus the final hidden state output h_T.
  CHECK_EQ(1, rnn_->output_blobs().size());

  for (int i = 0; i < rnn_->params().size(); ++i) {
    if (rnn_->param_owners()[i] == -1) {
      LOG(INFO) << "Adding parameter " << this->blobs_.size() << ": "
                << rnn_->param_display_names()[i];
      this->blobs_.push_back(rnn_->params()[i]);
    }
  }
  for (int i = 0; i < rnn_->layers().size(); ++i) {
    for (int j = 0; j < rnn_->layers()[i]->blobs().size(); ++j) {
      CHECK(rnn_->layers()[i]->param_propagate_down(j))
          << "param_propagate_down not set for layer " << i << ", param " << j;
    }
  }

  this->param_propagate_down_.resize(this->blobs_.size(), true);
  CHECK_EQ(9, this->blobs_.size());

  const int hidden_timestep_dim = buffer_size_ * hidden_dim_;
  Dtype* hidden_output_diff = h_output_blob_->mutable_cpu_diff();
  CHECK_EQ(hidden_timestep_dim, h_output_blob_->count());
  caffe_set(hidden_timestep_dim, Dtype(0), hidden_output_diff);
}

template <typename Dtype>
void ResetRNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), hidden_dim_, 1, 1);
  x_input_blob_->ShareData(*bottom[0]);
  x_input_blob_->ShareDiff(*bottom[0]);
  flush_input_blob_->ShareData(*bottom[1]);
  output_blob_->ShareData(*top[0]);
  output_blob_->ShareDiff(*top[0]);
}

template <typename Dtype>
void ResetRNNLayer<Dtype>::Reset() {
  Dtype* hidden_input_data = h_input_blob_->mutable_cpu_data();
  caffe_set(h_input_blob_->count(), Dtype(0), hidden_input_data);
}

template <typename Dtype>
void ResetRNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
  if (Caffe::phase() == Caffe::TEST) {
    rnn_->ShareWeightData();
  }

  const int timestep_dim = buffer_size_ * hidden_dim_;

  DCHECK_EQ(timestep_dim, h_input_blob_->count());
  DCHECK_EQ(timestep_dim, h_output_blob_->count());
  const Dtype* hidden_output_data = h_output_blob_->cpu_data();
  Dtype* hidden_input_data = h_input_blob_->mutable_cpu_data();
  caffe_copy(timestep_dim, hidden_output_data, hidden_input_data);

  // Run the LSTM in forward mode.
  rnn_->ForwardPrefilled();
}

template <typename Dtype>
void ResetRNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence index inputs.";

  rnn_->Backward();
}

#ifdef CPU_ONLY
STUB_GPU(ResetRNNLayer);
#endif

INSTANTIATE_CLASS(ResetRNNLayer);

}  // namespace caffe
