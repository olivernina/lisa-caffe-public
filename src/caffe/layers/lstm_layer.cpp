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
string LSTMLayer<Dtype>::int_to_str(const int t) {
  ostringstream num;
  num << t;
  return num.str();
}

template <typename Dtype>
void LSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  output_blobs_.clear();
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
  net_param.add_input("c_" + ts);
  net_param.add_input_dim(buffer_size_);
  net_param.add_input_dim(hidden_dim_);
  net_param.add_input_dim(1);
  net_param.add_input_dim(1);
  net_param.add_input("h_" + ts);
  net_param.add_input_dim(buffer_size_);
  net_param.add_input_dim(hidden_dim_);
  net_param.add_input_dim(1);
  net_param.add_input_dim(1);

  LayerParameter* flush_slice_param = net_param.add_layers();
  flush_slice_param->CopyFrom(slice_param);
  flush_slice_param->add_bottom("flush");

  {
    LayerParameter* w_xi_param = net_param.add_layers();
    w_xi_param->CopyFrom(biased_hidden_param);
    w_xi_param->add_bottom("x");
    w_xi_param->add_param("W_{xi}");
    w_xi_param->add_param("b_i");
    w_xi_param->add_top("W_{xi} x + b_i");
  }
  LayerParameter* w_xi_slice_param = net_param.add_layers();
  w_xi_slice_param->CopyFrom(slice_param);
  w_xi_slice_param->add_bottom("W_{xi} x + b_i");

  {
    LayerParameter* w_xf_param = net_param.add_layers();
    w_xf_param->CopyFrom(biased_hidden_param);
    w_xf_param->add_bottom("x");
    w_xf_param->add_param("W_{xf}");
    w_xf_param->add_param("b_f");
    w_xf_param->add_top("W_{xf} x + b_f");
  }
  LayerParameter* w_xf_slice_param = net_param.add_layers();
  w_xf_slice_param->CopyFrom(slice_param);
  w_xf_slice_param->add_bottom("W_{xf} x + b_f");

  {
    LayerParameter* w_xc_param = net_param.add_layers();
    w_xc_param->CopyFrom(biased_hidden_param);
    w_xc_param->add_bottom("x");
    w_xc_param->add_param("W_{xc}");
    w_xc_param->add_param("b_c");
    w_xc_param->add_top("W_{xc} x + b_c");
  }
  LayerParameter* w_xc_slice_param = net_param.add_layers();
  w_xc_slice_param->CopyFrom(slice_param);
  w_xc_slice_param->add_bottom("W_{xc} x + b_c");

  {
    LayerParameter* w_xo_param = net_param.add_layers();
    w_xo_param->CopyFrom(biased_hidden_param);
    w_xo_param->add_bottom("x");
    w_xo_param->add_param("W_{xo}");
    w_xo_param->add_param("b_o");
    w_xo_param->add_top("W_{xo} x + b_o");
  }
  LayerParameter* w_xo_slice_param = net_param.add_layers();
  w_xo_slice_param->CopyFrom(slice_param);
  w_xo_slice_param->add_bottom("W_{xo} x + b_o");

  string tm1s;
  for (int t = 1; t <= T_; ++t) {
    string tm1s = int_to_str(t - 1);
    string ts = int_to_str(t);

    flush_slice_param->add_top("flush_" + tm1s);

    w_xi_slice_param->add_top("W_{xi} x_" + ts + " + b_i");
    w_xf_slice_param->add_top("W_{xf} x_" + ts + " + b_f");
    w_xc_slice_param->add_top("W_{xc} x_" + ts + " + b_c");
    w_xo_slice_param->add_top("W_{xo} x_" + ts + " + b_o");

    // Add layers to flush the hidden and cell state, when beginning a new clip.
    {
      LayerParameter* flush_c_param = net_param.add_layers();
      flush_c_param->CopyFrom(sum_param);
      flush_c_param->mutable_eltwise_param()->set_coeff_blob(true);
      flush_c_param->add_bottom("c_" + tm1s);
      flush_c_param->add_bottom("flush_" + tm1s);
      flush_c_param->add_top("c_" + tm1s + "_flushed");
    }
    {
      LayerParameter* flush_h_param = net_param.add_layers();
      flush_h_param->CopyFrom(sum_param);
      flush_h_param->mutable_eltwise_param()->set_coeff_blob(true);
      flush_h_param->add_bottom("h_" + tm1s);
      flush_h_param->add_bottom("flush_" + tm1s);
      flush_h_param->add_top("h_" + tm1s + "_flushed");
    }

    // Add layers to compute the input vector i.
    // i_t = \sigmoid[
    //   W_{xi} x_t + W_{hi} h_{t-1} + W_{ci} c_{t-1} + b_i
    // ]
    //
    // (b_i computed in the W_{xi} InnerProductLayer, W_{ci} diagonal)
    {
      LayerParameter* w_hi_param = net_param.add_layers();
      w_hi_param->CopyFrom(hidden_param);
      w_hi_param->add_bottom("h_" + tm1s + "_flushed");
      w_hi_param->add_param("W_{hi}");
      w_hi_param->add_top("W_{hi} h_" + tm1s);
    }
    {
      LayerParameter* w_ci_param = net_param.add_layers();
      w_ci_param->CopyFrom(diag_hidden_param);
      w_ci_param->add_bottom("c_" + tm1s + "_flushed");
      w_ci_param->add_param("W_{ci}");
      w_ci_param->add_top("W_{ci} c_" + tm1s);
    }
    {
      LayerParameter* i_input_param = net_param.add_layers();
      i_input_param->CopyFrom(sum_param);
      i_input_param->add_bottom("W_{xi} x_" + ts + " + b_i");
      i_input_param->add_bottom("W_{hi} h_" + tm1s);
      i_input_param->add_bottom("W_{ci} c_" + tm1s);
      // identity component of W_{ci}
      i_input_param->add_bottom("c_" + tm1s + "_flushed");
      i_input_param->add_top("i_" + ts + "_input");
    }
    {
      LayerParameter* i_param = net_param.add_layers();
      i_param->CopyFrom(sigmoid_param);
      i_param->add_bottom("i_" + ts + "_input");
      i_param->add_top("i_" + ts);
    }

    // Add layers to compute the forgetting vector f.
    // f_t = \sigmoid[
    //   W_{xf} x_t + W_{hf} h_{t-1} + W_{cf} c_{t-1} + b_f
    // ]
    //
    // (b_f computed in the W_{xf} InnerProductLayer, W_{cf} diagonal)
    {
      LayerParameter* w_hf_param = net_param.add_layers();
      w_hf_param->CopyFrom(hidden_param);
      w_hf_param->add_bottom("h_" + tm1s + "_flushed");
      w_hf_param->add_param("W_{hf}");
      w_hf_param->add_top("W_{hf} h_" + tm1s);
    }
    {
      LayerParameter* w_cf_param = net_param.add_layers();
      w_cf_param->CopyFrom(diag_hidden_param);
      w_cf_param->add_bottom("c_" + tm1s + "_flushed");
      w_cf_param->add_param("W_{cf}");
      w_cf_param->add_top("W_{cf} c_" + tm1s);
    }
    {
      LayerParameter* f_input_param = net_param.add_layers();
      f_input_param->CopyFrom(sum_param);
      f_input_param->add_bottom("W_{xf} x_" + ts + " + b_f");
      f_input_param->add_bottom("W_{hf} h_" + tm1s);
      f_input_param->add_bottom("W_{cf} c_" + tm1s);
      // identity component of W_{cf}
      f_input_param->add_bottom("c_" + tm1s + "_flushed");
      f_input_param->add_top("f_" + ts + "_input");
    }
    {
      LayerParameter* f_param = net_param.add_layers();
      f_param->CopyFrom(sigmoid_param);
      f_param->add_bottom("f_" + ts + "_input");
      f_param->add_top("f_" + ts);
    }

    // Add layers to compute the cell vector c.
    // c_t = c_t_term_1 + c_t_term_2
    // c_t_term_1 = f_t .* c_{t-1}
    // c_t_term_2 = i_t .* \tanh[ W_{xc} x_t + W_{hc} h_{t-1} + b_c ]
    //
    // (b_c computed in the W_{xc} InnerProductLayer.)
    {
      LayerParameter* c_t_term_1_param = net_param.add_layers();
      c_t_term_1_param->CopyFrom(prod_param);
      c_t_term_1_param->add_bottom("f_" + ts);
      c_t_term_1_param->add_bottom("c_" + tm1s + "_flushed");
      c_t_term_1_param->add_top("c_" + ts + "_term_1");
    }
    {
      LayerParameter* w_hc_param = net_param.add_layers();
      w_hc_param->CopyFrom(hidden_param);
      w_hc_param->add_bottom("h_" + tm1s + "_flushed");
      w_hc_param->add_param("W_{hc}");
      w_hc_param->add_top("W_{hc} h_" + tm1s);
    }
    {
      LayerParameter* c_input_param = net_param.add_layers();
      c_input_param->CopyFrom(sum_param);
      c_input_param->add_bottom("W_{xc} x_" + ts + " + b_c");
      c_input_param->add_bottom("W_{hc} h_" + tm1s);
      c_input_param->add_top("c_" + ts + "_input");
    }
    {
      LayerParameter* c_act_param = net_param.add_layers();
      c_act_param->CopyFrom(tanh_param);
      c_act_param->add_bottom("c_" + ts + "_input");
      c_act_param->add_top("c_" + ts + "_act");
    }
    {
      LayerParameter* c_t_term_2_param = net_param.add_layers();
      c_t_term_2_param->CopyFrom(prod_param);
      c_t_term_2_param->add_bottom("i_" + ts);
      c_t_term_2_param->add_bottom("c_" + ts + "_act");
      c_t_term_2_param->add_top("c_" + ts + "_term_2");
    }
    {
      LayerParameter* c_param = net_param.add_layers();
      c_param->CopyFrom(sum_param);
      c_param->add_bottom("c_" + ts + "_term_1");
      c_param->add_bottom("c_" + ts + "_term_2");
      if (t == T_) {
        c_param->add_top("c_" + ts + "_copy");
      } else {
        c_param->add_top("c_" + ts);
      }
    }
    string c_t_name = "c_" + ts;
    if (t == T_) {
      LayerParameter* c_split_param = net_param.add_layers();
      c_split_param->CopyFrom(split_param);
      c_split_param->add_bottom("c_" + ts + "_copy");
      c_t_name += "_internal";
      c_split_param->add_top(c_t_name);
      c_split_param->add_top("c_" + ts);
    }

    // Add layers to compute the output vector o.
    // o_t = \sigmoid[
    //   W_{xo} x_t + W_{ho} h_{t-1} + W_{co} c_t + b_o
    // ]
    //
    // (b_c computed in the W_{xc} InnerProductLayer, W_{co} diagonal)
    {
      LayerParameter* w_ho_param = net_param.add_layers();
      w_ho_param->CopyFrom(hidden_param);
      w_ho_param->add_bottom("h_" + tm1s + "_flushed");
      w_ho_param->add_param("W_{ho}");
      w_ho_param->add_top("W_{ho} h_" + tm1s);
    }
    {
      LayerParameter* w_co_param = net_param.add_layers();
      w_co_param->CopyFrom(diag_hidden_param);
      w_co_param->add_bottom(c_t_name);
      w_co_param->add_param("W_{co}");
      w_co_param->add_top("W_{co} c_" + ts);
    }
    {
      LayerParameter* o_input_param = net_param.add_layers();
      o_input_param->CopyFrom(sum_param);
      o_input_param->add_bottom("W_{xo} x_" + ts + " + b_o");
      o_input_param->add_bottom("W_{ho} h_" + tm1s);
      o_input_param->add_bottom("W_{co} c_" + ts);
      // identity component of W_{co}
      o_input_param->add_bottom(c_t_name);
      o_input_param->add_top("o_" + ts + "_input");
    }
    {
      LayerParameter* o_param = net_param.add_layers();
      o_param->CopyFrom(sigmoid_param);
      o_param->add_bottom("o_" + ts + "_input");
      o_param->add_top("o_" + ts + "_internal");
      o_param->set_has_external_diff(true);
    }
    // Add a split layer so we have an internal diff (accumulated into
    // o_t_internal_copy) and an external diff (accumulated into o_t).
    {
      LayerParameter* o_split_param = net_param.add_layers();
      o_split_param->CopyFrom(split_param);
      o_split_param->add_bottom("o_" + ts + "_internal");
      o_split_param->add_top("o_" + ts + "_internal_copy");
      o_split_param->add_top("o_" + ts);
    }

    // Add layers to compute the hidden vector h.
    // h_t = o_t .* \tanh[ c_t ]
    {
      LayerParameter* c_t_act_param = net_param.add_layers();
      c_t_act_param->CopyFrom(tanh_param);
      c_t_act_param->add_bottom(c_t_name);
      c_t_act_param->add_top("c_" + ts + "_tanh");
    }
    {
      LayerParameter* h_t_param = net_param.add_layers();
      h_t_param->CopyFrom(prod_param);
      h_t_param->add_bottom("o_" + ts + "_internal_copy");
      h_t_param->add_bottom("c_" + ts + "_tanh");
      h_t_param->add_top("h_" + ts);
    }
  }

  lstm_.reset(new Net<Dtype>(net_param));

  lstm_->set_debug_info(this->layer_param_.lstm_param().lstm_debug_info());

  x_input_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("x").get());
  flush_input_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("flush").get());
  h_input_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("h_0").get());
  c_input_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("c_0").get());
  for (int t = 1; t <= T_; ++t) {
    string ts = int_to_str(t);
    output_blobs_.push_back(CHECK_NOTNULL(
        lstm_->blob_by_name("o_" + ts).get()));
  }
  ts = int_to_str(T_);
  h_output_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("h_" + ts).get());
  c_output_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("c_" + ts).get());

  // Should have x, flush, h_0, and c_0.
  CHECK_EQ(4, lstm_->input_blobs().size());
  // Should have one output for each timestep (o_t), plus the final hidden state
  // outputs h_T and c_T.
  CHECK_EQ(T_ + 2, lstm_->output_blobs().size());

  for (int i = 0; i < lstm_->params().size(); ++i) {
    if (lstm_->param_owners()[i] == -1) {
      LOG(INFO) << "Adding parameter " << this->blobs_.size() << ": "
                << lstm_->param_display_names()[i];
      this->blobs_.push_back(lstm_->params()[i]);
    }
  }
  for (int i = 0; i < lstm_->layers().size(); ++i) {
    for (int j = 0; j < lstm_->layers()[i]->blobs().size(); ++j) {
      CHECK(lstm_->layers()[i]->param_propagate_down(j))
          << "param_propagate_down not set for layer " << i << ", param " << j;
    }
  }

  this->param_propagate_down_.resize(this->blobs_.size(), true);
  CHECK_EQ(15, this->blobs_.size());

  const int hidden_timestep_dim = buffer_size_ * hidden_dim_;
  Dtype* hidden_output_diff = h_output_blob_->mutable_cpu_diff();
  Dtype* cell_output_diff = c_output_blob_->mutable_cpu_diff();
  CHECK_EQ(hidden_timestep_dim, h_output_blob_->count());
  caffe_set(hidden_timestep_dim, Dtype(0), hidden_output_diff);
  CHECK_EQ(hidden_timestep_dim, c_output_blob_->count());
  caffe_set(hidden_timestep_dim, Dtype(0), cell_output_diff);

  x_input_blob_->ShareData(*bottom[0]);
  x_input_blob_->ShareDiff(*bottom[0]);
  flush_input_blob_->ShareData(*bottom[1]);
}

template <typename Dtype>
void LSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), hidden_dim_, 1, 1);
}

template <typename Dtype>
void LSTMLayer<Dtype>::Reset() {
  Dtype* hidden_input_data = h_input_blob_->mutable_cpu_data();
  caffe_set(h_input_blob_->count(), Dtype(0), hidden_input_data);
  Dtype* cell_input_data = c_input_blob_->mutable_cpu_data();
  caffe_set(c_input_blob_->count(), Dtype(0), cell_input_data);
}

template <typename Dtype>
void LSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
  if (Caffe::phase() == Caffe::TEST) {
    lstm_->ShareWeightData();
  }

  const int timestep_dim = buffer_size_ * hidden_dim_;
  const Dtype* hidden_output_data = h_output_blob_->cpu_data();
  Dtype* hidden_input_data = h_input_blob_->mutable_cpu_data();
  caffe_copy(timestep_dim, hidden_output_data, hidden_input_data);
  const Dtype* cell_output_data = c_output_blob_->cpu_data();
  Dtype* cell_input_data = c_input_blob_->mutable_cpu_data();
  caffe_copy(timestep_dim, cell_output_data, cell_input_data);

  // Run the LSTM in forward mode.
  lstm_->ForwardPrefilled();

  // Copy the LSTM outputs.
  const Dtype* output_data;
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int t = 0; t < T_; ++t) {
    output_data = output_blobs_[t]->cpu_data();
    caffe_copy(timestep_dim, output_data, top_data + t * timestep_dim);
  }
}

template <typename Dtype>
void LSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence index inputs.";
  const int output_timestep_dim = buffer_size_ * hidden_dim_;
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* output_diff;
  for (int t = 0; t < T_; ++t) {
    CHECK_EQ(output_timestep_dim, output_blobs_[t]->count());
    output_diff = output_blobs_[t]->mutable_cpu_diff();
    caffe_copy(output_timestep_dim, top_diff + t * output_timestep_dim,
               output_diff);
  }

  lstm_->Backward();
}

#ifdef CPU_ONLY
STUB_GPU(LSTMLayer);
#endif

INSTANTIATE_CLASS(LSTMLayer);

}  // namespace caffe
