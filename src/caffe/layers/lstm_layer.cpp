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
  hidden_param.mutable_inner_product_param()->set_num_output(hidden_dim_ * 4);
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  hidden_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
  biased_hidden_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

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
  flush_slice_param->set_name("flush slice");

  LayerParameter* x_flatten_param = net_param.add_layers();
  x_flatten_param->set_type(LayerParameter_LayerType_FLATTEN);
  x_flatten_param->add_bottom("x");
  x_flatten_param->add_top("x_flat");
  x_flatten_param->set_name("x flatten");

  LayerParameter* x_slice_param = net_param.add_layers();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->add_bottom("x_flat");
  x_slice_param->set_name("x slice");

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
    x_slice_param->add_top("x_" + ts);

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
    {
      LayerParameter* input_concat_layer = net_param.add_layers();
      input_concat_layer->set_name("input_concat_" + ts);
      input_concat_layer->set_type(LayerParameter_LayerType_CONCAT);
      input_concat_layer->add_bottom("h_" + tm1s + "_flushed");
      input_concat_layer->add_bottom("x_" + ts);
      input_concat_layer->add_top("input_" + ts);
    }
    {
      LayerParameter* w_param = net_param.add_layers();
      w_param->CopyFrom(biased_hidden_param);
      w_param->set_name("transform_" + ts);
      w_param->add_param("W");
      w_param->add_param("b");
      w_param->add_bottom("input_" + ts);
      w_param->add_top("gate_input_" + ts);
    }

    // Add layers to compute the cell vector c.
    //
    // Inputs: c_{t-1}, i_t, f_t, o_t, g_t, flush_t
    // Outputs: c_t, h_t
    //
    // c_t_term_1 = flush_t * f_t .* c_{t-1}
    // c_t_term_2 = i_t .* g_t
    // c_t = c_t_term_1 + c_t_term_2
    // tanh_c_t = \tanh[c_t]
    // h_t = o_t .* tanh_c_t
    //
    // Backward:
    //
    // Inputs: dE/dc_t, dE/dh_t (and all inputs & outputs from Forward)
    // Outputs: dE/dc_{t-1}, dE/di_t, dE/df_t, dE/do_t, dE/dg_t
    //
    // dE/do_t = dE/dh_t * dh_t/do_t = dE/dh_t * tanh_c_t
    //
    // dE/dc_t_term_{1,2} = dE/dc_t + dE/dh_t * dh_t/dtanh_c_t * dtanh_c_t/dc_t
    //                    = dE/dc_t + dE/dh_t * o_t * (1 - tanh_c_t * tanh_c_t)
    // dE/di_t = dE/dc_t_term_2 * g_t
    // dE/df_t = dE/dc_t_term_1 * c_{t-1}
    // dE/dc_{t-1} = dE/dc_t_term_1 * f_t
    // dE/dg_t = dE/dc_t_term_2 * i_t
    //
    {
      LayerParameter* lstm_unit_param = net_param.add_layers();
      lstm_unit_param->set_type(LayerParameter_LayerType_LSTM_UNIT);
      lstm_unit_param->add_bottom("c_" + tm1s);
      lstm_unit_param->add_bottom("gate_input_" + ts);
      lstm_unit_param->add_bottom("flush_" + tm1s);
      lstm_unit_param->add_top("c_" + ts);
      lstm_unit_param->add_top("h_" + ts);
      lstm_unit_param->set_name("unit_" + ts);
    }
    output_concat_layer.add_bottom("h_" + ts);
  }
  net_param.add_layers()->CopyFrom(output_concat_layer);
  {
    LayerParameter* c_T_copy_param = net_param.add_layers();
    c_T_copy_param->CopyFrom(split_param);
    c_T_copy_param->add_bottom("c_" + int_to_str(T_));
    c_T_copy_param->add_top("c_T");
  }

  const string& layer_name = this->layer_param_.name();
  for (int i = 0; i < net_param.layers_size(); ++i) {
    LayerParameter* layer = net_param.mutable_layers(i);
    layer->set_name(layer_name + "_" + layer->name());
  }

  lstm_.reset(new Net<Dtype>(net_param));

  lstm_->set_debug_info(this->layer_param_.lstm_param().lstm_debug_info());

  x_input_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("x").get());
  flush_input_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("flush").get());
  h_input_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("h_0").get());
  c_input_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("c_0").get());
  output_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("h").get());
  ts = int_to_str(T_);
  h_output_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("h_" + ts).get());
  c_output_blob_ = CHECK_NOTNULL(lstm_->blob_by_name("c_T").get());

  // 4 inputs: x, flush, h_0, and c_0.
  CHECK_EQ(4, lstm_->input_blobs().size());
  // 2 outputs: h and c_T.
  CHECK_EQ(2, lstm_->output_blobs().size());

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
  CHECK_EQ(2, this->blobs_.size());

  const int hidden_timestep_dim = buffer_size_ * hidden_dim_;
  Dtype* hidden_output_diff = h_output_blob_->mutable_cpu_diff();
  Dtype* cell_output_diff = c_output_blob_->mutable_cpu_diff();
  CHECK_EQ(hidden_timestep_dim, h_output_blob_->count());
  caffe_set(hidden_timestep_dim, Dtype(0), hidden_output_diff);
  CHECK_EQ(hidden_timestep_dim, c_output_blob_->count());
  caffe_set(hidden_timestep_dim, Dtype(0), cell_output_diff);
}

template <typename Dtype>
void LSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), hidden_dim_, 1, 1);
  x_input_blob_->ShareData(*bottom[0]);
  x_input_blob_->ShareDiff(*bottom[0]);
  flush_input_blob_->ShareData(*bottom[1]);
  output_blob_->ShareData(*top[0]);
  output_blob_->ShareDiff(*top[0]);
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

  DCHECK_EQ(timestep_dim, h_input_blob_->count());
  DCHECK_EQ(timestep_dim, h_output_blob_->count());
  const Dtype* hidden_output_data = h_output_blob_->cpu_data();
  Dtype* hidden_input_data = h_input_blob_->mutable_cpu_data();
  caffe_copy(timestep_dim, hidden_output_data, hidden_input_data);

  DCHECK_EQ(timestep_dim, c_input_blob_->count());
  DCHECK_EQ(timestep_dim, c_output_blob_->count());
  const Dtype* cell_output_data = c_output_blob_->cpu_data();
  Dtype* cell_input_data = c_input_blob_->mutable_cpu_data();
  caffe_copy(timestep_dim, cell_output_data, cell_input_data);

  // Run the LSTM in forward mode.
  lstm_->ForwardPrefilled();
}

template <typename Dtype>
void LSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence index inputs.";

  lstm_->Backward();
}

#ifdef CPU_ONLY
STUB_GPU(LSTMLayer);
#endif

INSTANTIATE_CLASS(LSTMLayer);

}  // namespace caffe
