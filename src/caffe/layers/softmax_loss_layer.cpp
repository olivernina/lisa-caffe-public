#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type(LayerParameter_LayerType_SOFTMAX);
  softmax_layer_.reset(GetLayer<Dtype>(softmax_param));
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  if (bottom.size() > 2) {
    CHECK_EQ(bottom[0]->num(), bottom[2]->num());
    CHECK_EQ(bottom[1]->count(), bottom[2]->count());
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  loss_multiplier_.ReshapeLike(*bottom[1]);
  if (bottom.size() <= 2 &&
      loss_multiplier_.cpu_data()[loss_multiplier_.count() - 1] != Dtype(1)) {
    caffe_set(loss_multiplier_.count(), Dtype(1),
              loss_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* weights = NULL;
  if (bottom.size() > 2) {
    weights = bottom[2]->cpu_data();
  }
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  Dtype loss = 0;
  Dtype weight = 1;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
     const int int_label = static_cast<int>(label[i * spatial_dim + j]);
     DCHECK_LT(int_label, dim) << "Label for (" << i << ", " << j << ") is greater than number of classes.";
     loss -= weight * log(std::max(prob_data[i * dim + int_label * spatial_dim + j],
		Dtype(FLT_MIN)));
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* weights = NULL;
    if (bottom.size() > 2) {
      weights = bottom[2]->cpu_data();
    }
    int count = prob_.count();
    int num = prob_.num();
    int dim = count / num;
    int spatial_dim = prob_.height() * prob_.width();
    Dtype weight = 1;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        if (weights) {
          weight = *weights;
          for (int k = i * dim + j; k < (i + 1) * dim + j; k += spatial_dim) {
            bottom_diff[k] *= weight;
          }
          ++weights;
        }
        if (weight == Dtype(0)) { continue; }
        bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
            * spatial_dim + j] -= weight;
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    Dtype diff_scale = loss_weight / num / spatial_dim;
    caffe_scal(prob_.count(), diff_scale, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
