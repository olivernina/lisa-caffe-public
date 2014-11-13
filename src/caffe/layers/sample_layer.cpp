#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void SampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  rand_.Reshape(bottom[0]->num(), 1, 1, 1);
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void SampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype* rand_data = rand_.mutable_cpu_data();
  caffe_set(num, Dtype(-1), top_data);
  caffe_rng_uniform<Dtype>(rand_.count(), Dtype(0), Dtype(1), rand_data);
  for (int i = 0; i < num; ++i) {
    const Dtype r = rand_data[i];
    Dtype cum_sum = 0;
    for (int j = 0; j < dim; ++j) {
      cum_sum += bottom_data[j];
      if (cum_sum >= r) {
        top_data[i] = static_cast<Dtype>(j);
        break;
      }
    }
    bottom_data += dim;
  }
}

INSTANTIATE_CLASS(SampleLayer);

}  // namespace caffe
