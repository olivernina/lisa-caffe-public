#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FlattenLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int channels_out = bottom[0]->channels() * bottom[0]->height()
      * bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), channels_out, 1, 1);
  count_ = bottom[0]->num() * channels_out;
  CHECK_EQ(count_, bottom[0]->count());
  CHECK_EQ(count_, top[0]->count());
  top[0]->ShareData(*bottom[0]);
  top[0]->ShareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(FlattenLayer);

}  // namespace caffe
