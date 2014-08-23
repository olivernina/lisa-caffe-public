#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Idx2OneHotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  dim_ = this->layer_param_.idx21hot_param().dim();
  CHECK_GT(dim_, 0) << "Must specify positive output dimension.";
  CHECK_EQ(1, bottom[0]->channels());
}

template <typename Dtype>
void Idx2OneHotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(
      bottom[0]->num(), dim_, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void Idx2OneHotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int h = 0; h < bottom[0]->height(); ++h) {
      for (int w = 0; w < bottom[0]->width(); ++w) {
        const int index =
            static_cast<int>(bottom_data[bottom[0]->offset(n, 0, h, w)]);
        top_data[top[0]->offset(n, index, h, w)] = 1;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Idx2OneHotLayer);
#endif

INSTANTIATE_CLASS(Idx2OneHotLayer);

}  // namespace caffe
