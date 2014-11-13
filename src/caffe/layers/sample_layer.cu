#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SampleForward(const int nthreads, const int dim,
    const Dtype* bottom_data, const Dtype* rand_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const Dtype* offset_bottom_data = bottom_data + dim * index;
    const Dtype r = rand_data[index];
    Dtype cum_sum = 0;
    top_data[index] = -1;
    for (int i = 0; i < dim; ++i) {
      cum_sum += offset_bottom_data[i];
      if (cum_sum >= r) {
        top_data[index] = i;
        break;
      }
    }
  }
}

template <typename Dtype>
void SampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype* rand_data = rand_.mutable_gpu_data();
  caffe_gpu_rng_uniform<Dtype>(rand_.count(), Dtype(0), Dtype(1), rand_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  SampleForward<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
      num, dim, bottom_data, rand_data, top_data);
}

INSTANTIATE_CLASS(SampleLayer);

}  // namespace caffe
