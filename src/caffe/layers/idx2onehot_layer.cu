#include <algorithm>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Idx2OneHotForward(const int nthreads,
    const Dtype* bottom_data, const int dim, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int the_hot_one = static_cast<int>(bottom_data[index]);
    top_data[index * dim + the_hot_one] = Dtype(1);
  }
}

template <typename Dtype>
void Idx2OneHotLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  const int num = top[0]->num();
  // NOLINT_NEXT_LINE(whitespace/operators)
  Idx2OneHotForward<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
      num, bottom_data, dim_, top_data);
}

#ifdef CPU_ONLY
STUB_GPU(Idx2OneHotLayer);
#endif

INSTANTIATE_CLASS(Idx2OneHotLayer);

}  // namespace caffe
