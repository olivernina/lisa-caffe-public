#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ConcatAlongChannels(const int count, const int num,
    const int in_channels, const int out_channels, const int out_offset_channel,
    const int spatial_dim, const bool backwards, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    const int n = index / spatial_dim / in_channels;
    const int in_c = (index / spatial_dim) % in_channels;
    const int s = index % spatial_dim;
    const int out_c = in_c + out_offset_channel;
    const int out_index = (n * out_channels + out_c) * spatial_dim + s;
    if (backwards) {
      out[index] = in[out_index];
    } else {
      out[out_index] = in[index];
    }
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (concat_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      caffe_copy(bottom[i]->count(), bottom_data,
        top_data + top[0]->offset(offset_num));
      offset_num += bottom[i]->num();
    }
  } else if (concat_dim_ == 1) {
    const bool kBackwards = false;
    const int num = top[0]->num();
    const int out_channels = top[0]->channels();
    const int spatial_dim = top[0]->height() * top[0]->width();
    int offset_channel = 0;
    for (int i = 0; i < bottom.size(); ++i) {
      const int channels = bottom[i]->channels();
      const int count = bottom[i]->count();
      const Dtype* bottom_data = bottom[i]->gpu_data();
      ConcatAlongChannels<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, num, channels, out_channels, offset_channel,
          spatial_dim, kBackwards, bottom_data, top_data);
      offset_channel += channels;
    }
  } else {
    LOG(FATAL) << "concat_dim along dim" << concat_dim_ <<
      " not implemented yet";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  if (concat_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < bottom.size(); ++i) {
      Blob<Dtype>* blob = bottom[i];
      if (propagate_down[i]) {
        Dtype* bottom_diff = blob->mutable_gpu_diff();
        caffe_copy(blob->count(), top_diff + top[0]->offset(offset_num),
                       bottom_diff);
      }
      offset_num += blob->num();
    }
  } else if (concat_dim_ == 1) {
    const bool kBackwards = true;
    const int num = top[0]->num();
    const int out_channels = top[0]->channels();
    const int spatial_dim = top[0]->height() * top[0]->width();
    int offset_channel = 0;
    for (int i = 0; i < bottom.size(); ++i) {
      if (propagate_down[i]) {
        const int channels = bottom[i]->channels();
        const int count = bottom[i]->count();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        ConcatAlongChannels<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, num, channels, out_channels, offset_channel,
            spatial_dim, kBackwards, top_diff, bottom_diff);
        offset_channel += channels;
      }
    }
  } else {
    LOG(FATAL) << "concat_dim along dim" << concat_dim_ <<
      " not implemented yet";
  }
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_CLASS(ConcatLayer);

}  // namespace caffe
