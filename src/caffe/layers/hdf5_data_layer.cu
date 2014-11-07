/*
TODO:
- only load parts of the file, in accordance with a prototxt param "max_mem"
*/

#include <algorithm>
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  int num_rows_copied = 0;
  while (num_rows_copied < batch_size) {
    int num_rows_available = hdf_blobs_[0]->num() - current_row_;
    if (!num_rows_available) {
      if (num_files_ > 1) {
        ++current_file_;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          DLOG(INFO) << "Looping around to first file.";
        }
        LoadHDF5FileData(hdf_filenames_[current_file_].c_str());
      }
      current_row_ = 0;
      num_rows_available = hdf_blobs_[0]->num();
    }
    const int num_rows_needed = batch_size - num_rows_copied;
    const int num_rows_to_copy = min(num_rows_needed, num_rows_available);
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      const int data_dim = top[j]->count() / top[j]->num();
      caffe_copy(data_dim * num_rows_to_copy,
          &hdf_blobs_[j]->cpu_data()[current_row_ * data_dim],
          &top[j]->mutable_gpu_data()[num_rows_copied * data_dim]);
    }
    current_row_ += num_rows_to_copy;
    num_rows_copied += num_rows_to_copy;
  }
}

INSTANTIATE_CLASS(HDF5DataLayer);

}  // namespace caffe
