/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = this->layer_param_.top_size();
  hdf_blobs_.resize(top_size);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = 4;

  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());
  }

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  int num = hdf_blobs_[0]->num();
  for (int i = 1; i < top_size; ++i) {
    CHECK_EQ(hdf_blobs_[i]->num(), num);
  }
  DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->num() << " rows";
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_param().source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  source_file.close();
  num_files_ = hdf_filenames_.size();
  LOG(INFO) << "Number of HDF5 files: " << num_files_;

  // Load the first HDF5 file and initialize the line counter.
  Reset();

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  const int top_size = this->layer_param_.top_size();
  for (int i = 0; i < top_size; ++i) {
    top[i]->Reshape(batch_size, hdf_blobs_[i]->channels(),
                    hdf_blobs_[i]->height(), hdf_blobs_[i]->width());
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Reset() {
  current_file_ = 0;
  LoadHDF5FileData(hdf_filenames_[current_file_].c_str());
  current_row_ = 0;
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
          &top[j]->mutable_cpu_data()[num_rows_copied * data_dim]);
    }
    current_row_ += num_rows_to_copy;
    num_rows_copied += num_rows_to_copy;
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5DataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5DataLayer);

}  // namespace caffe
