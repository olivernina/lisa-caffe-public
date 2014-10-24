// Copyright 2014 BVLC and contributors.
// This program converts a set of videos to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_videoset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format:
//   7
//   3
//   subfolder1/video1frame1.JPEG
//   subfolder1/video1frame2.JPEG
//   subfolder1/video1frame3.JPEG
//   9
//   4
//   ...
// In the above example, the first video has label 7 and has 3 frames, with the
// first frame at path subfolder1/video1frame1.JPEG, etc.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;
using std::vector;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 4 || argc > 5) {
    printf("Convert a set of videos to the leveldb format used\n"
        "as input for Caffe.\n"
        "Usage:\n"
        "    convert_videoset ROOTFOLDER/ LISTFILE DB_NAME"
        " RANDOM_SHUFFLE_DATA[0 or 1]\n");
    return 1;
  }
  std::ifstream infile(argv[2]);
  vector<std::pair<vector<string>, int> > videos;
  string line;
  unsigned int label;
  unsigned int num_frames;
  int num_videos = 0;
  while (infile) {
    infile >> label;
    infile >> num_frames;
    LOG(INFO) << "Video " << num_videos << " has label " << label
              << " and #frames: " << num_frames;
    ++num_videos;
    CHECK_GE(num_frames, 1) << "Videos must contain at least 1 frame.";

    vector<string> frames(num_frames);
    //could also write logic to determine which chunk of video we take if videos too long
    for (int frame_id = 0;  frame_id < frames.size(); ++frame_id) {
      infile >> line;
      frames[frame_id] = line;
    }
    videos.push_back(std::make_pair(frames, label));
  }
  if (argc == 5 && argv[4][0] == '1') {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    std::random_shuffle(videos.begin(), videos.end());
  }
  LOG(INFO) << "A total of " << videos.size() << " videos.";

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO) << "Opening leveldb " << argv[3];
  leveldb::Status status = leveldb::DB::Open(
      options, argv[3], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[3];

  string root_folder(argv[1]);
  Datum frame_datum;
  int count = 0;
  const int kMaxKeyLength = 17;//kMaxKeyLength = 1024;
  const int kNumVideosPerBatch = 100;
  char key_cstr[kMaxKeyLength];
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  int data_size;
  bool data_size_initialized = false;
  Datum video_datum;
  for (int video_id = 0; video_id < videos.size(); ++video_id) {
    const vector<string>& frames = videos[video_id].first;
    const int label = videos[video_id].second;

//    if (video_id > 9498){
//      LOG(INFO) << 'where problem starts';
//    }

    for (int frame_id = 0; frame_id < frames.size(); ++frame_id) {  //have added item to datum: current_frame (did this in proto as well)
      const string& frame_path = root_folder + frames[frame_id];
      video_datum.set_current_frame(frame_id);
      video_datum.set_frames(frames.size());
      video_datum.set_label(label);
      video_datum.clear_data();
      if (!ReadImageToDatum(frame_path, label, &frame_datum)) {
        LOG(FATAL) << "Failed to read image to datum: " << frame_path;
      }
      if (!frame_id) {
        video_datum.set_channels(frame_datum.channels());
        video_datum.set_height(frame_datum.height());
        video_datum.set_width(frame_datum.width());
      }
      if (!data_size_initialized) {
        data_size = frame_datum.channels() *
                    frame_datum.height() * frame_datum.width();
        data_size_initialized = true;
        LOG(ERROR) << "Frame size is " << frame_datum.channels()
                   << " x " << frame_datum.height()
                   << " x " << frame_datum.width();
      } else {
        data_size = frame_datum.channels() *
                    frame_datum.height() * frame_datum.width();
//        const string& data = frame_datum.data();
//        if (data.size() !=data_size){
//          CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
//                                           << data.size();
//        }
      }
      video_datum.set_data(video_datum.data() + frame_datum.data());
      //now need to add this to batch...
      const char* video_key = frames[0].c_str();  
      const int num_chars_printed =  
          snprintf(key_cstr, kMaxKeyLength, "%08d%08d", video_id, frame_id); 
          //snprintf(key_cstr, kMaxKeyLength, "%08d%08d_%s", count, frame_id, video_key); //changed so that I could use "get" in datalayers
      CHECK_LE(num_chars_printed, kMaxKeyLength)
          << "Filename too long: " << key_cstr;
      string value;
      // get the value
      CHECK_EQ(video_datum.channels(),3);
      video_datum.SerializeToString(&value);
      batch->Put(string(key_cstr), value);

    }
    LOG(INFO) << "Processed video " << count  
              << " with " << frames.size() << " frames; total data size is "
              << video_datum.data().size();
    // sequential
    if (++count % kNumVideosPerBatch == 0) {
      status = db->Write(leveldb::WriteOptions(), batch);
      CHECK(status.ok()) << "LevelDB write failed.";
      LOG(ERROR) << "Processed " << count << " videos.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }
  // write the last batch
  if (count % kNumVideosPerBatch != 0) {
    status = db->Write(leveldb::WriteOptions(), batch);
    CHECK(status.ok()) << "LevelDB write failed.";
    LOG(ERROR) << "Processed " << count << " videos.";
  }
  delete batch;
  delete db;
  return 0;
}
