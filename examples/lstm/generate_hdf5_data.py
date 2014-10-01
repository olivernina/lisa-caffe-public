#!/usr/bin/env python

import h5py
import numpy as np
import os
import random
import sys

class SequenceGenerator():
  def __init__(self):
    self.dimension = 10
    self.batch_stream_length = 2000
    self.batch_num_streams = 8
    self.min_stream_length = 13
    self.max_stream_length = 17
    self.substream_names = None
    self.streams_initialized = False

  def streams_exhausted(self):
    return False

  def init_streams(self):
    self.streams = [None] * self.batch_num_streams
    self.stream_indices = [0] * self.batch_num_streams
    for i in range(self.batch_num_streams):
      self.reset_stream(i)
    self.streams_initialized = True

  def reset_stream(self, stream_index):
    streams = self.get_streams()
    stream_names = sorted(streams.keys())
    if self.substream_names is None:
      assert len(stream_names) > 0
      self.substream_names = stream_names
    assert self.substream_names == stream_names
    if self.streams[stream_index] is None:
      self.streams[stream_index] = {}
    stream_length = len(streams[stream_names[0]])
    for k, v in streams.iteritems():
      assert stream_length == len(v)
      self.streams[stream_index][k] = v
    self.stream_indices[stream_index] = 0

  def get_next_batch(self):
    if not self.streams_initialized:
      self.init_streams()
    batch_size = self.batch_num_streams * self.batch_stream_length
    batch = {}
    for name in self.substream_names:
      batch[name] = -1 * np.ones((batch_size, 1, 1, 1))
    batch_indicators = np.ones((batch_size, 1))
    for i in range(self.batch_num_streams):
      for t in range(self.batch_stream_length):
        index_in_batch = t * self.batch_num_streams + i
        for name in self.substream_names:
          if i >= len(self.stream_indices) or i > len(self.streams):
            import pdb; pdb.set_trace()
          batch[name][index_in_batch] = self.streams[i][name][self.stream_indices[i]]
        if self.stream_indices[i] == 0:
          batch_indicators[index_in_batch] = 0
        self.stream_indices[i] += 1
        if self.stream_indices[i] == len(self.streams[i][self.substream_names[0]]):
          self.reset_stream(i)
    return batch, batch_indicators

  def get_streams(self):
    raise Exception('get_streams should be overridden to return an iterable ' +
                    'of equal-length iterables.')

class InputOutputSequenceGenerator(SequenceGenerator):
  def get_streams(self):
    stream_length = random.randint((self.min_stream_length - 1) / 2,
                                   (self.max_stream_length - 1) / 2)
    stream = []
    for index in range(stream_length):
      stream.append(random.randint(1, self.dimension))

    out = {}
    out['data'] = list(reversed(stream)) + [0] + stream
    out['targets'] = [0] * len(stream) + stream + [0]
    out['stage_indicators'] = [0] * len(stream) + [1] * (len(stream) + 1)
    return out

class HDF5SequenceWriter():
  def __init__(self, sequence_generator, output_dir=None, verbose=False):
    self.generator = sequence_generator
    assert output_dir is not None  # required
    self.output_dir = output_dir
    if os.path.exists(output_dir):
      raise Exception('Output directory already exists: ' + output_dir)
    os.makedirs(output_dir)
    self.verbose = verbose
    self.filenames = []

  def write_batch(self):
    batch_comps, cont_indicators = self.generator.get_next_batch()
    batch_index = len(self.filenames)
    filename = '%s/batch_%d.h5' % (self.output_dir, batch_index)
    self.filenames.append(filename)
    h5file = h5py.File(filename, 'w')
    dataset = h5file.create_dataset('cont', shape=cont_indicators.shape, dtype=cont_indicators.dtype)
    dataset[:] = cont_indicators
    dataset = h5file.create_dataset('buffer_size', shape=(1,), dtype=np.int)
    dataset[:] = self.generator.batch_num_streams
    for key, batch in batch_comps.iteritems():
      if self.verbose:
        for s in range(self.generator.batch_num_streams):
          stream = np.array(self.generator.streams[s][key])
          print 'batch %d, stream %s, index %d: ' % (batch_index, key, s), stream
      h5dataset = h5file.create_dataset(key, shape=batch.shape, dtype=batch.dtype)
      h5dataset[:] = batch
      if self.verbose:
        if batch.shape[1] == 1:
          # assume index encoded data
          batch_shaped = batch[:, 0, 0, 0]
        else:
          # assume one-hot encoded data; convert to indexed by taking argmax
          batch_shaped = batch[:, :, 0, 0].argmax(axis=1)
        batch_shaped = batch_shaped.reshape((sg.batch_stream_length, sg.batch_num_streams))
        print "batch %d, stream %d:\n" % (batch_index, batch_comp_index), batch_shaped
    h5file.close()

  def write_to_exhaustion(self):
    while not self.generator.streams_exhausted():
      self.write_batch()

  def write_filelists(self):
    assert self.filenames is not None
    filelist_filename = '%s/hdf5_chunk_list.txt' % self.output_dir
    with open(filelist_filename, 'w') as listfile:
      for filename in self.filenames:
        listfile.write('%s\n' % filename)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    raise Exception('Usage: ./generate_hdf5_data.py <output_dir>')
  sg = InputOutputSequenceGenerator()
  verbose = False
  writer = HDF5SequenceWriter(sg, output_dir=sys.argv[1], verbose=verbose)
  num_batches = 200
  for b in range(num_batches):
    writer.write_batch()
  writer.write_filelists()
