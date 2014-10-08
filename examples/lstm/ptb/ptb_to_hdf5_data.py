#!/usr/bin/env python

import os
import random
import sys

sys.path.append('../')

from generate_hdf5_data import SequenceGenerator, HDF5SequenceWriter

# UNK_IDENTIFIER is the word used to identify unknown words in the PTB data files
UNK_IDENTIFIER = '<unk>'

class PTBSequenceGenerator(SequenceGenerator):
  def __init__(self, data_filename, vocabulary=None, shuffle=True):
    with open(data_filename, 'rb') as data_file:
      self.lines = data_file.readlines()
    if vocabulary is None:
      self.init_vocabulary()
    else:
      self.vocabulary = vocabulary
    if shuffle: random.shuffle(self.lines)
    self.line_index = 0
    self.num_resets = 0
    SequenceGenerator.__init__(self)

  def streams_exhausted(self):
    return self.num_resets > 0

  def init_vocabulary(self):
    print 'Initializing the vocabulary.'
    assert len(self.lines) > 0
    # initialize the vocabulary with the UNK word
    self.vocabulary = {UNK_IDENTIFIER: 0}
    self.vocabulary_inverted = [UNK_IDENTIFIER]
    self.vocab_counts = [0]
    num_words_dataset = 0
    for line in self.lines:
      split_line = line.split()
      num_words_dataset += len(split_line)
      for word in split_line:
        if word in self.vocabulary:
          self.vocab_counts[self.vocabulary[word]] += 1
        else:
          self.vocabulary_inverted.append(word)
          self.vocabulary[word] = len(self.vocab_counts)
          self.vocab_counts.append(1)
    num_words_vocab = len(self.vocabulary.keys())
    print ('Initialized the vocabulary with %d unique words ' +
           '(from %d total words in dataset).') % (num_words_vocab, num_words_dataset)
    assert len(self.vocab_counts) == num_words_vocab
    assert len(self.vocabulary_inverted) == num_words_vocab
    if self.vocab_counts[self.vocabulary[UNK_IDENTIFIER]] == 0:
      print 'Warning: the count for the UNK identifier "%s" was 0.' % UNK_IDENTIFIER

  def dump_vocabulary(self, vocab_filename):
    print 'Dumping vocabulary to file: %s' % vocab_filename
    with open(vocab_filename, 'wb') as vocab_file:
      for word in self.vocabulary_inverted:
        vocab_file.write('%s\n' % word)
    print 'Done.'

  def next_line(self):
    self.line_index += 1
    if self.line_index == len(self.lines):
      self.line_index = 0
      self.num_resets += 1

  def get_streams(self):
    stream = []
    for word in self.lines[self.line_index].split():
      if word in self.vocabulary:
        stream.append(self.vocabulary[word])
      else:  # unknown word; append UNK
        stream.append(self.vocabulary[UNK_IDENTIFIER])
    # increment the stream -- 0 will be the EOS character
    stream = [s + 1 for s in stream]
    out = {}
    out['data'] = [0] + stream
    out['targets'] = stream + [0]
    self.next_line()
    return out

if __name__ == "__main__":
  BUFFER_SIZE = 20
  DATASET_PATH_PATTERN = './ptb_data/ptb.%s.txt'
  OUTPUT_DIR = './ptb_hdf5_buffer_%d' % BUFFER_SIZE
  VOCAB_PATH = '%s/ptb_vocabulary.txt' % OUTPUT_DIR
  OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR
  DATASET_NAMES = ('train', 'valid', 'test')
  vocabulary = None
  for dataset in DATASET_NAMES:
    dataset_path = DATASET_PATH_PATTERN % dataset
    output_path = OUTPUT_DIR_PATTERN % dataset
    assert os.path.exists(dataset_path)
    sg = PTBSequenceGenerator(dataset_path, vocabulary=vocabulary)
    sg.batch_num_streams = BUFFER_SIZE
    writer = HDF5SequenceWriter(sg, output_dir=output_path)
    writer.write_to_exhaustion()
    writer.write_filelists()
    if vocabulary is None:
      vocabulary = sg.vocabulary
      sg.dump_vocabulary(VOCAB_PATH)
    assert vocabulary is not None
