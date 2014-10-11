#!/usr/bin/env python

import os
import random
random.seed(3)
import sys

sys.path.append('../')

from generate_hdf5_data import SequenceGenerator, HDF5SequenceWriter

# UNK_IDENTIFIERS are the words used to identify unknown words
UNK_IDENTIFIERS = ['<fr_unk>', '<en_unk>']

class WMTSequenceGenerator(SequenceGenerator):
  # filenames should be a list of
  #     [(french1, english1), ..., (frenchK, englishK)]
  def __init__(self, filenames, vocab_filenames):
    self.lines = []
    num_empty_lines = 0
    for a, b in filenames:
      with open(a, 'rb') as file_a:
        a_lines = file_a.readlines()
      with open(b, 'rb') as file_b:
        b_lines = file_b.readlines()
      num_lines = len(a_lines)
      assert num_lines == len(b_lines)
      print 'Adding %d lines from (%s, %s)' % (num_lines, a, b)
      for a_line, b_line in zip(a_lines, b_lines):
        a_line = a_line.strip()
        b_line = b_line.strip()
        if len(a_line) == 0 or len(b_line) == 0:
          num_empty_lines += 1
          continue
        self.lines.append((a_line, b_line))
    if num_empty_lines > 0:
      print 'Warning: ignoring %d empty lines.' % num_empty_lines
    self.line_index = 0
    self.num_resets = 0
    self.vocabulary = []
    self.vocabulary_inverted = []
    for index, vocab_filename in enumerate(vocab_filenames):
      self.init_vocabulary(vocab_filename, index)
    random.shuffle(self.lines)
    SequenceGenerator.__init__(self)

  def streams_exhausted(self):
    return self.num_resets > 0

  def init_vocabulary(self, vocab_filename, vocab_index):
    print 'Initializing the vocabulary.'
    # initialize the vocabulary with the UNK word
    assert vocab_index == len(self.vocabulary)
    vocabulary = {UNK_IDENTIFIERS[vocab_index]: 0}
    vocabulary_inverted = [UNK_IDENTIFIERS[vocab_index]]
    num_words_dataset = 0
    with open(vocab_filename, 'rb') as vocab_file:
      for line in vocab_file.readlines():
        split_line = line.split()
        word = split_line[0]
        num_words_dataset += 1
        assert word not in vocabulary
        vocabulary[word] = len(vocabulary_inverted)
        vocabulary_inverted.append(word)
    num_words_vocab = len(vocabulary.keys())
    print ('Initialized the vocabulary with %d unique words ' +
           '(from %d total words in dataset).') % (num_words_vocab, num_words_dataset)
    assert len(vocabulary_inverted) == num_words_vocab
    self.vocabulary.append(vocabulary)
    self.vocabulary_inverted.append(vocabulary_inverted)

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

  def line_to_stream(self, vocab_index, line):
    stream = []
    for word in line.split():
      if word in self.vocabulary[vocab_index]:
        stream.append(self.vocabulary[vocab_index][word])
      else:  # unknown word; append UNK
        stream.append(self.vocabulary[vocab_index][UNK_IDENTIFIERS[vocab_index]])
    # increment the stream -- 0 will be the EOS character
    stream = [s + 1 for s in stream]
    return stream

  def get_streams(self):
    line_a, line_b = self.lines[self.line_index]
    stream_a = self.line_to_stream(0, line_a)
    stream_b = self.line_to_stream(1, line_b)
    self.next_line()
    out = {}

    # encoding stage
    out['data'] = list(reversed(stream_a))
    out['targets'] = [0] * len(stream_a)
    out['stage_indicators'] = [0] * len(stream_a)
    out['encoder_cont'] = [0] + [1] * (len(stream_a) - 1)
    out['decoder_cont'] = [0] * len(stream_a)
    out['encoder_to_decoder'] = [0] * len(stream_a)

    # decoding stage
    out['data'] += [0] + stream_b
    out['targets'] += stream_b + [0]
    out['stage_indicators'] += [1] * (len(stream_b) + 1)
    out['encoder_cont'] += [1] + [0] * len(stream_b)
    out['decoder_cont'] += [0] + [1] * len(stream_b)
    out['encoder_to_decoder'] += [1] + [0] * len(stream_b)

    return out

if __name__ == "__main__":
  BUFFER_SIZE = 100
  DATASET_PATH_PATTERN = './wmt14_data/ptb.%s.txt'
  OUTPUT_DIR = './wmt_hdf5/buffer_%d' % BUFFER_SIZE
  VOCAB_PATH = '%s/wmt_vocabulary.txt' % OUTPUT_DIR
  OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR
  DATASETS = [
    ('train', [
       './wmt14_data/ccb2_pc30.%s.txt',
       './wmt14_data/crawl.%s.txt',
       './wmt14_data/dev08_11.%s.txt',
       './wmt14_data/ep7_pc45.%s.txt',
#        './wmt14_data/nc9.%s.txt',
       './wmt14_data/un2000_pc34.%s.txt'
     ]),
     ('valid', [
      './wmt14_data/dev/ntst1213_invocab.%s.txt'
     ])
  ]
  VOCAB = './wmt14_data/%sVocab.txt'
  LANGS = ['fr', 'en']

  vocab_paths = [VOCAB % lang for lang in LANGS]
  for dataset_name, dataset_path_patterns in DATASETS:
    dataset_paths = [tuple(list(p % lang for lang in LANGS))
                     for p in dataset_path_patterns]
    for path_a, path_b in dataset_paths:
      assert os.path.exists(path_a)
      assert os.path.exists(path_b)
    output_path = OUTPUT_DIR_PATTERN % dataset_name
    sg = WMTSequenceGenerator(dataset_paths, vocab_paths)
    sg.batch_num_streams = BUFFER_SIZE
    writer = HDF5SequenceWriter(sg, output_dir=output_path)
    writer.write_to_exhaustion()
    writer.write_filelists()
