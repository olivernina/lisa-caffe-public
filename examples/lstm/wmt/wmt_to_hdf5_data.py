#!/usr/bin/env python

import os
import random
random.seed(3)
import sys

sys.path.append('../')

from generate_hdf5_data import SequenceGenerator, HDF5SequenceWriter

# UNK_IDENTIFIERS are the words used to identify unknown words
UNK_IDENTIFIERS = [u'<fr_unk>', u'<en_unk>']

class WMTSequenceGenerator(SequenceGenerator):
  # filenames should be a list of
  #     [(french1, english1), ..., (frenchK, englishK)]
  def __init__(self, filenames, vocab_filenames=None, langs=[], lang_specs=None, chars=False):
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
    self.langs = langs
    self.lang_specs = lang_specs
    self.output_character_data = False
    self.chars = chars
    if vocab_filenames is None:
      assert self.chars
      for index, _ in enumerate(self.langs):
        self.init_vocabulary(None, index)
    else:
      for index, vocab_filename in enumerate(vocab_filenames):
        self.init_vocabulary(vocab_filename, index)
    random.shuffle(self.lines)
    SequenceGenerator.__init__(self)

  def streams_exhausted(self):
    return self.num_resets > 0

  def init_vocabulary(self, vocab_filename, vocab_index):
    print 'Initializing the vocabulary.'
    assert vocab_index == len(self.vocabulary)
    if self.chars:
      # self.init_char_vocabulary(vocab_index)
      self.init_char_vocabulary_easy(vocab_index)
    else:
      with open(vocab_filename, 'rb') as vocab_file:
        self.init_word_vocabulary(vocab_file, vocab_index)

  def init_char_vocabulary_easy(self, vocab_index, max_char=256):
    vocabulary = {UNK_IDENTIFIERS[vocab_index]: 0}
    vocabulary_inverted = [UNK_IDENTIFIERS[vocab_index]]
    for char_index in range(max_char):
      char = chr(char_index)
      vocabulary[char] = len(vocabulary_inverted)
      vocabulary_inverted.append(char)
    self.vocabulary.append(vocabulary)
    self.vocabulary_inverted.append(vocabulary_inverted)
    num_chars_vocab = len(vocabulary.keys())
    assert len(vocabulary_inverted) == num_chars_vocab
    print 'Initialized vocabulary (%s) with %d unique chars ' % \
          (self.langs[vocab_index], num_chars_vocab)

  def init_char_vocabulary(self, vocab_index):
    vocabulary_counts = {}
    for lines in self.lines:
      line = lines[vocab_index]
      for char in line:
        if char not in vocabulary_counts: vocabulary_counts[char] = 0
        vocabulary_counts[char] += 1
    vocabulary_by_count = sorted(vocabulary_counts.keys(),
                                 key=(lambda k: -1 * vocabulary_counts[k]))
    vocabulary = {UNK_IDENTIFIERS[vocab_index]: 0}
    vocabulary_inverted = [UNK_IDENTIFIERS[vocab_index]]
    for index, char in enumerate(vocabulary_by_count):
      vocabulary[char] = index
      vocabulary_inverted.append(char)
    assert len(vocabulary_inverted) == num_words_vocab
    self.vocabulary.append(vocabulary)
    self.vocabulary_inverted.append(vocabulary_inverted)
    num_chars_vocab = len(vocabulary.keys())
    print 'Initialized vocabulary (%s) with %d unique chars ' % \
          (self.langs[vocab_index], num_chars_vocab)

  def init_word_vocabulary(self, vocab_file, vocab_index):
    # initialize the vocabulary with the UNK word
    vocabulary = {UNK_IDENTIFIERS[vocab_index]: 0}
    vocabulary_inverted = [UNK_IDENTIFIERS[vocab_index]]
    num_words_dataset = 0
    for line in vocab_file.readlines():
      split_line = line.split()
      word = split_line[0]
      num_words_dataset += 1
      assert word not in vocabulary
      vocabulary[word] = len(vocabulary_inverted)
      vocabulary_inverted.append(word)
    num_words_vocab = len(vocabulary.keys())
    print ('Initialized vocabulary (%s) with %d unique words ' +
           '(from %d total words in dataset).') % \
          (self.langs[vocab_index], num_words_vocab, num_words_dataset)
    assert len(vocabulary_inverted) == num_words_vocab
    self.vocabulary.append(vocabulary)
    self.vocabulary_inverted.append(vocabulary_inverted)

  def dump_vocabulary(self, vocab_filename, vocab_index):
    print 'Dumping vocabulary to file: %s' % vocab_filename
    with open(vocab_filename, 'wb') as vocab_file:
      for word in self.vocabulary_inverted[vocab_index]:
        vocab_file.write('%s\n' % word)
    print 'Done.'

  def next_line(self):
    num_lines = float(len(self.lines))
    if self.line_index % 10000 == 0:
      print 'Processed %d/%d (%f%%) lines' % (self.line_index, num_lines,
                                              100 * self.line_index / num_lines)
    self.line_index += 1
    if self.line_index == num_lines:
      self.line_index = 0
      self.num_resets += 1

  def line_to_stream(self, vocab_index, line):
    if self.chars:
      return self.line_to_char_stream(vocab_index, line)
    else:
      return self.line_to_word_stream(vocab_index, line)

  def line_to_char_stream(self, vocab_index, line):
    stream = []
    for char in line:
      if char in self.vocabulary[vocab_index]:
        stream.append(self.vocabulary[vocab_index][char])
      else:  # unknown char; append UNK
        print 'Warning: found unknown char: %s' % char
        stream.append(self.vocabulary[vocab_index][UNK_IDENTIFIERS[vocab_index]])
    # increment the stream -- 0 will be the EOS character
    stream = [s + 1 for s in stream]
    return stream

  def line_to_word_stream(self, vocab_index, line):
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
    parallel_lines = self.lines[self.line_index]
    input_length = 0
    output_length = 0
    streams = []
    inputs = []
    outputs = []
    # Loop over the languages once to compute the streams and input/output length.
    for lang_index, lang_line in enumerate(parallel_lines):
      lang = self.langs[lang_index]
      lang_spec = self.lang_specs[lang]
      streams.append(self.line_to_stream(lang_index, lang_line))
      stream_length = len(streams[-1])
      inputs.append(lang_spec['input'])
      if inputs[-1] and stream_length > input_length:
        input_length = stream_length
      outputs.append(lang_spec['output'])
      if outputs[-1] and stream_length > output_length:
        output_length = stream_length
    # Loop again to create the dict of output data.
    out = {}
    out['stage_indicators'] = [0] * (input_length + 1) + [1] * (output_length + 1)
    for lang, lang_stream, input, output in \
        zip(self.langs, streams, inputs, outputs):
      # encoding stage
      if input:
        input_stream = list(reversed(lang_stream))
        # prepend EOS padding (empty if len(input_stream) == input_length)
        pad = [0] * (input_length - len(input_stream))
        out['data_%s' % lang] = pad + [0] + input_stream
        out['targets_%s' % lang] = pad + input_stream + [0]
        out['cont_%s' % lang] = pad + [0] + [1] * len(input_stream)
      else:
        zeros = [0] * (input_length + 1)
        out['data_%s' % lang] = list(zeros)
        out['targets_%s' % lang] = list(zeros)
        out['cont_%s' % lang] = list(zeros)
      # decoding stage
      if output:
        output_stream = list(lang_stream)
        # append EOS padding (empty if len(output_stream) == output_length)
        pad = [0] * (output_length - len(output_stream))
        out['data_%s' % lang] += [0] + output_stream + pad
        out['targets_%s' % lang] += output_stream + [0] + pad
        out['cont_%s' % lang] += [1] * (len(output_stream) + 1) + pad
      else:
        zeros = [0] * (output_length + 1)
        out['data_%s' % lang] += list(zeros)
        out['targets_%s' % lang] += list(zeros)
        out['cont_%s' % lang] += list(zeros)
    self.next_line()
    return out

def lang_specs_to_str(langs, lang_specs):
  out = []
  for lang in langs:
    spec = lang_specs[lang]
    out.append('%s_%s%s' %
        (lang, 'i' if spec['input'] else '', 'o' if spec['output'] else ''))
  return '-'.join(out)

def preprocess_en_to_fr_chars():
  BUFFER_SIZE = 100
  BATCH_STREAM_LENGTH = 100 * 1000  # 100k
  VAL_ROOT = './wmt14_data/wmt_raw/dev/'
  DATASETS = [
    ('train', [
       './wmt14_data/wmt_raw/training/europarl-v7.fr-en.%s',
     ]),
     ('valid', [
       './wmt14_data/wmt_raw/dev/newsdev2014.%s',
       './wmt14_data/wmt_raw/dev/newssyscomb2009.%s',
       './wmt14_data/wmt_raw/dev/news-test2008.%s',
       './wmt14_data/wmt_raw/dev/newstest2009.%s',
       './wmt14_data/wmt_raw/dev/newstest2010.%s',
       './wmt14_data/wmt_raw/dev/newstest2011.%s',
       './wmt14_data/wmt_raw/dev/newstest2012.%s',
       './wmt14_data/wmt_raw/dev/newstest2013.%s',
     ])
  ]
  LANGS = ['fr', 'en']
  LANG_SPECS = {
    'fr': {'input': True, 'output': True},
    'en': {'input': True, 'output': True},
  }
  OUTPUT_DIR = './wmt_char_hdf5/%s/buffer_%d' % (lang_specs_to_str(LANGS, LANG_SPECS), BUFFER_SIZE)
  OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR
  vocab_out_paths = ['wmt_char_hdf5/vocabs/vocabulary.%s.txt' % lang
                     for lang in LANGS]
  for dataset_name, dataset_path_patterns in DATASETS:
    dataset_paths = [tuple(list(p % lang for lang in LANGS))
                     for p in dataset_path_patterns]
    for path_a, path_b in dataset_paths:
      assert os.path.exists(path_a)
      assert os.path.exists(path_b)
    output_path = OUTPUT_DIR_PATTERN % dataset_name
    sg = WMTSequenceGenerator(dataset_paths, langs=LANGS, lang_specs=LANG_SPECS, chars=True)
    sg.batch_stream_length = BATCH_STREAM_LENGTH
    sg.batch_num_streams = BUFFER_SIZE
    for vocab_index, vocab_out_path in enumerate(vocab_out_paths):
      sg.dump_vocabulary(vocab_out_path, vocab_index)
    writer = HDF5SequenceWriter(sg, output_dir=output_path)
    writer.write_to_exhaustion()
    writer.write_filelists()

def preprocess_en_to_fr_words():
  BUFFER_SIZE = 10
  BATCH_STREAM_LENGTH = 100000 # 100k
  DATASET_PATH_PATTERN = './wmt14_data/ptb.%s.txt'
  DATASETS = [
    ('train', [
       './wmt14_data/ccb2_pc30.%s.txt',
       './wmt14_data/crawl.%s.txt',
       './wmt14_data/dev08_11.%s.txt',
       './wmt14_data/ep7_pc45.%s.txt',
       './wmt14_data/nc9.%s.txt',
       './wmt14_data/un2000_pc34.%s.txt'
     ]),
     ('valid', [
      './wmt14_data/dev/ntst1213_invocab.%s.txt'
     ])
  ]
  VOCAB = './wmt14_data/%sVocab.txt'
  LANGS = ['fr', 'en']
  LANG_SPECS = {
    'fr': {'input': True, 'output': True},
    'en': {'input': False, 'output': True},
  }
  OUTPUT_DIR = './wmt_hdf5/%s/buffer_%d' % (lang_specs_to_str(LANGS, LANG_SPECS), BUFFER_SIZE)
  OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR

  vocab_paths = [VOCAB % lang for lang in LANGS]
  for dataset_name, dataset_path_patterns in DATASETS:
    dataset_paths = [tuple(list(p % lang for lang in LANGS))
                     for p in dataset_path_patterns]
    for path_a, path_b in dataset_paths:
      assert os.path.exists(path_a)
      assert os.path.exists(path_b)
    output_path = OUTPUT_DIR_PATTERN % dataset_name
    sg = WMTSequenceGenerator(dataset_paths, vocab_paths, langs=LANGS, lang_specs=LANG_SPECS)
    sg.batch_stream_length = BATCH_STREAM_LENGTH
    sg.batch_num_streams = BUFFER_SIZE
    writer = HDF5SequenceWriter(sg, output_dir=output_path)
    writer.write_to_exhaustion()
    writer.write_filelists()
  vocab_out_paths = ['%s/vocabulary.%s.txt' % (OUTPUT_DIR, lang)
                     for lang in LANGS]
  for vocab_index, vocab_out_path in enumerate(vocab_out_paths):
    sg.dump_vocabulary(vocab_out_path, vocab_index)


if __name__ == "__main__":
#   preprocess_en_to_fr_words()
  preprocess_en_to_fr_chars()
