#!/usr/bin/env python

import os
import random
random.seed(3)
import sys

sys.path.append('../')

from generate_hdf5_data import SequenceGenerator, HDF5SequenceWriter

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = '<unk>'

class FlickrSequenceGenerator(SequenceGenerator):
  # filenames should be a list of
  #     [(french1, english1), ..., (frenchK, englishK)]
  def __init__(self, dataset, vocab_filename, langs=[], max_words=20):
    self.max_words = max_words
    num_empty_lines = 0
    self.images = []
    num_missing = 0
    known_images = {}
    for image_list_path, image_root in zip(dataset['image_list'], dataset['image_root']):
      with open(image_list_path, 'rb') as image_list:
        image_filenames = image_list.readlines()
      for image_filename in image_filenames:
        image_filename = image_filename.strip()
        image_path = '%s/%s' % (image_root, image_filename)
        if os.path.isfile(image_path):
          assert image_filename not in known_images  # no duplicates allowed
          known_images[image_filename] = {}
          known_images[image_filename]['path'] = image_path
          known_images[image_filename]['sentences'] = []
        else:
          num_missing += 1
          print 'Warning (#%d): image not found: %s' % (num_missing, image_path)
    for caption_list_path in dataset['caption_list']:
      with open(caption_list_path, 'rb') as caption_list:
        captions = caption_list.readlines()
      for caption in captions:
        caption_words = caption.split()
        assert len(caption_words) >= 1
        caption_filename, _ = caption_words[0].split('#')
        if caption_filename in known_images:
          sentence = caption_words[1:]
          known_images[caption_filename]['sentences'].append(sentence)
    self.image_sentence_pairs = []
    num_no_sentences = 0
    for image_filename, metadata in known_images.iteritems():
      if not metadata['sentences']:
        num_no_sentences += 1
        print 'Warning (#%d): image with no sentences: %s' % (num_no_sentences, image_filename)
      for sentence in metadata['sentences']:
        self.image_sentence_pairs.append((metadata['path'], sentence))
    self.index = 0
    self.num_resets = 0
    self.num_truncates = 0
    self.num_pads = 0
    self.num_outs = 0
    self.init_vocabulary(vocab_filename)
    random.shuffle(self.image_sentence_pairs)
    self.image_list = []
    SequenceGenerator.__init__(self)

  def streams_exhausted(self):
    return self.num_resets > 0

  def init_vocabulary(self, vocab_filename):
    print 'Initializing the vocabulary.'
    with open(vocab_filename, 'rb') as vocab_file:
      self.init_word_vocabulary(vocab_file)

  def init_word_vocabulary(self, vocab_file):
    # initialize the vocabulary with the UNK word
    vocabulary = {UNK_IDENTIFIER: 0}
    vocabulary_inverted = [UNK_IDENTIFIER]
    num_words_dataset = 0
    for line in vocab_file.readlines():
      split_line = line.split()
      word = split_line[0]
      num_words_dataset += 1
      assert word not in vocabulary
      vocabulary[word] = len(vocabulary_inverted)
      vocabulary_inverted.append(word)
    num_words_vocab = len(vocabulary.keys())
    print ('Initialized vocabulary with %d unique words ' +
           '(from %d total words in dataset).') % \
          (num_words_vocab, num_words_dataset)
    assert len(vocabulary_inverted) == num_words_vocab
    self.vocabulary = vocabulary
    self.vocabulary_inverted = vocabulary_inverted

  def dump_vocabulary(self, vocab_filename):
    print 'Dumping vocabulary to file: %s' % vocab_filename
    with open(vocab_filename, 'wb') as vocab_file:
      for word in self.vocabulary_inverted:
        vocab_file.write('%s\n' % word)
    print 'Done.'

  def dump_image_file(self, image_filename, dummy_image_filename=None):
    print 'Dumping image list to file: %s' % image_filename
    with open(image_filename, 'wb') as image_file:
      for word in self.image_list:
        image_file.write('%s\n' % word)
    if dummy_image_filename is not None:
      print 'Dumping image list with dummy labels to file: %s' % dummy_image_filename
      with open(dummy_image_filename, 'wb') as image_file:
        for word in self.image_list:
          image_file.write('%s 0\n' % word)
    print 'Done.'

  def next_line(self):
    num_lines = float(len(self.image_sentence_pairs))
    if self.index % 10000 == 0:
      print 'Processed %d/%d (%f%%) lines' % (self.index, num_lines,
                                              100 * self.index / num_lines)
    self.index += 1
    if self.index == num_lines:
      self.index = 0
      self.num_resets += 1

  def line_to_stream(self, sentence):
    stream = []
    for word in sentence:
      if word in self.vocabulary:
        stream.append(self.vocabulary[word])
      else:  # unknown word; append UNK
        stream.append(self.vocabulary[UNK_IDENTIFIER])
    # increment the stream -- 0 will be the EOS character
    stream = [s + 1 for s in stream]
    return stream

  def get_streams(self):
    image_filename, line = self.image_sentence_pairs[self.index]
    stream = self.line_to_stream(line)
    out = {}
    if self.max_words < len(stream) + 1:
      self.num_truncates += 1
      # print 'Warning (#%d): truncating length %d stream' % (self.num_truncates, len(stream))
      stream = stream[:(self.max_words - 1)]
      assert self.max_words == len(stream) + 1
    pad = self.max_words - (len(stream) + 1)
    assert pad >= 0
    if pad > 0:
      self.num_pads += 1
    self.num_outs += 1
    padding = [0] * pad
    out['stage_indicators'] = [1] * (len(stream) + 1) + padding
    out['input_sentence'] = [0] + stream + padding
    out['target_sentence'] = stream + [0] + padding
    self.image_list.append(image_filename)
    self.next_line()
    return out

def preprocess_flickr():
  VOCAB_FILE = './vocabs/vocab_coco_flickr1M_30k.txt'
  IMAGE_LIST_PATTERN = './flickr30k/flickr30k_%s_names.txt'
  IMAGE_PATTERN = './flickr30k/flickr30k-%s-images'
  CAPTION_PATTERN = './cocoflickr/mix_data/flickr30k_%s_CleanCaptions.txt'

  BUFFER_SIZE = 100
  DATASETS = [
    ('train', 30000, {
      'image_list' : [IMAGE_LIST_PATTERN % 'train'],
      'image_root' : [IMAGE_PATTERN % 'train'],
      'caption_list' : [CAPTION_PATTERN % 'train'],
    }),
    ('valid', 1500, {
      'image_list' : [IMAGE_LIST_PATTERN % 'val'],
      'image_root' : [IMAGE_PATTERN % 'val'],
      'caption_list' : [CAPTION_PATTERN % 'val'],
    }),
  ]
  MAX_WORDS = 20
  OUTPUT_DIR = './cocoflickr/flickr30k_hdf5/buffer_%d_maxwords_%d' % (BUFFER_SIZE, MAX_WORDS)
  OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR

  vocab = None
  for dataset_name, batch_stream_length, dataset in DATASETS:
    output_path = OUTPUT_DIR_PATTERN % dataset_name
    sg = FlickrSequenceGenerator(dataset, VOCAB_FILE)
    sg.batch_stream_length = batch_stream_length
    sg.batch_num_streams = BUFFER_SIZE
    writer = HDF5SequenceWriter(sg, output_dir=output_path)
    writer.write_to_exhaustion()
    writer.write_filelists()
    vocab_out_path = '%s/vocabulary.txt' % OUTPUT_DIR
    sg.dump_vocabulary(vocab_out_path)
    image_out_path = '%s/image_list.txt' % output_path
    image_dummy_labels_out_path = '%s/image_list.with_dummy_labels.txt' % output_path
    sg.dump_image_file(image_out_path, image_dummy_labels_out_path)
    num_outs = sg.num_outs
    num_pads = sg.num_pads
    num_truncates = sg.num_truncates
    print 'Padded %d/%d sequences; truncated %d/%d sequences' % \
        (num_pads, num_outs, num_truncates, num_outs)


if __name__ == "__main__":
  preprocess_flickr()
