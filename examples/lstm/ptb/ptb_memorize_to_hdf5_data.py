#!/usr/bin/env python

import os
import random
import sys

sys.path.append('../')

from generate_hdf5_data import HDF5SequenceWriter
from ptb_to_hdf5_data import PTBSequenceGenerator

# UNK_IDENTIFIER is the word used to identify unknown words in the PTB data files
UNK_IDENTIFIER = '<unk>'

class PTBMemorizeSequenceGenerator(PTBSequenceGenerator):
  def get_streams(self):
    stream = PTBSequenceGenerator.get_streams(self)['data'][1:]
    out = {}
    out['data'] = list(reversed(stream)) + [0] + stream
    out['targets'] = [0] * len(stream) + stream + [0]
    out['stage_indicators'] = [0] * len(stream) + [1] * (len(stream) + 1)
    return out

if __name__ == "__main__":
  DATASET_PATH_PATTERN = './ptb_data/ptb.%s.txt'
  OUTPUT_DIR = './ptb_memorize_hdf5'
  VOCAB_PATH = '%s/ptb_vocabulary.txt' % OUTPUT_DIR
  OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR
  DATASET_NAMES = ('train', 'valid', 'test')
  vocabulary = None
  for dataset in DATASET_NAMES:
    dataset_path = DATASET_PATH_PATTERN % dataset
    output_path = OUTPUT_DIR_PATTERN % dataset
    assert os.path.exists(dataset_path)
    sg = PTBMemorizeSequenceGenerator(dataset_path, vocabulary=vocabulary)
    sg.batch_num_streams = 200
    writer = HDF5SequenceWriter(sg, output_dir=output_path)
    writer.write_to_exhaustion()
    writer.write_filelists()
    if vocabulary is None:
      vocabulary = sg.vocabulary
      sg.dump_vocabulary(VOCAB_PATH)
    assert vocabulary is not None
