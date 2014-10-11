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

    # encoding stage
    out['data'] = list(reversed(stream))
    out['targets'] = [0] * len(stream)
    out['stage_indicators'] = [0] * len(stream)
    out['encoder_cont'] = [0] + [1] * (len(stream) - 1)
    out['decoder_cont'] = [0] * len(stream)
    out['encoder_to_decoder'] = [0] * len(stream)

    # decoding stage
    out['data'] += [0] + stream
    out['targets'] += stream + [0]
    out['stage_indicators'] += [1] * (len(stream) + 1)
    out['encoder_cont'] += [1] + [0] * len(stream)
    out['decoder_cont'] += [0] + [1] * len(stream)
    out['encoder_to_decoder'] += [1] + [0] * len(stream)

    return out

if __name__ == "__main__":
  BUFFER_SIZE = 20
  DATASET_PATH_PATTERN = './ptb_data/ptb.%s.txt'
  OUTPUT_DIR = './ptb_memorize_hdf5_buffer_%d' % BUFFER_SIZE
  VOCAB_PATH = '%s/ptb_vocabulary.txt' % OUTPUT_DIR
  OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR
  DATASET_NAMES = ('train', 'valid', 'test')
  vocabulary = None
  for dataset in DATASET_NAMES:
    dataset_path = DATASET_PATH_PATTERN % dataset
    output_path = OUTPUT_DIR_PATTERN % dataset
    assert os.path.exists(dataset_path)
    sg = PTBMemorizeSequenceGenerator(dataset_path, vocabulary=vocabulary)
    sg.batch_num_streams = BUFFER_SIZE
    writer = HDF5SequenceWriter(sg, output_dir=output_path)
    writer.write_to_exhaustion()
    writer.write_filelists()
    if vocabulary is None:
      vocabulary = sg.vocabulary
      sg.dump_vocabulary(VOCAB_PATH)
    assert vocabulary is not None
