#!/usr/bin/env python

DEVICE_ID = -1
# DATA_FILE = './ptb_hdf5/train_batches/batch_0.h5'
# VOCAB_FILE = './ptb_hdf5/ptb_vocabulary.txt'
# MODEL_FILE = './snapshots/ptb_lstm_accum_grads_without_buggy_slicelayer_mom0.99_iter_9000.caffemodel'
# NET_FILE = './ptb_lstm_eval_net.prototxt'

# DATA_FILE = './ptb_memorize_hdf5_buffer_20/train_batches/batch_0.h5'
# VOCAB_FILE = './ptb_memorize_hdf5_buffer_20/ptb_vocabulary.txt'
# MODEL_FILE = './snapshots/ptb_memorize_lstm.mom0.99_lr0.01_iter_110000.caffemodel'
# # NET_FILE = './ptb_memorize_lstm_nostage_net.prototxt'
# NET_FILE = './ptb_memorize_lstm_eval_net.prototxt'

# DATA_FILE = './wmt_hdf5/buffer_10/valid_batches/batch_0.h5'
# LANGS = ['fr', 'en']
# VOCAB_FILES = ['./wmt_hdf5/buffer_10/vocabulary.%s.txt' % lang for lang in LANGS]
# MODEL_FILE = './snapshots/wmt_lstm_four_layer_2500d_400d_lr0.5_mom0.0_bs1000_buf100_iter_8000.caffemodel'
# NET_FILE = './wmt_lstm_4layer_eval_net.prototxt'
# NET_FILE = None

DATA_FILE = './wmt_hdf5/fr_io-en_o/buffer_10/train_batches/batch_0.h5'
LANGS = ['fr', 'en']
VOCAB_FILES = ['./wmt_hdf5/buffer_10/vocabulary.%s.txt' % lang for lang in LANGS]
MODEL_FILE = './snapshots/wmt_lstm_four_layer_2500d_400d_lr0.5_mom0.0_bs1000_buf100_iter_8000.caffemodel'
NET_FILE = './wmt_lstm_4layer_eval_net.newdatanames.prototxt'
# NET_FILE = None

import h5py
import numpy as np
import sys
from tabulate import tabulate

sys.path.append('../../../python/')
import caffe

if MODEL_FILE is None or NET_FILE is None:
  net = None
else:
  net = caffe.Net(NET_FILE, MODEL_FILE)
  net.set_phase_test()
  if DEVICE_ID >= 0:
    net.set_mode_gpu()
    net.set_device(DEVICE_ID)
  else:
    net.set_mode_cpu()

vocabs = []
for vocab_filename in VOCAB_FILES:
  with open(vocab_filename, 'r') as vocab_file:
#     vocabs.append(['<EOS>'] + [s.encode('ascii', errors='replace') for s in vocab_file.read().split()])
    vocabs.append([u'<EOS>'])
    for s in vocab_file.read().split():
      vocabs[-1].append(s.decode('utf8'))
#       vocabs[-1].append(s.decode('utf8').encode('ascii', errors='xmlcharrefreplace'))


def do_iteration():
  net.forward()
  data = {}
  for key in net.blobs.keys():
    data['net_' + key] = net.blobs[key].data.copy()
  data['net_predind'] = data['net_predict'].argmax(axis=1)
  data['net_pred_prob'] = np.array([data['net_probs'][i, argmax] for i, argmax
                                    in enumerate(data['net_predind'])])
  net_targets = np.array(data['net_targets'] if 'net_targets' in data else data['net_targets_en'])
  data['net_correct_prob'] = np.array([data['net_probs'][i, label] for i, label
                                       in enumerate(np.array(net_targets.reshape(-1),
                                                             dtype=np.int))])
  print 'Keys: ', data.keys()
  if 'net_accuracy' in data:
    print "Accuracy: %f" % data['net_accuracy']

  h5file = h5py.File(DATA_FILE)

  buffer_size = h5file['buffer_size'][0]

  for key in h5file.keys():
    if key == 'buffer_size': continue
    data['input_' + key] = h5file[key][:].copy()

  shaped_data = {}
  vocab_indices = {
    'input_data': 0,
    'input_data_en': 1,
    'input_data_fr': 0,
    'input_targets': 1,
    'input_targets_en': 1,
    'input_targets_fr': 0,
    'input_encoder_data': 0,
    'input_decoder_data': 1,
    'net_data': 0,
    'net_data_fr': 0,
    'net_data_en': 1,
    'net_targets': 0,
    'net_targets_fr': 0,
    'net_targets_en': 1,
    'net_encoder_data': 0,
    'net_decoder_data': 1,
    'net_predind': 1,
  }
  int_keys = set(('input_cont', 'net_cont',
#       'input_encoder_cont', 'net_encoder_cont',
#       'input_decoder_cont', 'net_decoder_cont',
      'input_encoder_to_decoder', 'net_encoder_to_decoder',
      'input_stage_indicators', 'net_stage_indicators')).union(vocab_indices.keys())
  reshape_keys = \
      set(('net_correct_prob', 'net_pred_prob')).union(int_keys)
  for key, val in data.iteritems():
    old_shape = filter(lambda s: s != 1, val.shape)
    if key in reshape_keys:
      new_shape = (old_shape[0] / buffer_size, buffer_size) + filter(lambda s: s != 1, old_shape[1:])
    else:
      new_shape = old_shape
    if key in int_keys:
      new_type = np.int
    else:
      new_type = val.dtype
    try:
      shaped_data[key] = sdata = np.array(val.reshape(new_shape), new_type).T
    except:
      import pdb; pdb.set_trace()
    if key in vocab_indices:
      shaped_data[key + '_s'] = [[vocabs[vocab_indices[key]][j] for j in sdata[i]]
                                 for i in range(len(sdata))]
  return shaped_data

num_iterations = 5
# num_iterations = 1
num_streams = 10
num_timesteps = -1
displays = [
  ('cont', 'net_cont', 'net_cont', 'input_cont'),
  ('enc', 'net_encoder_cont', 'net_encoder_cont', 'input_encoder_cont'),
  ('dec', 'net_decoder_cont', 'net_decoder_cont', 'input_decoder_cont'),
  ('enc2dec', 'net_encoder_to_decoder', 'net_encoder_to_decoder', 'input_encoder_to_decoder'),
  ('stage', 'net_stage_indicators', 'net_stage_indicators', 'input_stage_indicators'),
  ('enc_input', 'net_encoder_data_s', 'net_encoder_data', 'input_encoder_data'),
  ('dec_input', 'net_decoder_data_s', 'net_decoder_data', 'input_decoder_data'),
  ('input_fr', 'net_data_fr_s', 'net_data_fr', 'input_data_fr'),
  ('input_en', 'net_data_en_s', 'net_data_en', 'input_data_en'),
  ('target', 'net_targets_s', 'net_targets', 'input_targets'),
  ('target', 'net_targets_fr_s', 'net_targets_fr', 'input_targets_fr'),
  ('target', 'net_targets_en_s', 'net_targets_en', 'input_targets_en'),
  ('p_target', 'net_correct_prob', None, None),
  ('pred', 'net_predind_s', None, None),
  ('p_pred', 'net_pred_prob', None, None),
]
display_skip = [False] * len(displays)
all_tables = [[] for _ in range(num_streams)]
for iteration in range(num_iterations):
  shaped_data = do_iteration()
  for s, table in enumerate(all_tables):
    if num_timesteps < 0:
      num_timesteps = len(shaped_data[displays[0][2]][s])
    for t in range(num_timesteps):
      row = []
      for index, disp_keys in enumerate(displays):
        _, key, check_key_a, check_key_b = disp_keys
        if key not in shaped_data and not display_skip[index]:
          print 'Warning: key %s not found.' % key
          display_skip[index] = True
        if display_skip[index]:
          continue
        if check_key_a is not None and check_key_b is not None:
          try:
            if shaped_data[check_key_a][s][t] != \
                   shaped_data[check_key_b][s][iteration*num_timesteps + t]:
              print 'Warning: bad match'
#             assert shaped_data[check_key_a][s][t] == \
#                    shaped_data[check_key_b][s][iteration*num_timesteps + t]
          except:
            import pdb; pdb.set_trace()
        item = shaped_data[key][s][t]
        row.append(item)
      row.append('correct' if row[-1] == row[-3] else '')
      table.append(row)

headers = [d[0] for i, d in enumerate(displays) if not display_skip[i]] + ['correct']
for s, table in enumerate(all_tables):
  tabulated = tabulate(table, headers=headers)
  print u'\nStream {0} ({1} iterations):\n{2}'.format(s, num_iterations, tabulated)
