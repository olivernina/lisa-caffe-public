#!/usr/bin/env python

DEVICE_ID = 3
# DATA_FILE = './ptb_hdf5/train_batches/batch_0.h5'
# VOCAB_FILE = './ptb_hdf5/ptb_vocabulary.txt'
# MODEL_FILE = './snapshots/ptb_lstm_accum_grads_without_buggy_slicelayer_mom0.99_iter_9000.caffemodel'
# NET_FILE = './ptb_lstm_eval_net.prototxt'

# DATA_FILE = './ptb_memorize_hdf5_buffer_20/train_batches/batch_0.h5'
# VOCAB_FILE = './ptb_memorize_hdf5_buffer_20/ptb_vocabulary.txt'
# MODEL_FILE = './snapshots/ptb_memorize_lstm.mom0.99_lr0.01_iter_110000.caffemodel'
# # NET_FILE = './ptb_memorize_lstm_nostage_net.prototxt'
# NET_FILE = './ptb_memorize_lstm_eval_net.prototxt'

# DATA_FILE = './wmt_hdf5/buffer_100/train_batches/batch_0.h5'
# LANGS = ['fr', 'en']
# VOCAB_FILES = ['./wmt_hdf5/buffer_100/wmt_vocabulary.%s.txt' % lang for lang in LANGS]
# MODEL_FILE = './snapshots/wmt_lstm_four_layer_enc_dec_lr0.7_mom0.0_iter_110000.caffemodel'
# NET_FILE = './wmt_dec_enc_lstm_4layer_eval_net.prototxt'

# DATA_FILE = './wmt_hdf5/fr_io-en_o/buffer_10/train_batches/batch_0.h5'
# LANGS = ['fr', 'en']
# VOCAB_FILES = ['./wmt_hdf5/buffer_10/vocabulary.%s.txt' % lang for lang in LANGS]
# MODEL_FILE = './snapshots/wmt_lstm_four_layer_2500d_400d_lr0.5_mom0.0_bs1000_buf100_iter_8000.caffemodel'
# NET_FILE = './wmt_lstm_4layer_eval_net.newdatanames.prototxt'
# NET_FILE = None

# DATA_FILE = './wmt_char_hdf5/fr_io-en_io/buffer_100/train_batches/batch_0.h5'
# LANGS = ['fr', 'en']
# # VOCAB_FILES = ['./wmt_char_hdf5/vocabs/vocabulary.%s.txt' % lang for lang in LANGS]
# VOCAB_FILES = [None, None]
# MODEL_FILE = './snapshots/wmt_char_lr0.1_mom0.9_iter_38000.caffemodel'
# NET_FILE = './wmt_char_lstm_4layer_eval_net.prototxt'

# DATA_FILE = './wmt_char_hdf5/fr_io-en_io/buffer_100/train_batches/batch_0.h5'
# LANGS = ['fr', 'en']
# # VOCAB_FILES = ['./wmt_char_hdf5/vocabs/vocabulary.%s.txt' % lang for lang in LANGS]
# VOCAB_FILES = [None, None]
# # MODEL_FILE = './snapshots/wmt_char_memorize_1layer_lr0.5_adagrad_iter_22000.caffemodel'
# MODEL_FILE = './snapshots/wmt_char_memorize_1layer_lr0.1_mom_0.9_nesterov_iter_54000.caffemodel'
# NET_FILE = './wmt_char_lstm_1layer_eval_net.prototxt'

# DATA_FILE = './wmt_hdf5/fr_io-en_o/buffer_20/train_batches/batch_0.h5'
# LANGS = ['fr', 'en']
# VOCAB_FILES = ['./wmt_hdf5/fr_io-en_o/buffer_20/vocabulary.%s.txt' % lang for lang in LANGS]
# MODEL_FILE = './snapshots/wmt_fixed_lstm_two_layer_input_skips_1000d_400d_lr0.1_mom0.9_bs400_buf20_iter_82000.caffemodel'
# NET_FILE = './wmt_lstm_2layer_inputskips_eval_net.prototxt'

DATA_FILE = './wmt_hdf5/fr_o-en_i/buffer_100/train_batches/batch_0.h5'
LANGS = ['fr', 'en']
VOCAB_FILES = ['./wmt_hdf5/fr_o-en_i/buffer_100/vocabulary.%s.txt' % lang for lang in LANGS]
MODEL_FILE = './snapshots/wmt_en2fr_lstm_four_layer_enc_dec_lr0.7_mom0.0_bs2000_embed1000_undoseqlengthdiv_iter_10000.caffemodel'
NET_FILE = './wmt_dec_enc_lstm_4layer_net.en_to_fr.prototxt'

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
  vocabs.append([u'<EOS>'])
  if vocab_filename is None:
    vocabs[-1].append([u'<unk>'])
    vocabs[-1] += [unichr(c) for c in range(256)]
    continue
  with open(vocab_filename, 'r') as vocab_file:
#     vocabs.append(['<EOS>'] + [s.encode('ascii', errors='replace') for s in vocab_file.read().split()])
    for s in vocab_file.read().split():
      vocabs[-1].append(s.decode('utf8', errors='replace'))
#       vocabs[-1].append(s.decode('utf8', errors='replace').encode('ascii', errors='replace'))
#       vocabs[-1].append(s.decode('utf8'))
#       vocabs[-1].append(s.decode('utf8').encode('ascii', errors='xmlcharrefreplace'))


def do_iteration():
  net.forward()
  data = {}
  for key in net.blobs.keys():
    data['net_' + key] = net.blobs[key].data.copy()
    if 'predict' in key:
      predind_key = key.replace('predict', 'predind')
      data['net_' + predind_key] = data['net_' + key].argmax(axis=1)
  for key in data.keys():
    if 'predind' in key:
      probs_key = key.replace('predind', 'probs')
      if probs_key in data:
        out_key = probs_key.replace('probs', 'pred_prob')
        data[out_key] = np.array([data[probs_key][i, argmax] for i, argmax
                                  in enumerate(data[key])])
        targets_key = key.replace('predind', 'targets')
        net_targets = np.array(data[targets_key])
        out_key = probs_key.replace('probs', 'correct_prob')
        data[out_key] = np.array([data[probs_key][i, label] for i, label
                                 in enumerate(np.array(net_targets.reshape(-1), dtype=np.int))])
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
#     'net_targets': 1,
    'net_targets': 0,
    'net_targets_fr': 0,
    'net_targets_en': 1,
    'net_encoder_data': 0,
    'net_decoder_data': 1,
    'net_predind': 0,
    'net_predind_en': 1,
    'net_predind_fr': 0,
  }
  int_keys = set(('input_cont', 'net_cont',
      'input_cont_en', 'net_cont_en',
      'input_cont_fr', 'net_cont_fr',
      'input_encoder_cont', 'net_encoder_cont',
      'input_decoder_cont', 'net_decoder_cont',
      'input_encoder_to_decoder', 'net_encoder_to_decoder',
      'input_stage_indicators', 'net_stage_indicators')).union(vocab_indices.keys())
  reshape_keys = \
      set(('net_correct_prob', 'net_pred_prob',
           'net_correct_prob_en', 'net_correct_prob_fr',
           'net_pred_prob_en', 'net_pred_prob_fr',
           )).union(int_keys)
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

num_iterations = 10
# num_iterations = 1
num_streams = 100
num_timesteps = -1
displays = [
  ('cont', 'net_cont', 'net_cont', 'input_cont'),
  ('cont_en', 'net_cont_en', 'net_cont_en', 'input_cont_en'),
  ('cont_fr', 'net_cont_fr', 'net_cont_fr', 'input_cont_fr'),
  ('enc', 'net_encoder_cont', 'net_encoder_cont', 'input_encoder_cont'),
  ('dec', 'net_decoder_cont', 'net_decoder_cont', 'input_decoder_cont'),
  ('enc2dec', 'net_encoder_to_decoder', 'net_encoder_to_decoder', 'input_encoder_to_decoder'),
  ('stage', 'net_stage_indicators', 'net_stage_indicators', 'input_stage_indicators'),
  ('enc_input', 'net_encoder_data_s', 'net_encoder_data', 'input_encoder_data'),
  ('dec_input', 'net_decoder_data_s', 'net_decoder_data', 'input_decoder_data'),
  ('input_fr', 'net_data_fr_s', 'net_data_fr', 'input_data_fr'),
  ('input_en', 'net_data_en_s', 'net_data_en', 'input_data_en'),
  ('target', 'net_targets_s', 'net_targets', 'input_targets'),
  ('target_fr', 'net_targets_fr_s', 'net_targets_fr', 'input_targets_fr'),
  ('target_en', 'net_targets_en_s', 'net_targets_en', 'input_targets_en'),
  ('p_target', 'net_correct_prob', None, None),
  ('p_target_fr', 'net_correct_prob_fr_s', 'net_correct_prob_fr', 'input_correct_prob_fr'),
  ('p_target_en', 'net_correct_prob_en_s', 'net_correct_prob_en', 'input_correct_prob_en'),
  ('pred', 'net_predind_s', None, None),
  ('pred_en', 'net_predind_en_s', None, None),
  ('pred_fr', 'net_predind_fr_s', None, None),
  ('p_pred', 'net_pred_prob', None, None),
  ('p_pred_en', 'net_pred_prob_en', None, None),
  ('p_pred_fr', 'net_pred_prob_fr', None, None),
]
display_skip = [False] * len(displays)
all_tables = [[] for _ in range(num_streams)]
for iteration in range(num_iterations):
  print 'Iteration %d' % iteration
  shaped_data = do_iteration()
  for s, table in enumerate(all_tables):
    if num_timesteps < 0:
      num_timesteps = None
      for display in displays:
        if display[2] in shaped_data:
          try:
            num_timesteps = len(shaped_data[display[2]][s])
            break
          except:
            pass
      assert num_timesteps is not None
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
        try:
          item = shaped_data[key][s][t]
        except:
          import pdb; pdb.set_trace()
        row.append(item)
      row.append('correct' if row[-1] == row[-3] else '')
      table.append(row)

try:
  headers = [d[0] for i, d in enumerate(displays) if not display_skip[i]] + ['correct']
  for s, table in enumerate(all_tables):
    tabulated = tabulate(table, headers=headers)
    print '\nStream {0} ({1} iterations):\n{2}'.format(s, num_iterations, tabulated.encode('utf8', errors='replace'))
except:
  import pdb; pdb.set_trace()
