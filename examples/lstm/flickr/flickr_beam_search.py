#!/usr/bin/env python

DEVICE_ID = 2

import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys

sys.path.append('../../../python/')
import caffe

from flickr_to_hdf5_data import *

def vocab_inds_to_sentence(vocab, inds):
  sentence = ' '.join([vocab[i] for i in inds])
  # Capitalize first character.
  sentence = sentence[0].upper() + sentence[1:]
  # Replace <EOS> with '.', or append '...'.
  if sentence.endswith(' ' + vocab[0]):
    sentence = sentence[:-(len(vocab[0]) + 1)] + '.'
  else:
    sentence += '...'
  return sentence
  
def preprocess_image(net, image_path):
  image = plt.imread(image_path)
  print 'Read image with shape %s, range (%f, %f) from: %s' % \
      (image.shape, image.min(), image.max(), image_path)
  # Crop the center 224 / 256 of the image.
  crop_edge_ratio = (256. - 224.) / 256. / 2
  ch = int(image.shape[0] * crop_edge_ratio + 0.5)
  cw = int(image.shape[1] * crop_edge_ratio + 0.5)
  cropped_image = image[ch:-ch, cw:-cw]
  preprocessed_image = net.preprocess('data', cropped_image)[np.newaxis]
  print 'Preprocessed image has shape %s, range (%f, %f)' % \
      (preprocessed_image.shape,
       preprocessed_image.min(),
       preprocessed_image.max())
  return preprocessed_image

def image_to_descriptor(net, image_path, output_name='fc8'):
  image = preprocess_image(net, image_path)
  net.forward(data=image)
  output_blob = net.blobs[output_name].data.copy()
  return output_blob
  
def predict_single_word(net, image_features, previous_word, output='probs'):
  cont = 0 if previous_word == 0 else 1
  cont_input = np.array([[[[cont]]]])
  word_input = np.array([[[[previous_word]]]])
  net.forward(image_features=image_features,
      cont_sentence=cont_input, input_sentence=word_input)
  output_preds = net.blobs[output].data.reshape(-1)
  return output_preds

def predict_single_word_from_all_previous(net, image, previous_words):
  probs = predict_single_word(net, image, 0)
  for index, word in enumerate(previous_words):
    probs = predict_single_word(net, image, word)
  return probs

# Strategy must be either 'beam' or 'sample'.
# If 'beam', do a max likelihood beam search with beam size num_samples.
# Otherwise, sample with temperature temp.
def predict_image_caption(net, image, strategy={'type': 'beam'}):
  assert 'type' in strategy
  assert strategy['type'] in ('beam', 'sample')
  if strategy['type'] == 'beam':
    return predict_image_caption_beam_search(net, image, strategy)
  num_samples = strategy['num'] if 'num' in strategy else 1
  samples = []
  sample_probs = []
  for _ in range(num_samples):
    sample, sample_prob = sample_image_caption(net, image, strategy)
    samples.append(sample)
    sample_probs.append(sample_prob)
  return samples, sample_probs

def softmax(softmax_inputs, temp):
  exp_inputs = np.exp(temp * softmax_inputs)
  exp_inputs_sum = exp_inputs.sum()
  if math.isnan(exp_inputs_sum):
    return exp_inputs * float('nan')
  elif math.isinf(exp_inputs_sum):
    assert exp_inputs_sum > 0  # should not be -inf
    return np.zeros_like(exp_inputs)
  eps_sum = 1e-8
  return exp_inputs / max(exp_inputs_sum, eps_sum)

def random_choice_from_probs(softmax_inputs, temp):
  probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?

def sample_image_caption(net, image, strategy, net_output='predict'):
  sentence = []
  probs = []
  eps_prob = 1e-8
  temp = strategy['temp'] if 'temp' in strategy else 1.0
  while not sentence or sentence[-1] != 0:
    previous_word = sentence[-1] if sentence else 0
    softmax_inputs = \
        predict_single_word(net, image, previous_word, output=net_output)
    word = random_choice_from_probs(softmax_inputs, temp)
    sentence.append(word)
    probs.append(softmax(softmax_inputs, 1.0)[word])
  return sentence, probs

def predict_image_caption_beam_search(net, image, strategy):
  beam_size = strategy['beam_size'] if 'beam_size' in strategy else 1
  assert beam_size >= 1
  beams = [[]]
  beams_complete = 0
  beam_probs = [[]]
  beam_log_probs = [0.]
  current_input_word = 0  # first input is EOS
  max_length = 50
  while beams_complete < len(beams):
    expansions = []
    for beam_index, beam_log_prob, beam in \
        zip(range(len(beams)), beam_log_probs, beams):
      if beam:
        previous_word = beam[-1]
        if previous_word == 0:
          exp = {'prefix_beam_index': beam_index, 'extension': [],
                 'prob_extension': [], 'log_prob': beam_log_prob}
          expansions.append(exp)
          continue  # don't expand this beam; it was already ended with an EOS
      else:
        previous_word = 0  # EOS is first word
      if beam_size == 1:
        probs = predict_single_word(net, image, previous_word)
      else:
        probs = predict_single_word_from_all_previous(net, image, beam)
      expansion_inds = probs.argsort()[-beam_size:]
      for ind in expansion_inds:
        prob = probs[ind]
        extended_beam_log_prob = beam_log_prob + math.log(prob)
        exp = {'prefix_beam_index': beam_index, 'extension': [ind],
               'prob_extension': [prob], 'log_prob': extended_beam_log_prob}
        expansions.append(exp)
    # Sort expansions in decreasing order of probability.
    expansions.sort(key=lambda expansion: -1 * expansion['log_prob'])
    expansions = expansions[:beam_size]
    new_beams = \
        [beams[e['prefix_beam_index']] + e['extension'] for e in expansions]
    new_beam_probs = \
        [beam_probs[e['prefix_beam_index']] + e['prob_extension'] for e in expansions]
    beam_log_probs = [e['log_prob'] for e in expansions]
    beams_complete = 0
    for new_beam, expansion in zip(new_beams, expansions):
      if new_beam[-1] == 0: beams_complete += 1
    beams, beam_probs = new_beams, new_beam_probs
  return beams, beam_probs

def run_pred_iter(net, image, strategies=[{'type': 'beam'}]):
  outputs = []
  for strategy in strategies:
    captions, probs = predict_image_caption(net, image, strategy=strategy)
    for caption, prob in zip(captions, probs):
      output = {}
      output['caption'] = caption
      output['prob'] = prob
      output['gt'] = False
      output['source'] = strategy
      outputs.append(output)
  return outputs

def score_caption(net, image, caption, is_gt=True, caption_source='gt'):
  output = {}
  output['caption'] = caption
  output['gt'] = is_gt
  output['source'] = caption_source
  output['prob'] = []
  probs = predict_single_word(net, image, 0)
  for word in caption:
    output['prob'].append(probs[word])
    probs = predict_single_word(net, image, word)
  return output

def next_image_gt_pair(fsg):
  streams = fsg.get_streams(do_padding=False, do_truncation=False)
  image_path = fsg.image_list[-1]
  gt = streams['target_sentence']
  return image_path, gt

def gen_stats(prob):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += math.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  try:
    stats['perplex'] = math.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = math.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  return stats

def run_pred_iters(fsg, image_net, pred_net, num_iterations,
                   strategies=[{'type': 'beam'}], display_vocab=None):
  outputs = {}
  num_pairs = 0
  last_image_path = ''
  while num_pairs < num_iterations:
    image_path, gt_caption = next_image_gt_pair(fsg)
    num_pairs += 1
    did_predictions = False
    if last_image_path != image_path:
      image_features = image_to_descriptor(image_net, image_path)
    if image_path not in outputs:
      did_predictions = True
      outputs[image_path] = run_pred_iter(pred_net, image_features, strategies=strategies)
    outputs[image_path].append(score_caption(pred_net, image_features, gt_caption))
    if display_vocab is not None:
      if did_predictions:
        display_outputs = outputs[image_path]
      else:
        display_outputs = [outputs[image_path][-1]]
      for output in display_outputs:
        caption, prob, gt, source = \
            output['caption'], output['prob'], output['gt'], output['source']
        caption_string = vocab_inds_to_sentence(display_vocab, caption)
        if gt:
          tag = 'Actual'
        else:
          tag = 'Generated'
        stats = gen_stats(prob)
        print '%s caption (length %d, log_p = %f, log_p_word = %f):\n%s' % \
            (tag, stats['length'], stats['log_p'], stats['log_p_word'], caption_string)
  return outputs

def to_html_row(columns, header=False):
  out = '<tr>'
  for column in columns:
    if header: out += '<th>'
    else: out += '<td>'
    try:
      if int(column) < 1e8 and int(column) == float(column):
        out += '%d' % column
      else:
        out += '%0.04f' % column
    except:
      out += '%s' % column
    if header: out += '</th>'
    else: out += '</td>'
  out += '</tr>'
  return out

def to_html_output(outputs, vocab):
  out = ''
  for image_path, captions in outputs.iteritems():
    for c in captions: c['stats'] = gen_stats(c['prob'])
    # Sort captions by log probability.
    captions.sort(key=lambda c: -c['stats']['log_p_word'])
    out += '<img src="%s"><br>\n' % image_path
    out += '<table border="1">\n'
    column_names = ('Source', '#Words', 'Perplexity/Word', 'Caption')
    out += '%s\n' % to_html_row(column_names, header=True)
    for c in captions:
      caption, gt, source, stats = \
          c['caption'], c['gt'], c['source'], c['stats']
      caption_string = vocab_inds_to_sentence(vocab, caption)
      if gt:
        source = 'ground truth'
        caption_string = '<em>%s</em>' % caption_string
      else:
        if source['type'] == 'beam':
          source = 'beam (size %d)' % source['beam_size']
        elif source['type'] == 'sample':
          source = 'sample (temp %f)' % source['temp']
        else:
          raise Exception('Unknown type: %s' % source['type'])
        caption_string = '<strong>%s</strong>' % caption_string
      columns = (source, stats['length'] - 1,
                 stats['perplex_word'], caption_string)
      out += '%s\n' % to_html_row(columns)
    out += '</table>\n'
    out += '<br>\n\n' 
    out += '<br>' * 2
  return out

def main():
  # NET_FILE = './alexnet_to_lstm_net.deploy.prototxt'
  IMAGE_NET_FILE = './alexnet_to_lstm_net.image_to_fc8.deploy.prototxt'
  LSTM_NET_FILE = './alexnet_to_lstm_net.word_to_preds.deploy.prototxt'
  MODEL_FILE = './snapshots/coco_flickr_30k_alexnet_to_lstm_4layer_lr0.1_mom_0.9_iter_37000.caffemodel'

  # Set up the net.
  # net = caffe.Net(NET_FILE, MODEL_FILE)
  image_net = caffe.Net(IMAGE_NET_FILE, MODEL_FILE)
  lstm_net = caffe.Net(LSTM_NET_FILE, MODEL_FILE)
  nets = [image_net, lstm_net]
  channel_mean = np.array([104, 117, 123])[:, np.newaxis, np.newaxis]
  image_net.set_mean('data', channel_mean, mode='channel')
  image_net.set_channel_swap('data', (2, 1, 0))
  image_net.set_phase_test()
  for net in nets:
    if DEVICE_ID >= 0:
      net.set_mode_gpu()
      net.set_device(DEVICE_ID)
    else:
      net.set_mode_cpu()

  RESULTS_DIR = './html_multiresults'
  STRATEGIES = [
    {'type': 'sample', 'temp': 0.75, 'num': 3},
    {'type': 'sample', 'temp': 1.0, 'num': 3},
    {'type': 'sample', 'temp': 3.0, 'num': 3},
    {'type': 'sample', 'temp': 5.0, 'num': 3},
    {'type': 'beam', 'beam_size': 1},
    {'type': 'beam', 'beam_size': 3},
    {'type': 'beam', 'beam_size': 5},
  ]
  NUM_CHUNKS = 5
  NUM_OUT_PER_CHUNK = 10

  _, _, val_datasets = DATASETS[1]
  flickr_dataset = [val_datasets[0]]
  coco_dataset = [val_datasets[1]]
  datasets = [flickr_dataset, coco_dataset]
  dataset_names = ['coco', 'flickr']

  for dataset, dataset_name in zip(datasets, dataset_names):
    fsg = FlickrSequenceGenerator(dataset, VOCAB_FILE, 0, align=False, shuffle=False)
    eos_string = '<EOS>'
    vocab = [eos_string] + fsg.vocabulary_inverted
    offset = 0
    for c in range(NUM_CHUNKS):
      html_out_filename = '%s/%s.%s.%d_to_%d.html' % \
          (RESULTS_DIR, dataset_name, 'multistrategy', offset, offset + NUM_OUT_PER_CHUNK)
      if os.path.exists(html_out_filename):
        raise Exception('HTML out path exists: %s' % html_out_filename)
      outputs = run_pred_iters(fsg, image_net, lstm_net, NUM_OUT_PER_CHUNK,
          strategies=STRATEGIES, display_vocab=vocab)
      html_out = to_html_output(outputs, vocab)
      if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
      html_out_file = open(html_out_filename, 'w')
      html_out_file.write(html_out)
      html_out_file.close()
      offset += NUM_OUT_PER_CHUNK
      print 'Wrote HTML output to:', html_out_filename

if __name__ == "__main__":
  main()
