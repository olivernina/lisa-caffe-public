#!/usr/bin/env python

DEVICE_ID = 2

import h5py
from math import log
import matplotlib.pyplot as plt
import numpy as np
import os
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

def predict_single_word(net, image, previous_word):
  cont = 0 if previous_word == 0 else 1
  cont_input = np.array([[[[cont]]]])
  word_input = np.array([[[[previous_word]]]])
  outputs = net.forward(data=image,
                        cont_sentence=cont_input,
                        input_sentence=word_input)
  probs = outputs['probs'].reshape(-1)
  return probs

def predict_single_word_from_all_previous(net, image, previous_words):
  probs = predict_single_word(net, image, 0)
  for index, word in enumerate(previous_words):
    probs = predict_single_word(net, image, word)
  return probs

def predict_image_caption(net, image_path, beam_size=1):
  assert beam_size >= 1
  beams = [[]]
  beams_complete = 0
  beam_log_probs = [0.]
  current_input_word = 0  # first input is EOS
  max_length = 50
  image = preprocess_image(net, image_path)
  while beams_complete < len(beams):
    expansions = []
    for beam_index, beam_log_prob, beam in \
        zip(range(len(beams)), beam_log_probs, beams):
      if beam:
        previous_word = beam[-1]
        if previous_word == 0:
          exp = {'prefix_beam_index': beam_index, 'extension': [],
                 'log_prob': beam_log_prob}
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
        extended_beam_log_prob = beam_log_prob + log(probs[ind])
        exp = {'prefix_beam_index': beam_index, 'extension': [ind],
               'log_prob': extended_beam_log_prob}
        expansions.append(exp)
    # Sort expansions in decreasing order of probability.
    expansions.sort(key=lambda expansion: -1 * expansion['log_prob'])
    expansions = expansions[:beam_size]
    beam_log_probs = [e['log_prob'] for e in expansions]
    new_beams = \
        [beams[e['prefix_beam_index']] + e['extension'] for e in expansions]
    beams_complete = 0
    for new_beam, expansion in zip(new_beams, expansions):
      if new_beam[-1] == 0: beams_complete += 1
    beams = new_beams
  return beams, beam_log_probs

def to_html_output(outputs, vocab):
  out = ''
  for output in outputs:
    gt = vocab_inds_to_sentence(vocab, output['gt'])
    image_path = output['image_path']
    out += '<img src="%s"><br>\n' % image_path
    out += '<strong>Actual caption:</strong><br>%s<br><br>\n' % gt
    out += '<strong>Predicted captions:</strong><br>'
    probs, preds = output['pred_probs'], output['preds']
    for prob, pred in zip(probs, preds):
      pred_sentence = vocab_inds_to_sentence(vocab, pred)
      out += '(%f) %s<br>' % (prob, pred_sentence)
    out += '<br>\n\n' 
    out += '<br>' * 2
  return out

def run_pred_iter(fsg, net, beam_size=1):
  streams = fsg.get_streams(do_padding=False, do_truncation=False)
  image_path = fsg.image_list[-1]
  num_bad_iters = 0
  try:
    preds, pred_probs = predict_image_caption(net, image_path,
        beam_size=beam_size)
  except Exception as e:
    num_bad_iters += 1
    print '(#%d) Warning: skipping image %s; got exception:' % \
        (num_bad_iters, image_path)
    print e
    return {}
  gt = streams['target_sentence']
  output = {'gt': gt, 'preds': preds, 'pred_probs': pred_probs,
            'image_path': image_path}
  return output

def run_pred_iters(fsg, net, num_iterations, beam_size=1, display_vocab=None):
  outputs = []
  for _ in range(num_iterations):
    output = run_pred_iter(fsg, net, beam_size=beam_size)
    if not output: continue
    outputs.append(output)
    if display_vocab is not None:
      gt, preds, probs = output['gt'], output['preds'], output['pred_probs']
      gt_sentence = vocab_inds_to_sentence(display_vocab, gt)
      print 'Ground truth caption (length %d):\n' % len(gt), gt_sentence
      for pred, prob in zip(preds, probs):
        pred_sentence = vocab_inds_to_sentence(display_vocab, pred)
        print 'Predicted caption, log_p = %f (length %d):\n' % (prob, len(pred)), pred_sentence
      print ''
  return outputs

def main():
  NET_FILE = './alexnet_to_lstm_net.deploy.prototxt'
  MODEL_FILE = './snapshots/coco_flickr_30k_alexnet_to_lstm_4layer_lr0.1_mom_0.9_iter_37000.caffemodel'

  # Set up the net.
  net = caffe.Net(NET_FILE, MODEL_FILE)
  channel_mean = np.array([104, 117, 123])[:, np.newaxis, np.newaxis]
  net.set_mean('data', channel_mean, mode='channel')
  net.set_channel_swap('data', (2, 1, 0))
  net.set_phase_test()
  if DEVICE_ID >= 0:
    net.set_mode_gpu()
    net.set_device(DEVICE_ID)
  else:
    net.set_mode_cpu()

  RESULTS_DIR = './html_results'
  BEAM_SIZE = 1
  NUM_CHUNKS = 8
  NUM_OUT_PER_CHUNK = 25

  _, _, val_datasets = DATASETS[1]
  flickr_dataset = [val_datasets[0]]
  coco_dataset = [val_datasets[1]]
  datasets = [flickr_dataset, coco_dataset]
  dataset_names = ['flickr', 'coco']

  for dataset, dataset_name in zip(datasets, dataset_names):
    fsg = FlickrSequenceGenerator(dataset, VOCAB_FILE, 0, align=False)
    eos_string = '<EOS>'
    vocab = [eos_string] + fsg.vocabulary_inverted
    offset = 0
    for c in range(NUM_CHUNKS):
      outputs = run_pred_iters(fsg, net, NUM_OUT_PER_CHUNK,
          beam_size=BEAM_SIZE, display_vocab=vocab)
      html_out = to_html_output(outputs, vocab)
      if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
      html_out_filename = '%s/%s.beam_%d.%d.offset_%d.html' % \
          (RESULTS_DIR, dataset_name, BEAM_SIZE, NUM_OUT_PER_CHUNK, offset)
      html_out_file = open(html_out_filename, 'w')
      html_out_file.write(html_out)
      html_out_file.close()
      offset += NUM_OUT_PER_CHUNK
      print 'Wrote HTML output to:', html_out_filename

if __name__ == "__main__":
  main()
