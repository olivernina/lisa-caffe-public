#!/usr/bin/env python

DEVICE_ID = 2

import h5py
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

def predict_image_caption(net, image_path):
  pred_sentence = []
  current_input_word = 0  # first input is EOS
  max_length = 50
  image = preprocess_image(net, image_path)
  while len(pred_sentence) < max_length and \
      (len(pred_sentence) == 0 or current_input_word != 0):
    cont_word = 1 if pred_sentence else 0
    cont_input = np.array([[[[cont_word]]]])
    word_input = np.array([[[[current_input_word]]]])
    outputs = net.forward(data=image, cont_sentence=cont_input,
                          input_sentence=word_input)
    probs = outputs['probs'].reshape(-1)
    most_likely_word_ind = probs.argmax()
    pred_sentence.append(most_likely_word_ind)
    # Set the next input word as the predicted word.
    current_input_word = most_likely_word_ind
  return pred_sentence

def to_html_output(outputs, vocab):
  out = ''
  for output in outputs:
    gt = vocab_inds_to_sentence(vocab, output['gt'])
    pred = vocab_inds_to_sentence(vocab, output['pred'])
    image_path = output['image_path']
    out += '<img src="%s"><br>\n' % image_path
    out += '<strong>Actual caption:</strong><br>%s<br><br>\n' % gt
    out += '<strong>Predicted caption:</strong><br>%s<br><br>\n\n' % pred
    out += '<br>' * 2
  return out

def run_pred_iter(fsg, net):
  streams = fsg.get_streams(do_padding=False, do_truncation=False)
  image_path = fsg.image_list[-1]
  num_bad_iters = 0
  try:
    pred = predict_image_caption(net, image_path)
  except Exception as e:
    num_bad_iters += 1
    print '(#%d) Warning: skipping image %s; got exception:' % \
        (num_bad_iters, image_path)
    print e
    return {}
  gt = streams['target_sentence']
  output = {'gt': gt, 'pred': pred, 'image_path': image_path}
  return output

def run_pred_iters(fsg, net, num_iterations, display_vocab=None):
  outputs = []
  for _ in range(num_iterations):
    output = run_pred_iter(fsg, net)
    if output:
      outputs.append(output)
      if display_vocab is not None:
        gt, pred = output['gt'], output['pred']
        gt_sentence = vocab_inds_to_sentence(display_vocab, gt)
        pred_sentence = vocab_inds_to_sentence(display_vocab, pred)
        print 'Ground truth caption (length %d):\n' % len(gt), gt_sentence
        print 'Predicted caption (length %d):\n' % len(pred), pred_sentence
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
      outputs = run_pred_iters(fsg, net, NUM_OUT_PER_CHUNK, display_vocab=vocab)
      html_out = to_html_output(outputs, vocab)
      if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
      html_out_filename = '%s/%s.%d.offset_%d.html' % \
          (RESULTS_DIR, dataset_name, NUM_OUT_PER_CHUNK, offset)
      html_out_file = open(html_out_filename, 'w')
      html_out_file.write(html_out)
      html_out_file.close()
      offset += NUM_OUT_PER_CHUNK
      print 'Wrote HTML output to:', html_out_filename

if __name__ == "__main__":
  main()
