#!/usr/bin/env python

DEVICE_ID = 1

from collections import OrderedDict
import cPickle as pickle
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

def random_choice_from_probs(softmax_inputs, temp=1.0, already_softmaxed=False):
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1.0
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?

def sample_image_caption(net, image, strategy, net_output='predict', max_length=50):
  sentence = []
  probs = []
  eps_prob = 1e-8
  temp = strategy['temp'] if 'temp' in strategy else 1.0
  if max_length < 0: max_length = float('inf')
  while len(sentence) < max_length and (not sentence or sentence[-1] != 0):
    previous_word = sentence[-1] if sentence else 0
    softmax_inputs = \
        predict_single_word(net, image, previous_word, output=net_output)
    word = random_choice_from_probs(softmax_inputs, temp)
    sentence.append(word)
    probs.append(softmax(softmax_inputs, 1.0)[word])
  return sentence, probs

def predict_image_caption_beam_search(net, image, strategy, max_length=50):
  beam_size = strategy['beam_size'] if 'beam_size' in strategy else 1
  assert beam_size >= 1
  beams = [[]]
  beams_complete = 0
  beam_probs = [[]]
  beam_log_probs = [0.]
  current_input_word = 0  # first input is EOS
  while beams_complete < len(beams):
    expansions = []
    for beam_index, beam_log_prob, beam in \
        zip(range(len(beams)), beam_log_probs, beams):
      if beam:
        previous_word = beam[-1]
        if len(beam) >= max_length or previous_word == 0:
          exp = {'prefix_beam_index': beam_index, 'extension': [],
                 'prob_extension': [], 'log_prob': beam_log_prob}
          expansions.append(exp)
          # Don't expand this beam; it was already ended with an EOS,
          # or is the max length.
          continue
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
    for beam in new_beams:
      if beam[-1] == 0 or len(beam) >= max_length: beams_complete += 1
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

def all_image_gt_pairs(fsg):
  data = OrderedDict()
  prev_image_path = None
  while True:
    image_path, gt = next_image_gt_pair(fsg)
    if image_path in data:
      if image_path != prev_image_path:
        break
      data[image_path].append(gt)
    else:
      data[image_path] = [gt]
    prev_image_path = image_path
  print 'Found %d images with %d captions' % (len(data.keys()), len(data.values()))
  return data

def gen_stats(prob, normalizer=None):
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
  if normalizer is not None:
    norm_stats = gen_stats(normalizer)
    stats['normed_perplex'] = \
        stats['perplex'] / norm_stats['perplex']
    stats['normed_perplex_word'] = \
        stats['perplex_word'] / norm_stats['perplex_word']
  return stats

def run_pred_iters(image_net, pred_net, images, image_gt_pairs,
                   strategies=[{'type': 'beam'}], display_vocab=None):
  outputs = OrderedDict()
  num_pairs = 0
  descriptor_image_path = ''
  for image_path in images:
    gt_captions = image_gt_pairs[image_path]
    assert image_path not in outputs
    num_pairs += 1
    if descriptor_image_path != image_path:
      try:
        image_features = image_to_descriptor(image_net, image_path)
      except:
        print 'WARNING: could not compute image descriptor for image, skipping: %s' % image_path
        continue
      descriptor_image_path = image_path
    outputs[image_path] = \
        run_pred_iter(pred_net, image_features, strategies=strategies)
    for gt_caption in gt_captions:
      outputs[image_path].append(
          score_caption(pred_net, image_features, gt_caption))
    if display_vocab is not None:
      for output in outputs[image_path]:
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
    for c in captions:
      if not 'stats' in c:
        c['stats'] = gen_stats(c['prob'])
    # Sort captions by log probability.
    if 'normed_perplex' in captions[0]['stats']:
      captions.sort(key=lambda c: c['stats']['normed_perplex'])
    else:
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
        if 'correct' in c:
          caption_string = '<font color="%s">%s</font>' % \
              ('green' if c['correct'] else 'red', caption_string)
        else:
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
  out.replace('<unk>', 'UNK')  # sanitize...
  return out

def retrieval_image_list(dataset, cache_dir):
  image_list_filename = '%s/image_paths.txt' % cache_dir
  if os.path.exists(image_list_filename):
    with open(image_list_filename, 'r') as image_list_file:
      image_paths = [i.strip() for i in image_list_file.readlines()]
      assert set(image_paths) == set(dataset.keys())
  else:
    image_paths = dataset.keys()
    with open(image_list_filename, 'w') as image_list_file:
      image_list_file.write('\n'.join(image_paths) + '\n')
  return image_paths

def compute_descriptors(net, image_list, output_name='fc8'):
  batch = np.zeros_like(net.blobs['data'].data)
  batch_shape = batch.shape
  batch_size = batch_shape[0]
  descriptors_shape = (len(image_list), ) + net.blobs[output_name].data.shape[1:]
  descriptors = np.zeros(descriptors_shape)
  for batch_start_index in range(0, len(image_list), batch_size):
    batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
    for batch_index, image_path in enumerate(batch_list):
      batch[batch_index:(batch_index + 1)] = preprocess_image(net, image_path)
    print 'Computing descriptors for images %d-%d of %d' % \
        (batch_start_index, batch_start_index + batch_size - 1, len(image_list))
    net.forward(data=batch)
    print 'Done'
    descriptors[batch_start_index:(batch_start_index + batch_size)] = \
        net.blobs[output_name].data
  return descriptors

def retrieval_descriptors(net, image_list, cache_dir):
  descriptor_filename = '%s/descriptors.npz' % cache_dir
  if os.path.exists(descriptor_filename):
    descriptors = np.load(descriptor_filename)['descriptors']
  else:
    descriptors = compute_descriptors(net, image_list)
    np.savez_compressed(descriptor_filename, descriptors=descriptors)
  return descriptors

def retrieval_caption_list(dataset, image_list, cache_dir):
  caption_list_filename = '%s/captions.pkl' % cache_dir
  if os.path.exists(caption_list_filename):
    with open(caption_list_filename, 'rb') as caption_list_file:
      captions = pickle.load(caption_list_file)
  else:
    captions = []
    for image in image_list:
      for caption in dataset[image]:
        captions.append({'source_image': image, 'caption': caption})
    # Sort by length for performance.
    captions.sort(key=lambda c: len(c['caption']))
    with open(caption_list_filename, 'wb') as caption_list_file:
      pickle.dump(captions, caption_list_file)
  return captions

def sample_captions(net, image_features,
    output_name='probs', caption_source='sample'):
  cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
  word_input = np.zeros_like(net.blobs['input_sentence'].data)
  batch_size = image_features.shape[0]
  outputs = []
  output_captions = [[] for b in range(batch_size)]
  output_probs = [[] for b in range(batch_size)]
  caption_index = 0
  num_done = 0
  while num_done < batch_size:
    if caption_index == 0:
      cont_input[:] = 0
    elif caption_index == 1:
      cont_input[:] = 1
    if caption_index == 0:
      word_input[:] = 0
    else:
      for index in range(batch_size):
        word_input[index] = \
            output_captions[index][caption_index - 1] if \
            caption_index <= len(output_captions[index]) else 0
    net.forward(image_features=image_features,
        cont_sentence=cont_input, input_sentence=word_input)
    net_output_probs = net.blobs[output_name].data
    for index in range(batch_size):
      # If the caption is empty, or non-empty but the last word isn't EOS,
      # predict another word.
      if not output_captions[index] or output_captions[index][-1] != 0:
        next_word_sample = random_choice_from_probs(net_output_probs[index],
                                                    already_softmaxed=True)
        output_captions[index].append(next_word_sample)
        output_probs[index].append(net_output_probs[index, next_word_sample])
        if next_word_sample == 0: num_done += 1
    print '%d/%d done after word %d' % (num_done, batch_size, caption_index)
    caption_index += 1
  for prob, caption in zip(output_probs, output_captions):
    output = {}
    output['caption'] = caption
    output['prob'] = prob
    output['gt'] = False
    output['source'] = caption_source
    outputs.append(output)
  return outputs

def score_captions(net, image_index, descriptor, captions,
                   output_name='probs', caption_source='gt'):
  cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
  word_input = np.zeros_like(net.blobs['input_sentence'].data)
  image_features = np.zeros_like(net.blobs['image_features'].data)
  batch_size = image_features.shape[0]
  assert descriptor.shape == image_features.shape[1:]
  for index in range(batch_size):
    image_features[index] = descriptor
  outputs = []
  for batch_start_index in range(0, len(captions), batch_size):
    caption_batch = captions[batch_start_index:(batch_start_index + batch_size)]
    caption_index = 0
    probs_batch = [[] for b in range(batch_size)]
    num_done = 0
    while num_done < batch_size:
      if caption_index == 0:
        cont_input[:] = 0
      elif caption_index == 1:
        cont_input[:] = 1
      for index, caption in enumerate(caption_batch):
        word_input[index] = \
            caption['caption'][caption_index - 1] if \
            0 < caption_index < len(caption['caption']) else 0
      net.forward(image_features=image_features,
          cont_sentence=cont_input, input_sentence=word_input)
      output_probs = net.blobs[output_name].data
      for index, probs, caption in \
          zip(range(batch_size), probs_batch, caption_batch):
        if caption_index == len(caption['caption']) - 1:
          num_done += 1
        if caption_index < len(caption['caption']):
          word = caption['caption'][caption_index]
          probs.append(output_probs[index, word].reshape(-1)[0])
      print '(Image %d) Computed probs for word %d of captions %d-%d (%d done)' % \
          (image_index, caption_index, batch_start_index,
           batch_start_index + batch_size - 1, num_done)
      caption_index += 1
    for prob, caption in zip(probs_batch, caption_batch):
      output = {}
      output['caption'] = caption['caption']
      output['prob'] = prob
      output['gt'] = True
      output['source'] = caption_source
      outputs.append(output)
  return outputs

def retrieval_caption_scores(net, index, descriptor, captions, cache_dir,
                             output_name='probs', caption_source='gt'):
  caption_scores_dir = '%s/caption_scores' % cache_dir
  if not os.path.exists(caption_scores_dir):
    os.makedirs(caption_scores_dir)
  caption_scores_filename  = '%s/scores_image_%06d.pkl' % \
      (caption_scores_dir, index)
  if os.path.exists(caption_scores_filename):
    with open(caption_scores_filename, 'rb') as caption_scores_file:
      outputs = pickle.load(caption_scores_file)
  else:
    outputs = score_captions(net, index, descriptor, captions,
        output_name=output_name, caption_source=caption_source)
    with open(caption_scores_filename, 'wb') as caption_scores_file:
      pickle.dump(outputs, caption_scores_file)
  return outputs

def retrieval_caption_stats(caption_list, image_path, caption_scores,
    image_index, cache_dir, mean_caption_scores):
  caption_score_dir = '%s/final_caption_scores' % cache_dir
  if not os.path.exists(caption_score_dir):
    os.makedirs(caption_score_dir)
  caption_score_filename = '%s/final_caption_scores_%d.pkl' % \
      (caption_score_dir, image_index)
  if os.path.exists(caption_score_filename):
    with open(caption_score_filename, 'rb') as caption_score_file:
      caption_scores = pickle.load(caption_score_file)
  else:
    num_correct = 0
    for caption, score, mean_score in \
        zip(caption_list, caption_scores, mean_caption_scores):
      assert caption['caption'] == score['caption']
      assert mean_score['caption'] == score['caption']
      score['stats'] = gen_stats(score['prob'], normalizer=mean_score['prob'])
      score['correct'] = (image_path == caption['source_image'])
      num_correct += score['correct']
    with open(caption_score_filename, 'wb') as caption_score_file:
      pickle.dump(caption_scores, caption_score_file)
  return caption_scores

def retrieval_eval_image_to_caption(
    caption_scores, image_path, image_index, cache_dir):
  caption_scores_by_prob_desc = sorted(caption_scores, \
      key=lambda s: s['stats']['normed_perplex'])
  caption_scores_by_prob_per_word_desc = sorted(caption_scores, \
      key=lambda s: s['stats']['normed_perplex_word'])
  recall = {}
  for method, score_list in [('prob', caption_scores_by_prob_desc),
      ('prob_per_word', caption_scores_by_prob_per_word_desc)]:
    correct = np.array([s['correct'] for s in score_list])
    num_correct = np.sum(correct)
    correct_indices = np.where(correct)
    print 'Method %s: (mean, median) correct index of GT label: (%d, %d)' % \
        (method, np.mean(correct_indices), np.median(correct_indices))
    recall[method] = np.cumsum(correct)
    # for rank in [1, 5, 10, 50, 100]:
    for rank in [1, 5, 10, 50]:
      print 'Method %s: recall at rank %d is %d / %d = %f' % \
          (method, rank, recall[method][rank - 1], num_correct,
           recall[method][rank - 1] / float(num_correct))
  html_im2cap_dir = '%s/html_im2cap' % cache_dir
  if not os.path.exists(html_im2cap_dir):
    os.makedirs(html_im2cap_dir)
  html_out_filename = '%s/ranked_captions_image_%d.html' % \
      (html_im2cap_dir, image_index)
  if os.path.exists(html_out_filename):
    print 'HTML report already exists at %s; skipping' % html_out_filename
    return recall
  html_out = to_html_output({image_path: caption_scores}, vocab)
  html_out_file = open(html_out_filename, 'w')
  html_out_file.write(html_out)
  html_out_file.close()
  print 'Wrote HTML report to: %s' % html_out_filename
  return recall

def retrieval_eval_caption_to_image(caption_list, caption_scores, caption_index,
    cache_dir, vocab):
  scores = []
  gt_image_path = caption_list[caption_index]['source_image']
  for image_path, score_list in caption_scores.iteritems():
    score = score_list[caption_index]
    score['correct'] = image_path == gt_image_path
    if not 'stats' in score: score['stats'] = gen_stats(score['prob'])
    stats = score['stats']
    scores.append((stats['log_p_word'], stats['log_p'], score['correct']))
  scores_by_p_word = sorted(scores, key=lambda s: -s[0])
  scores_by_p = sorted(scores, key=lambda s: -s[1])
  correct_ranks = {'log_p': -1, 'log_p_word': -1}
  for index in range(len(scores)):
    if scores_by_p[index][2]:
      correct_ranks['log_p'] = index
      print 'Method log_p: caption %d ranked ground truth image at %d' \
          % (caption_index, index + 1)
      break
  for index in range(len(scores)):
    if scores_by_p_word[index][2]:
      correct_ranks['log_p_word'] = index
      print 'Method log_p_word: caption %d ranked ground truth image at %d' \
          % (caption_index, index + 1)
      break
  return correct_ranks

# Does an end-to-end retrieval experiment on dataset, which must be a dict
# mapping an image path to a list of "correct" captions for for that path.
def retrieval_experiment(image_net, word_net, dataset, vocab, cache_dir):
  if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
  image_list = retrieval_image_list(dataset, cache_dir)
  descriptors = retrieval_descriptors(image_net, image_list, cache_dir)
  caption_list = retrieval_caption_list(dataset, image_list, cache_dir)
  mean_descriptor = descriptors.mean(axis=0)
  mean_caption_scores = retrieval_caption_scores(word_net, len(image_list),
      mean_descriptor, caption_list, cache_dir)
  caption_scores = {}
  caption_stats = {}
  final_caption_scores = {}
  caption_recalls = {}
  # recall_ranks = [1, 5, 10, 50, 100]
  recall_ranks = [1, 5, 10, 50]
  all_recalls = {}
  # Evaluate image to caption task.
  for image_index, image_path, descriptor in \
      zip(range(len(image_list)), image_list, descriptors):
    caption_scores[image_path] = retrieval_caption_scores(
        word_net, image_index, descriptor, caption_list, cache_dir)
    caption_stats[image_path] = retrieval_caption_stats(
        caption_list, image_path, caption_scores[image_path], image_index,
        cache_dir, mean_caption_scores)
    caption_recalls[image_path] = retrieval_eval_image_to_caption(
        caption_stats[image_path], image_path, image_index, cache_dir)
    for method, recall in caption_recalls[image_path].iteritems():
      if method not in all_recalls:
        all_recalls[method] = {}
        for rank in recall_ranks:
          all_recalls[method][rank] = []
      for rank in recall_ranks:
        all_recalls[method][rank].append(recall[rank - 1] / float(recall[-1]))
  # Evaluate caption to image task.
  caption_to_image_ranks = []
  for caption_index in range(len(caption_list)):
    caption_to_image_ranks.append(retrieval_eval_caption_to_image(
        caption_list, caption_scores, caption_index, cache_dir, vocab))
  all_caption_ranks = {}
  for caption_ranks in caption_to_image_ranks:
    for method, image_rank in caption_ranks.iteritems():
      if method not in all_caption_ranks:
        all_caption_ranks[method] = []
      all_caption_ranks[method].append(image_rank)
  # Print results
  for method, rank_to_recall in all_recalls.iteritems():
    for rank in recall_ranks:
      print 'Image to caption: method %s: mean recall at %d is: %f' % \
          (method, rank, np.mean(rank_to_recall[rank]))
  for method, ranks in all_caption_ranks.iteritems():
    print 'Caption to image: method %s: (mean, median) correct rank is (%d, %d)' % \
        (method, np.mean(np.array(ranks, dtype=np.float)), np.median(np.array(ranks, dtype=np.float)))
    for recall_rank in recall_ranks:
      num_recalled = 0
      for rank in ranks:
        if recall_rank - 1 >= rank:
          num_recalled += 1
      recall = float(num_recalled) / len(ranks)
      print 'Caption to image: method %s: mean recall at %d is %f' % \
          (method, recall_rank, recall)

def flickr_sample_all():
  NUM_SAMPLES_PER_IMAGE = 10
  IMAGE_NET_FILE = './alexnet_to_lstm_net.image_to_fc8.batch50.deploy.prototxt'
  LSTM_NET_FILE = './alexnet_to_lstm_net.word_to_preds.batch500.deploy.prototxt'
  TAG = 'ft_all'
  if TAG == 'fc8_raw':
    ITER = 37000
    MODEL_FILE = './snapshots/coco_flickr_30k_alexnet_to_lstm_4layer_' + \
                 'lr0.1_mom_0.9_iter_%d.caffemodel' % ITER
  elif TAG == 'ft_all':
    ITER = 30000
    MODEL_FILE = './snapshots/coco_flickr_30k_alexnet_to_lstm_4layer_' + \
                 'lr0.01_mom_0.9_ftend2end_iter_%d.caffemodel' % ITER
  else:
    raise Exception('Unknown tag: %s' % TAG)
  NET_TAG = '%s_iter_%d' % (TAG, ITER)
  # Set up the nets.
  image_net = caffe.Net(IMAGE_NET_FILE, MODEL_FILE)
  lstm_net = caffe.Net(LSTM_NET_FILE, MODEL_FILE)
  nets = [image_net, lstm_net]
  channel_mean = np.array([104, 117, 123])[:, np.newaxis, np.newaxis]
  image_net.set_mean('data', channel_mean, mode='channel')
  image_net.set_channel_swap('data', (2, 1, 0))
  for net in nets:
    net.set_phase_test()
    if DEVICE_ID >= 0:
      net.set_mode_gpu()
      net.set_device(DEVICE_ID)
    else:
      net.set_mode_cpu()
  _, _, val_datasets = DATASETS[1]
  flickr_dataset = [val_datasets[0]]
  coco_dataset = [val_datasets[1]]
  datasets = [flickr_dataset]
  dataset_names = ['flickr30k', 'coco']
  fsg = FlickrSequenceGenerator(flickr_dataset, VOCAB_FILE, 0, align=False, shuffle=False)
  image_gt_pairs = all_image_gt_pairs(fsg)
  image_list = image_gt_pairs.keys()
  eos_string = '<EOS>'
  vocab = [eos_string] + fsg.vocabulary_inverted
  descriptors = compute_descriptors(image_net, image_list)
  image_features = np.zeros_like(lstm_net.blobs['image_features'].data)
  word_batch_size = image_features.shape[0]
  assert word_batch_size % NUM_SAMPLES_PER_IMAGE == 0
  images_per_batch = word_batch_size / NUM_SAMPLES_PER_IMAGE
  num_samples = len(image_list) * NUM_SAMPLES_PER_IMAGE
  num_batches = num_samples / word_batch_size + (num_samples % word_batch_size > 0)
  num_images_done = 0
  image_sample_index = 0
  out_samples = []
  sample_dir = './cocoflickr/samples'
  sample_filename = '%s/%s_flickr_val_samples.pkl' % (sample_dir, NET_TAG)
  if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
  if os.path.exists(sample_filename):
    raise Exception('Sample file already exists: %s' % sample_filename)
  for batch_index in range(num_batches):
    sample_image_paths = []
    for batch_offset in range(word_batch_size):
      image_index = batch_index * images_per_batch + batch_offset / NUM_SAMPLES_PER_IMAGE
      image_features[batch_offset] = descriptors[image_index]
      sample_image_paths.append(image_list[image_index])
    print '(%d/%d) Computing %d samples for %d images' % \
        (batch_index, num_batches, NUM_SAMPLES_PER_IMAGE, images_per_batch)
    samples = sample_captions(lstm_net, image_features)
    assert len(samples) == word_batch_size
    for sample_image_path, sample in zip(sample_image_paths, samples):
      sample['source'] = sample_image_path
    out_samples += samples
  print 'Saving samples to: %s' % sample_filename
  with open(sample_filename, 'wb') as sample_file:
    pickle.dump(out_samples, sample_file)
  print 'Done.'

def main():
  # NET_FILE = './alexnet_to_lstm_net.deploy.prototxt'
#   IMAGE_NET_FILE = './alexnet_to_lstm_net.image_to_fc8.batch50.deploy.prototxt'
#   LSTM_NET_FILE = './alexnet_to_lstm_net.word_to_preds.batch500.deploy.prototxt'
  IMAGE_NET_FILE = './alexnet_to_lstm_net.image_to_fc8.deploy.prototxt'
  LSTM_NET_FILE = './alexnet_to_lstm_net.word_to_preds.deploy.prototxt'
  # TAG = 'ft_all'
  # TAG = 'fc8_raw'
  # for TAG in ['ft_lm_plus_alexnet']:
  # for TAG in ['fc8_raw']:
  for TAG in ['ft_all']:
    if TAG == 'fc8_raw':
      ITER = 37000
      MODEL_FILE = './snapshots/coco_flickr_30k_alexnet_to_lstm_4layer_' + \
                   'lr0.1_mom_0.9_iter_%d.caffemodel' % ITER
    elif TAG == 'ft_all':
      ITER = 30000
      MODEL_FILE = './snapshots/coco_flickr_30k_alexnet_to_lstm_4layer_' + \
                   'lr0.01_mom_0.9_ftend2end_iter_%d.caffemodel' % ITER
    elif TAG == 'ft_lm_plus_alexnet':
      ITER = 6000
      IMAGE_NET_FILE = './alexnet_to_lstm_net.image_to_fc8.deploy.prototxt'
      LSTM_NET_FILE = './alexnet_to_lstm_net.word_to_preds.deploy.prototxt'
      MODEL_FILE = './snapshots/flickr_30k_alexnet_to_lstm_4layer_lr0.1_' + \
          'mom0.9_noimagebaseline_add_alexnet_iter_%d.caffemodel' % ITER
    else:
      raise Exception('Unknown tag: %s' % TAG)
    NET_TAG = '%s_iter_%d' % (TAG, ITER)

    # Set up the nets.
    image_net = caffe.Net(IMAGE_NET_FILE, MODEL_FILE)
    lstm_net = caffe.Net(LSTM_NET_FILE, MODEL_FILE)
    nets = [image_net, lstm_net]
    channel_mean = np.array([104, 117, 123])[:, np.newaxis, np.newaxis]
    image_net.set_mean('data', channel_mean, mode='channel')
    image_net.set_channel_swap('data', (2, 1, 0))
    for net in nets:
      net.set_phase_test()
      if DEVICE_ID >= 0:
        net.set_mode_gpu()
        net.set_device(DEVICE_ID)
      else:
        net.set_mode_cpu()

    RESULTS_DIR = './html_results_kiros'
    STRATEGIES = [
#       {'type': 'sample', 'temp': 0.75, 'num': 3},
#       {'type': 'sample', 'temp': 1.0, 'num': 3},
#       {'type': 'sample', 'temp': 3.0, 'num': 3},
#       {'type': 'sample', 'temp': 5.0, 'num': 3},
#       {'type': 'beam', 'beam_size': 1},
#       {'type': 'beam', 'beam_size': 3},
      {'type': 'beam', 'beam_size': 5},
    ]
    NUM_CHUNKS = 100
    NUM_OUT_PER_CHUNK = 10
    START_CHUNK = 20
    # START_CHUNK = 5

    _, _, val_datasets = DATASETS[1]
    flickr_dataset = [val_datasets[0]]
    coco_dataset = [val_datasets[1]]
    datasets = [flickr_dataset]
    dataset_names = ['flickr30k', 'coco']

    do_retrieval_experiment = False
    if do_retrieval_experiment:
      fsg = FlickrSequenceGenerator(flickr_dataset, VOCAB_FILE, 0, align=False, shuffle=False)
      image_gt_pairs = all_image_gt_pairs(fsg)
      eos_string = '<EOS>'
      vocab = [eos_string] + fsg.vocabulary_inverted
      retrieval_cache_dir = './cocoflickr/flickr_val_retrieval/%s' % NET_TAG
#     retrieval_cache_dir = './cocoflickr/mini_flickr_val_retrieval/%s' % NET_TAG
#     mini_num = 10
#     mini_image_gt_pairs = {}
#     for key, val in image_gt_pairs.iteritems():
#       mini_image_gt_pairs[key] = val
#       if len(mini_image_gt_pairs.keys()) >= mini_num: break
#     retrieval_experiment(image_net, lstm_net, mini_image_gt_pairs, vocab,
#         retrieval_cache_dir)
      retrieval_experiment(image_net, lstm_net, image_gt_pairs, vocab,
          retrieval_cache_dir)
      import pdb; pdb.set_trace()
    else:
      # kiros_images = [l.strip() for l in open('./compare_kiros/kiros_images.txt', 'r').readlines()]
      # image_gt_pairs = OrderedDict()
      # for image in kiros_images:
      #   image_gt_pairs[image] = []
      for dataset, dataset_name in zip(datasets, dataset_names):
        fsg = FlickrSequenceGenerator(dataset, VOCAB_FILE, 0, align=False, shuffle=False)
        image_gt_pairs = all_image_gt_pairs(fsg)
        eos_string = '<EOS>'
        vocab = [eos_string] + fsg.vocabulary_inverted
        offset = 0
        for c in range(START_CHUNK, NUM_CHUNKS):
          chunk_start = c * NUM_OUT_PER_CHUNK
          chunk_end = (c + 1) * NUM_OUT_PER_CHUNK
          chunk = kiros_images[chunk_start:chunk_end]
          html_out_filename = '%s/%s.%s.%d_to_%d.html' % \
              (RESULTS_DIR, dataset_name, NET_TAG, chunk_start, chunk_end)
          if os.path.exists(html_out_filename):
            print 'HTML output exists, skipping:', html_out_filename
            continue
          else:
            print 'HTML output will be written to:', html_out_filename
          outputs = run_pred_iters(image_net, lstm_net, chunk, image_gt_pairs,
              strategies=STRATEGIES, display_vocab=vocab)
          html_out = to_html_output(outputs, vocab)
          if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
          html_out_file = open(html_out_filename, 'w')
          html_out_file.write(html_out)
          html_out_file.close()
          offset += NUM_OUT_PER_CHUNK
          print 'Wrote HTML output to:', html_out_filename

if __name__ == "__main__":
  # main()
  flickr_sample_all()
