#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

mode = 'train'
train_sample_rate = 100
mode = 'test'
test_mode = 'on_val'
test_interval = 1000

log_dir = './logs/ablate/'

if mode == 'train':
  log_ext = '.txt.train'
elif mode == 'test':
  log_ext = '.txt.test'
else:
  raise Exception('Unknown mode: %s' % mode)

filenames = [f for f in os.listdir(log_dir) if f.endswith(log_ext)]

styles = ['r', 'g', 'b']

assert len(filenames) <= len(styles)
styles = styles[:len(filenames)]

plot_args = []
legend = []
for style, filename in zip(styles, filenames):
  plot = []
  lines = open('%s/%s' % (log_dir, filename), 'r').readlines()[1:]
  if mode == 'train':
    lines = lines[::train_sample_rate]
  else:
    assert len(lines) % 2 == 0
    if test_mode == 'on_train':
      lines = lines[::2]
    elif test_mode == 'on_val':
      lines = lines[1::2]
    else:
      raise Exception('Unknown test_mode: %s' % test_mode)
  for index, line in enumerate(lines):
    line_parts = line.split()
    if mode == 'train':
      assert 3 <= len(line_parts) <= 4
      if len(line_parts) == 3:
        num_iters, seconds, loss = line_parts
      elif len(line_parts) == 4:
        num_iters, seconds, loss, lr = line_parts
      assert int(num_iters) / train_sample_rate == index
    elif mode == 'test':
      assert len(line_parts) == 3
      seconds, acc, loss = line_parts
    plot.append(loss)
  if mode == 'train':
    plot_args.append(range(0, len(plot) * train_sample_rate, train_sample_rate))
  else:
    plot_args.append(range(0, len(plot) * test_interval, test_interval))
  plot_args.append(plot)
  plot_args.append(style)
  if '.factored_2layer.' in filename:
    key = '2 layer, factored'
  elif '.unfactored_2layer.' in filename:
    key = '2 layer, unfactored'
  elif '.unfactored_1layer.' in filename:
    key = '1 layer'
  else:
    raise Exception('Unknown filename: %s' % filename)
  legend.append(key)

plot_args = tuple(plot_args)

handles = plt.plot(*plot_args)
plt.figlegend(handles, tuple(legend), 'center')
plt.xlabel('# Iterations')
if mode == 'train':
  plt.ylabel('Training Loss')
  disp_mode = 'train'
elif mode == 'test':
  plt.ylabel('Validation Loss')
  disp_mode = 'test_' + test_mode
plt.savefig('ablate_log_plot.%s.pdf' % disp_mode)
