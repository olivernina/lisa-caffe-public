#!/usr/bin/env python

import cPickle as p
import random

data = p.load(open('./jeff_kiros_captions.pkl'))

# methods = ('jeff_fc8_raw', 'jeff_ft_all')
method = 'jeff_ft_all'
num_out = 50
keys = data.keys()
random.shuffle(keys)
num_examples = 0
for key in keys:
  results = data[key]
  if method not in results:
    continue
  print '''\\begin{subfigure}[b]{0.3\\textwidth}
                \\includegraphics[width=0.9\\textwidth]{%s}
                \\caption{%s}
                \\label{fig:Img1Id}
         \\end{subfigure}%%
         \\quad''' % ('compare_kiros/kiros_images/images/%s' % key, results[method][0])
  num_examples += 1
  if num_examples >= num_out: break
