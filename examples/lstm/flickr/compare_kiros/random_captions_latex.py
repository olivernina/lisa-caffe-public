#!/usr/bin/env python

import cPickle as p
import random

data = p.load(open('./jeff_kiros_captions.pkl'))

print '\\begin{tabular}{ll}'
# methods = ('jeff_fc8_raw', 'jeff_ft_all')
methods = ('jeff_ft_all', )
num_out = 50
keys = data.keys()
random.shuffle(keys)
num_examples = 0
for key in keys:
  results = data[key]
  out = '\\raisebox{-.5\\height}{\\includegraphics[scale=0.25]{%s}} & ' % key
  skip_key = False
  for method in methods:
    if method not in results:
      skip_key = True
      break
    result = results[method][0]
    out += result
  if skip_key: continue
  out += '\\\\'
  print out
  num_examples += 1
  if num_examples >= num_out: break
print '\\end{tabular}'
