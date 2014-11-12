#!/usr/bin/env python

import cPickle as pickle
import os
import re

url_regex = re.compile('src="(http://.*.jpg)\s*"')
filename_regex = re.compile('http:.*\/(.*\.jpg)')
jeff_url_regex = re.compile('src="(\./.*.jpg)\s*"')
jeff_filename_regex = re.compile('.*\/(.*\.jpg)')
jeff_caption_regex = re.compile('<strong>(.*)</strong>')

def kiros_line_to_caption(line):
  prefix = '<br> '
  assert line.startswith(prefix)
  line = line[len(prefix):].strip()
  if line.endswith(' .'):
    line = line[:-2] + '.'
  line = line[0].upper() + line[1:]
  return line

def jeff_line_to_caption(line):
  match = jeff_caption_regex.search(line.strip())
  assert match is not None
  return match.group(1)

def parse_kiros_lines(html_lines):
  output_list = open('kiros_images/image_urls.txt', 'w')
  data = {}
  url = None
  next_num = 0
  filename = None
  line_index = 0
  while line_index < len(html_lines):
    line = html_lines[line_index].strip()
    results = url_regex.search(line)
    if results is not None:
      url = results.group(1)
      filename_results = filename_regex.search(url)
      assert filename_results is not None
      filename = filename_results.group(1)
      output_list.write(url + '\n')
    if filename not in data:
      data[filename] = {'url': url}
    if line == '<br><b>Original:</b>':
      data[filename]['orig'] = [kiros_line_to_caption(html_lines[line_index + 1])]
      line_index += 1
    elif line == '<br><br><b>Nearest Neighbour:</b>':
      data[filename]['nn'] = [kiros_line_to_caption(html_lines[line_index + 1])]
      line_index += 1
    elif line == '<br><b>TreeTalk:</b>':
      data[filename]['treetalk'] = [kiros_line_to_caption(html_lines[line_index + 1])]
      line_index += 1
    elif line == '<br><br><b>Top-5 model samples:</b>':
      data[filename]['kiros'] = []
      for i in range(5):
        data[filename]['kiros'].append(kiros_line_to_caption(html_lines[line_index + 1 + i]))
      line_index += 5
    line_index += 1
  output_list.close()
  return data

def parse_jeff_lines(html_lines, method_name):
  data = {}
  url = None
  next_num = 0
  filename = None
  line_index = 0
  while line_index < len(html_lines):
    line = html_lines[line_index].strip()
    results = jeff_url_regex.search(line)
    if results is not None:
      url = results.group(1)
      filename_results = jeff_filename_regex.search(url)
      assert filename_results is not None
      filename = filename_results.group(1)
    if filename is not None and filename not in data:
      data[filename] = {}
    if line == '<tr><th>Source</th><th>#Words</th><th>Perplexity/Word</th><th>Caption</th></tr>':
      data[filename][method_name] = []
      for i in range(5):
        data[filename][method_name].append(jeff_line_to_caption(html_lines[line_index + i + 1]))
      line_index += 5
    line_index += 1
  return data

def merge_results(result_dicts):
  out_dict = result_dicts[0].copy()
  for result_dict in result_dicts[1:]:
    for key, val in result_dict.iteritems():
      if key not in out_dict: out_dict[key] = {}
      for subkey, subval in val.iteritems():
        assert subkey not in out_dict[key]
        out_dict[key][subkey] = subval
  return out_dict

html_lines = open('lstm_scnlm.html', 'r').readlines()
kiros_data = parse_kiros_lines(html_lines)

jeff_result_root = '../html_results_kiros'
jeff_result_sets = ['ft_all', 'fc8_raw']
jeff_result_files = []
jeff_data = []
for set_name in jeff_result_sets:
  filenames = ['%s/%s' % (jeff_result_root, f) for f in
      os.listdir(jeff_result_root) if f.endswith('.html') and set_name in f]
  lines = []
  for filename in filenames:
    lines += open(filename, 'r').readlines()
  jeff_data.append(parse_jeff_lines(lines, 'jeff_' + set_name))

data = merge_results([kiros_data] + jeff_data)
pickle.dump(data, open('jeff_kiros_captions.pkl', 'wb'))
