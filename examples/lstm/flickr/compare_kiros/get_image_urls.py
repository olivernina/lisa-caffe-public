#!/usr/bin/env python

import re

regex = re.compile('src="(http://.*.jpg)\s*"')

html_lines = open('lstm_scnlm.html', 'r').readlines()
output_list = open('kiros_images/image_urls.txt', 'w')
for line in html_lines:
  results = regex.search(line)
  if results is not None:
    output_list.write(results.group(1) + '\n')
output_list.close()
