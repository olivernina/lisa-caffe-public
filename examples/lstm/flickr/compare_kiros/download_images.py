#!/usr/bin/env python

import os
import re
import urllib

regex = re.compile('http:.*\/(.*\.jpg)')

image_dir = 'kiros_images/images'
if not os.path.exists(image_dir): os.makedirs(image_dir)
urls = [line.strip() for line in open('kiros_images/image_urls.txt', 'r').readlines()]
filenames = [regex.match(url).group(1) for url in urls]
for url, filename in zip(urls, filenames):
  out_filename = '%s/%s' % (image_dir, filename)
  print 'Downloading %s to: %s' % (url, out_filename)
  urllib.urlretrieve(url, out_filename)
