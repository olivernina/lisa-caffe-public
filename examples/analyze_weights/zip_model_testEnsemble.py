import numpy as np
caffe_root = '/home/lisaanne/caffe-forward-backward/'
import sys
sys.path.insert(0, caffe_root + 'python')
import os
import caffe
import scipy
import random
import copy
from analyze_functions import *
import pickle
caffe.set_mode_gpu()
caffe.set_device(4)
import frankenNet
from zip_model import *
home_dir = '/home/lisaanne/caffe-forward-backward/examples/analyze_weights/'

num_models = 1

MODEL = home_dir + 'alexnet_deploy.prototxt'
PRETRAINED = ['/mnt/y/lisaanne/Caffenets/alexnet_seed%d/caffe_alexnet_train_iter_300000.caffemodel' %n for n in range(1,16)]
alexnetProto = 'alexnet_frankenNet.prototxt'
alexnetTrainProto = 'alexnet_frankenNet_train.prototxt' 
frankenReplaceDict = {}
frankenReplaceDict['FILTERS_CONV1'] = str(num_models*96)
frankenReplaceDict['FILTERS_CONV2'] = str(num_models*256)
frankenReplaceDict['FILTERS_CONV3'] = str(num_models*384)
frankenReplaceDict['FILTERS_CONV4'] = str(num_models*384)
frankenReplaceDict['FILTERS_CONV5'] = str(num_models*256)
frankenReplaceDict['FILTERS_FC6'] = str(num_models*4096)
frankenReplaceDict['FILTERS_FC7'] = str(num_models*4096)
#################################################################################

all_probs = []
accuracies = []

for i in range(0,15):
  fNet = zippedModel(num_models, MODEL, [PRETRAINED[i]])
  fNet.initModel(alexnetProto, alexnetTrainProto, frankenReplaceDict, 'fNet_tmp')
  fNet.concatWeights(['conv1','conv2','conv3', 'conv4', 'conv5'],['fc6','fc7','fc8'])
  print 'On model ', i
  accuracy, labels, probs = fNet.netOutputs(1000)
  accuracies.append(accuracy)
  all_probs.append(probs)
  del fNet

for i in range(0,15):
  print 'Accuracy for net %d is %f.' %(i, accuracies[i])

for i in range(0,15):
  probs = np.zeros((all_probs[0].shape))
  for j in range(0,i+1):
    probs += all_probs[j]
  labels_pred = np.argmax(probs,1)
  num_correct = len(np.where(labels_pred == labels)[0]) 
  print 'Accuracies combining %d nets is %f.' %(i, float(num_correct)/len(labels))


