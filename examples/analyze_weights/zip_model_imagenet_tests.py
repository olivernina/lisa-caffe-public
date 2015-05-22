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

#Initialize parameters

num_models = 2
if len(sys.argv) == 2:
  num_models = int(sys.argv[1])

similarityFilesConv1 = [('/mnt/y/lisaanne/modelZip/modelSimilarity/alexnet_%s_15Net_fromNet%d.p' %('conv1',n)) for n in range(0,15)]
MODEL = home_dir + 'alexnet_deploy.prototxt'
PRETRAINED = ['/mnt/y/lisaanne/Caffenets/alexnet_seed%d/caffe_alexnet_train_iter_300000.caffemodel' %n for n in range(1,15)]
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

layer = 'conv1'
fNet = zippedModel(num_models, MODEL, PRETRAINED)
fNet.initModel(alexnetProto, alexnetTrainProto, frankenReplaceDict, 'fNet_tmp')
fNet.concatWeights(['conv1','conv2','conv3', 'conv4', 'conv5'],['fc6','fc7','fc8'])
print 'FrankenNet accuracy is %f.' %fNet.testNet(50)
#fNet.loadSimilarity(similarityFiles, layer)
fNet.determineSimilarity('conv1','pool1')
fNet.determineGraftDictV1(layer)
#fNet.loadGraftDict('results/graft_dict_conv1_V1.p')
#modelzip!  This will define new attribute fNet.zipNet.
zipNetConv1 = fNet.modelZip(layer, 'FILTERS_CONV1', 'fZipConv1_tmp.prototxt')
#Test to make sure that grafting works (with no zipping yet)
fNet.testGraftDict(layer)
print 'Zipped model accuracy is: %f.' %zipNetConv1.testZipNet()
#need to write fine-tuning code
zipNetConv1.ftZipModel('solver_zipModel_finetune.prototxt','ft_zippedConv1')


