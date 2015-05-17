import numpy as np
import matplotlib.pyplot as plt
caffe_root = '/home/lisaanne/caffe-dev/'
import sys
sys.path.insert(0, caffe_root + 'python')
import os
import caffe
import scipy
import random
import copy
caffe.set_mode_gpu()

random.seed(1701)

class frankenNet(object):
  def __init__(self, num_models):

    self.num_models = num_models

    home_dir = '/home/lisaanne/caffe-dev/examples/analyze_weights/'
    self.home_dir = home_dir
    MODEL = home_dir + 'cifar10_full_deploy.prototxt'
    PRETRAINED = ['trained_models/cifar10_full_model1_iter_70000.caffemodel',
                  'trained_models/snapshots_rs_107583_iter_70000.caffemodel', 
                  'trained_models/snapshots_rs_11255_iter_70000.caffemodel', 
                  'trained_models/snapshots_rs_52681_iter_70000.caffemodel', 
                  'trained_models/snapshots_rs_80573_iter_70000.caffemodel',
                  'trained_models/snapshots_rs_1361_iter_70000.caffemodel', 
                  'trained_models/snapshots_rs_4572_iter_70000.caffemodel',
                  'trained_models/snapshots_rs_5916_iter_70000.caffemodel']
    nets = []
    for model in PRETRAINED:
       nets.append(caffe.Net(MODEL, home_dir+model, caffe.TEST))

    self.trained_nets = nets

    fNetProto = open(home_dir + 'cifar10_frankenNet.prototxt', 'rb')
    protoLines = fNetProto.readlines()
    fNetProto.close()
    protoLines = [x.replace('FILTERS_CONV1',str(num_models*32)) for x in protoLines]
    protoLines = [x.replace('FILTERS_CONV2',str(num_models*32)) for x in protoLines]
    protoLines = [x.replace('FILTERS_CONV3',str(num_models*64)) for x in protoLines]

    fNetTmp = open(home_dir+'fNet_tmp.prototxt', 'wb')
    for line in protoLines:
      fNetTmp.write(line)
    fNetTmp.close()

    self.net = caffe.Net(home_dir + 'fNet_tmp.prototxt', caffe.TEST)

  def initNetParams(self):
    nets = self.trained_nets
    self.net.params['ip1'][1].data[...] = 0
    #put parameters into frankenNet   
    for i, net in enumerate(nets[0:self.num_models]):

      self.net.params['conv1'][0].data[i*32:i*32+32,...] = copy.deepcopy(net.params['conv1'][0].data)
      self.net.params['conv1'][1].data[i*32:i*32+32] = copy.deepcopy(net.params['conv1'][1].data)
      self.net.params['conv2'][0].data[i*32:i*32+32,i*32:i*32+32,...] = net.params['conv2'][0].data
      self.net.params['conv2'][1].data[i*32:i*32+32] = net.params['conv2'][1].data
      self.net.params['conv3'][0].data[i*64:i*64+64,i*32:i*32+32,...] = net.params['conv3'][0].data
      self.net.params['conv3'][1].data[i*64:i*64+64] = net.params['conv3'][1].data
      self.net.params['ip1'][0].data[:,i*1024:i*1024+1024,...] = net.params['ip1'][0].data
      self.net.params['ip1'][1].data[...] += net.params['ip1'][1].data

  def testNet(self):
    num_videos = 0
    num_correct = 0
    for i in range(0,100):
      out = self.net.forward()
      probs = out['probs']
      labels = out['label-out']

      labels_pred_conv1 = np.argmax(probs[:,0:10],1)
      num_correct += len(np.where(labels_pred_conv1 == labels)[0])

      num_videos += 100
    return float(num_correct)/num_videos

  def randRemove(self, layer, proportion):
     #remove a random proportion of filters
     num_channels = self.net.params[layer][0].data.shape[0]
     elim_channels = range(num_channels)
     random.shuffle(elim_channels)
     elim_channels = elim_channels[0:int(proportion*num_channels)]
     for c in elim_channels:
       self.net.params[layer][0].data[c,...] = 0
       self.net.params[layer][1].data[c] = 0

  def orderedRemove(self, layer, rem_filters):
    #remove filters specified by rem_filters from layer
    for f in rem_filters: 
       self.net.params[layer][0].data[f,...] = 0
       self.net.params[layer][1].data[f] = 0

#PUT WRITING OF THIS FUNCTION ON HOLD
#  def orderChannels(self, refNet, compNet, layer):
#    #This function will order channels in compNet such that the corresponding compNet channels are the channels MOST aligned with refNet.  ISSUES: this is not one to one mapping.
#
#    #refNet: number indicating the reference net we will align filters from compNet to
#    #compNet: number indicating the compNet.  We will change order of compNet filters
#    #layer: Layer in which we are comparing filters
#
#    #Note: assumes previous layers are ordered
#
#    filters_per_net = self.net.params[layer].data.shape[0]/self.num_models
#    refNetParams = self.net.params[layer].data[refNet*filters_per_net:refNet*filters_per_net+filters_per_net,...]
#    compNetParams = self.net.params[layer].data[compNet*filters_per_net:compNet*filters_per_net+filters_per_net,...]
#    alignedParams = np.zeros(compNetParams.shape)



  def cleanUp(self):
    os.remove(self.home_dir+'fNet_tmp.prototxt')

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)

def randomFilterRemoval():
  #Test for randomly removing filters from conv1
  rem_prop = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  accuracies = []
  for r in rem_prop:
    fNet.randRemove('conv1', r)
    accuracies.append(fNet.testNet())
    fNet.initNetParams()

  for i, r in enumerate(rem_prop):
    print 'Accuracy removing %f of filters is %f' %(r, accuracies[i]) 
 
def orderedFilterRemoval():

num_models = 5
if len(sys.argv) == 2:
  num_models = int(sys.argv[1])

fNet = frankenNet(num_models)

fNet.initNetParams()
accuracyRef = fNet.testNet()

#randomFilterRemoval()
orderedFilterRemoval()


#print reference accuracy and cleanUp
print 'Accuracy combining %d nets is %f' %(num_models, accuracyRef) 
fNet.cleanUp()








