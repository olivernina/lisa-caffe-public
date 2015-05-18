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
from analyze_functions import *
import pickle
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
     channels_drop = int(proportion*num_channels)
     elim_channels = elim_channels[0:channels_drop]

     #layer_index = self.net.params.keys().index(layer)
     #prop_layers = self.net.params.keys()[layer_index:]
     #self.net.params[layer][0].data[...] = self.net.params[layer][0].data[...]*(num_channels/(num_channels - float(channels_drop)))
     for c in elim_channels:
       self.net.params[layer][0].data[c,...] = 0
       self.net.params[layer][1].data[c] = 0

  def orderedRemove(self, layer, rem_filters):
    #remove filters specified by rem_filters from layer
    self.net.forward()
    for f in rem_filters: 
       #mean_activations = np.mean(self.net.blobs['norm1'].data[:,f,...])
       self.net.params[layer][0].data[f,...] = 0
       self.net.params[layer][1].data[f] = 0
       #self.net.params['conv2'][1].data[...] += mean_activations

    self.net.params['ip1'][1].data[...] -= (1./64)*self.net.params['ip1'][1].data

  def findSimilar(self, refNet, compNet, layer, activation, f):
  #Find filter in compNet most similar to refNet.  If below certain threshold replace.  
    self.net.forward()
    filters_per_net = self.net.params[layer][0].data.shape[0]/self.num_models
    corr = []
    for im in range(0,100):
      a = self.net.blobs[activation].data[im:im+1,refNet*filters_per_net+f,...]
      b = self.net.blobs[activation].data[im:im+1,compNet*filters_per_net:compNet*filters_per_net+filters_per_net,...]
      res = normxcorrNFast(a,b[0,...])
      maxes = np.max(np.max(res,1),1)
      corr.append(maxes)
    sums = np.zeros(corr[0].shape)
    for s in corr:
      sums += s
    sim_filter = np.argmax(sums) + compNet*filters_per_net
    sim_measure = sums/100
    return sim_filter, sim_measure 

  def replace(self, layer, rep, fill, graft_later=0):
    filters_per_net = self.net.params['conv2'][0].data.shape[0]/self.num_models
    if graft_later == 0:
      sim = []
      for i in range(filters_per_net):
        sim.append(self.findSimilar(0, 1, 'conv2', 'pool2', i))
      for i, s in enumerate(sim):
        self.net.params['conv2'][0].data[sim,rep,...] = copy.deepcopy(self.net.params['conv2'][0].data[i,fill,...])
    self.net.params[layer][0].data[rep,...] = copy.deepcopy(self.net.params[layer][0].data[fill,...])
    self.net.params[layer][1].data[rep] = copy.deepcopy(self.net.params[layer][1].data[fill,...])

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
    accuracy = fNet.testNet()
    fNet.initNetParams()
    print 'Accuracy removing %f of filters is %f' %(r, accuracy) 
 
def orderedFilterRemoval():
  #Test for removing filters one by one from 2 model frankenNet
  for i in range(32,64):
    fNet.orderedRemove('conv1', [i])
    accuracy = fNet.testNet()
    print 'Accuracy removing filters up to %d of second net is %f.' %(i, accuracy)    
    fNet.initNetParams()

def graftFilter(graft_later, from_net, onto_net):
  average_rep_accuracies = 0
  average_rep_accuracies_vec = []
  simMeasure_all = []
  for origFilter in range(0,32):
    print 'Orig filter is ', origFilter
    similarFilter, simMeasure = fNet.findSimilar(from_net, onto_net, 'conv1', 'pool1', origFilter)
    simMeasure_all.append(simMeasure)
    fNet.replace('conv1', origFilter, similarFilter, graft_later)
    a = fNet.testNet()
    average_rep_accuracies += a
    average_rep_accuracies_vec.append(a)
    print 'Accuracy replacing filter with similar is %f.' %(a)
    fNet.initNetParams()

  print 'Average rep accuracy is %f.' %(average_rep_accuracies/32)
  
  return average_rep_accuracies, simMeasure_all 

#    fNet.orderedRemove('conv1', [origFilter])
#    a = fNet.testNet()
#    average_del_accuracies += a
#    print 'Accuracy deleting similar filter is %f.' %(a)
#    fNet.initNetParams()
#    print 'Accuracy after initializing test parameters is %f.' %(fNet.testNet())
  

num_models = 5
if len(sys.argv) == 2:
  num_models = int(sys.argv[1])

fNet = frankenNet(num_models)

fNet.initNetParams()
accuracyRef = fNet.testNet()

#randomFilterRemoval()
#orderedFilterRemoval()

average_accuracies, simMeasure = graftFilter(0,0,1)
save_data = {}
save_data['accuracies'] = average_accuracies
save_data['simMeasure'] = simMeasure
pickle.dump(save_data, open('results/AnS_conv1_2net_graftFilter001.p','wb'))

average_accuracies, simMeasure = graftFilter(1,0,1)
save_data = {}
save_data['accuracies'] = average_accuracies
save_data['simMeasure'] = simMeasure
pickle.dump(save_data, open('results/AnS_conv1_2net_graftFilter101.p','wb'))

average_accuracies, simMeasure = graftFilter(0,1,0)
save_data = {}
save_data['accuracies'] = average_accuracies
save_data['simMeasure'] = simMeasure
pickle.dump(save_data, open('results/AnS_conv1_2net_graftFilter010.p','wb'))

average_accuracies, simMeasure = graftFilter(1,1,0)
save_data = {}
save_data['accuracies'] = average_accuracies
save_data['simMeasure'] = simMeasure
pickle.dump(save_data, open('results/AnS_conv1_2net_graftFilter110.p','wb'))

#print reference accuracy and cleanUp
print 'Accuracy combining %d nets is %f' %(num_models, accuracyRef) 
fNet.cleanUp()








