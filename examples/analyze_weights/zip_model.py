import numpy as np
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
import frankenNet

def pickBestAccuracy(net, filts, layer):
  acc = 0
  for f in filts:
    fNet.replace(layer, filts, f)
    a = fNet.testNet()
    if a > acc:
      acc = a
      r = f
    fNet.initNetParams()
  return r, acc

class zippedModel(frankenNet.frankenNet):

  def trainZipModel(self, solver):
    solver = caffe.get_solver(solver)
    for layer in solver.net.params.keys():
      for i in range(len(solver.net.params[layer])):
        solver.net.params[layer][i].data[...] = copy.deepcopy(self.zipNet.params[layer][i].data)
    solver.solve()
    
    for layer in solver.net.params.keys():
      for i in range(len(solver.net.params[layer])):
        self.zipNet.params[layer][i].data[...] = copy.deepcopy(solver.net.params[layer][i].data)
    
    print self.testZipNet(self.zipNet)


  def loadSimilarity(self, similarityFiles, layer):
    self.similarity[layer] = {}
    if not len(similarityFiles) == self.num_models:
      print 'Number similarityFiles needs to be the same as num_models!'
      return
    else:
      for i, similarityFile in enumerate(similarityFiles):
        loadFile = pickle.load(open(similarityFile,'rb'))
        self.similarity[layer][i] = loadFile['simMeasure']

  def determineGraftDictV1(self, layer):
    #greedy search of filters    
    filters_per_model = self.net.params[layer][0].data.shape[0]/self.num_models
    #sim0 = self.similarity  #similarity of model1 to other models to save time.
    filters_to_be_matched = range(filters_per_model*self.num_models)
    graft_dict = {}
    for i in filters_to_be_matched:
      graft_dict[i] = i

    for n in range(self.num_models):
      sim = self.similarity[layer][n]
      for f in range(filters_per_model):
        base_filter = n*32+f
        if base_filter in filters_to_be_matched:
          similar_filters = [base_filter]                        
          for n2 in range(n,self.num_models-1):
            s = (n2+1)*32+np.argmax(sim[f][n2])
            if s in filters_to_be_matched:  #what if a later filter has better similarity?
              similar_filters.append(s)
              filters_to_be_matched.remove(s) 

          #should try replacing with correlation and see how well this does.
          #should also try to intelligently combine filters (?)
          fillFilter, acc = pickBestAccuracy(self.net, similar_filters,layer) #do based off similarity
          print 'For filter %d accuracy is %f.' %(base_filter, acc)
          for sf in similar_filters:
            graft_dict[sf] = fillFilter
    self.graft_dict[layer] =  graft_dict
    save_data = {}
    #save_data['graft_dict'] = self.graft_dict
    #pickle.dump(save_data, open('results/graft_dict_conv1_V1.p','wb'))

  def loadGraftDict(self, graft_file):
    f = pickle.load(open(graft_file,'rb'))
    self.graft_dict = f['graft_dict']

  def testGraftDict(self, layer):
    graft_dict = self.graft_dict[layer]
    for key in graft_dict.keys():
      self.net.params[layer][0].data[key,...] = copy.deepcopy(self.net.params[layer][0].data[graft_dict[key],...])
      self.net.params[layer][1].data[key] = copy.deepcopy(self.net.params[layer][1].data[graft_dict[key]])
    print 'Accuracy for graft dict on layer %s is %f.' %(layer, self.testNet())
    self.initNetParams()
  
  def testZipNet(self,model):
    num_videos = 0
    num_correct = 0
    for i in range(0,100):
      out = model.forward()
      probs = out['probs']
      labels = out['label-out']

      labels_pred_conv1 = np.argmax(probs[:,0:10],1)
      num_correct += len(np.where(labels_pred_conv1 == labels)[0])

      num_videos += 100
    return float(num_correct)/num_videos

                     
  def modelZip(self, layer):

    #a lot of clean up needs to happen so that this will be general for all layers

    graft_dict = self.graft_dict[layer]

    #need to make more generalizable
    next_layer = 'conv2'

    layer_filters = np.unique(np.array(graft_dict.values()))
    layer_filters_dict = {}
    for key in graft_dict.keys():
      layer_filters_dict[key] = np.where(layer_filters == graft_dict[key])[0][0]
    num_filters = len(layer_filters)
    origNumFilters = self.net.params[layer][0].data.shape[0]/self.num_models
    #all channels in layer+1 filters must be of size num_filters

    #must create new net with new filter dimensions
    fNetProto = open(self.home_dir + 'cifar10_frankenNet.prototxt','rb')
    protoLines = fNetProto.readlines()
    fNetProto.close()

    ##THIS NEEDS TO BE STREAMLINED TO WORK WITH OTHER LAYERS AND OTHER NETS
    protoLines = [x.replace('FILTERS_CONV1', str(num_filters)) for x in protoLines]
    protoLines = [x.replace('FILTERS_CONV2', str(num_models*32)) for x in protoLines]
    protoLines = [x.replace('FILTERS_CONV3', str(num_models*64)) for x in protoLines]
    fNetTmp = open(self.home_dir + 'zipNet_tmp.prototxt','wb')
    for line in protoLines:
      fNetTmp.write(line)
    fNetTmp.close()
    self.zipNet = caffe.Net(self.home_dir + 'zipNet_tmp.prototxt', caffe.TEST)

    #Copy parameters into zip layer
    for i, lf in enumerate(layer_filters):
      self.zipNet.params[layer][0].data[i,...] = copy.deepcopy(self.net.params[layer][0].data[lf,...])
      self.zipNet.params[layer][1].data[i] = copy.deepcopy(self.net.params[layer][1].data[lf])

    #shuffle channels from layer up to correspond to zipped layers
    for key in layer_filters_dict.keys():
      origNet = key/origNumFilters
      filterRange = range(origNet*origNumFilters,origNet*origNumFilters+origNumFilters)
      self.zipNet.params['conv2'][0].data[filterRange,layer_filters_dict[key],...] = copy.deepcopy(self.net.params['conv2'][0].data[filterRange, key,...])
    for i,b in enumerate(self.net.params['conv2'][1].data):
      self.zipNet.params['conv2'][1].data[i] = copy.deepcopy(b)
   
    #copy channels from higher layers from original filters 
    nets = self.trained_nets
    self.zipNet.params['ip1'][1].data[...] = 0
    for i, net in enumerate(nets[0:self.num_models]):
      self.zipNet.params['conv3'][0].data[i*64:i*64+64,i*32:i*32+32,...] = copy.deepcopy(net.params['conv3'][0].data)
      self.zipNet.params['conv3'][1].data[i*64:i*64+64] = copy.deepcopy(net.params['conv3'][1].data)
      self.zipNet.params['ip1'][0].data[:,i*1024:i*1024+1024,...] = copy.deepcopy(net.params['ip1'][0].data)
      self.zipNet.params['ip1'][1].data[...] += copy.deepcopy(net.params['ip1'][1].data)

num_models = 5
if len(sys.argv) == 2:
  num_models = int(sys.argv[1])

similarityFiles = ['results/AnS_conv1_5net_graft5Filters001_centerCorr.p',
                   'results/AnS_conv1_5net_graft5Filters001_centerCorr_fromNet1.p',
                   'results/AnS_conv1_5net_graft5Filters001_centerCorr_fromNet2.p',
                   'results/AnS_conv1_5net_graft5Filters001_centerCorr_fromNet3.p',
                   'results/AnS_conv1_5net_graft5Filters001_centerCorr_fromNet4.p']

layer = 'conv1'
fNet = zippedModel(num_models)
fNet.initNetParams()
print 'FrankenNet accuracy is %f.' %fNet.testNet()
fNet.loadSimilarity(similarityFiles, layer)
#fNet.determineGraftDictV1(layer)
fNet.loadGraftDict('results/graft_dict_conv1_V1.p')
#Test to make sure that grafting works (no model zipping yet)
#modelzip!  This will define new attribute fNet.zipNet.
fNet.modelZip(layer)
fNet.testGraftDict(layer)
print 'Zipped model accuracy is: %f.' %fNet.testZipNet(fNet.zipNet)
fNet.trainZipModel('solver_zipModel_finetune.prototxt')
#need to write fine-tuning code

