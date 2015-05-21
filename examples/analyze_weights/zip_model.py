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
home_dir = '/home/lisaanne/caffe-forward-backward/examples/analyze_weights'

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
  
  def __init__(self, num_models, MODEL, PRETRAINED):

    self.num_models = num_models

    self.home_dir = home_dir
    nets = []
    for model in PRETRAINED:
       nets.append(caffe.Net(MODEL, home_dir+model, caffe.TEST))

    self.trained_nets = nets
    self.MODEL = MODEL
    self.PRETRAINED = PRETRAINED

  def concatWeights(self,convlayers, iplayers):
    nets = self.trained_nets
    #put conv parameters into zipNet   
    for i, net in enumerate(nets[0:self.num_models]):
      for l in convlayers: 
        f = net.params[l][0].data.shape[0]
        c = net.params[l][0].data.shape[1]
        if l == 'conv1':  
          self.netTEST.params[l][0].data[i*f:i*f+f,...] = copy.deepcopy(net.params[l][0].data)
          self.netTEST.params[l][1].data[i*f:i*f+f] = copy.deepcopy(net.params[l][1].data)
          self.netTRAIN.params[l][0].data[i*f:i*f+f,...] = copy.deepcopy(net.params[l][0].data)
          self.netTRAIN.params[l][1].data[i*f:i*f+f] = copy.deepcopy(net.params[l][1].data)
        else:
          self.netTEST.params[l][0].data[i*f:i*f+f,i*c:i*c+c,...] = copy.deepcopy(net.params[l][0].data)
          self.netTEST.params[l][1].data[i*f:i*f+f] = copy.deepcopy(net.params[l][1].data)
          self.netTRAIN.params[l][0].data[i*f:i*f+f,i*c:i*c+c,...] = copy.deepcopy(net.params[l][0].data)
          self.netTRAIN.params[l][1].data[i*f:i*f+f] = copy.deepcopy(net.params[l][1].data)

    #put ip parameters into zipNet   
    for l in iplayers:
      self.netTEST.params[l][1].data[...] = 0
      self.netTRAIN.params[l][1].data[...] = 0
    for i, net in enumerate(nets[0:self.num_models]):
      for l in iplayers:
        f = net.params['ip1'][0].data.shape[1] 
        self.netTEST.params[l][0].data[:,i*f:i*f+f,...] = net.params[l][0].data
        self.netTEST.params[l][1].data[...] += net.params[l][1].data
        self.netTRAIN.params[l][0].data[:,i*f:i*f+f,...] = net.params[l][0].data
        self.netTRAIN.params[l][1].data[...] += net.params[l][1].data
  
  def initModel(self, proto, trainProto, replace_dict, tmp_save_proto):

    for p in [proto, trainProto]:
      fNetProto = open(home_dir + p, 'rb')
      protoLines = fNetProto.readlines()
      fNetProto.close()
      for key in replace_dict.keys():
        protoLines = [x.replace(key,replace_dict[key]) for x in protoLines]

      fNetTmp = open(home_dir+tmp_save_proto + p, 'wb')
      for line in protoLines:
        fNetTmp.write(line)
      fNetTmp.close()

    self.netTEST = caffe.Net(home_dir + tmp_save_proto + proto, caffe.TEST)
    self.netTRAIN = caffe.Net(home_dir + tmp_save_proto + proto, caffe.TRAIN)
    self.convlayers = []
    self.iplayers = [] 
    for l in self.netTEST.params.keys():
      if len(self.netTEST.params[l][0].data.shape) == 2:
        self.iplayers.append(l)
      else:
        self.convlayers.append(l)
    self.layers = self.convlayers
    self.layers.extend(self.iplayers)
    self.similarity = {}
    self.graft_dict = {}
    self.template_proto = proto
    self.save_proto = tmp_save_proto + proto
    self.template_train_proto = trainProto 
    self.train_proto = tmp_save_proto + trainProto
    self.replace_dict = replace_dict

  def ftZipModel(self, solver, snap):
    fSolver = open(home_dir + solver, 'rb')
    sLines = fSolver.readlines()
    fSolver.close()
    sLines = [x.replace('NET', self.train_proto) for x in sLines]
    sLines = [x.replace('SNAPSHOT', snap) for x in sLines]
    fSolverTmp = open(home_dir + 'solver_tmp.prototxt', 'wb')
    for line in sLines:
      fSolverTmp.write(line)
    fSolverTmp.close()

    solver = caffe.get_solver('solver_tmp.prototxt')
    for layer in solver.net.params.keys():
      for i in range(len(solver.net.params[layer])):
        solver.net.params[layer][i].data[...] = copy.deepcopy(self.netTEST.params[layer][i].data)
    solver.solve()
    
    for layer in solver.net.params.keys():
      for i in range(len(solver.net.params[layer])):
        self.netTEST.params[layer][i].data[...] = copy.deepcopy(solver.net.params[layer][i].data)
        self.netTRAIN.params[layer][i].data[...] = copy.deepcopy(solver.net.params[layer][i].data)
    
    print self.testZipNet()


  def loadSimilarity(self, similarityFiles, layer):
    self.similarity[layer] = {}
    if not len(similarityFiles) == self.num_models:
      print 'Number similarityFiles needs to be the same as num_models!'
      return
    else:
      for i, similarityFile in enumerate(similarityFiles):
        loadFile = pickle.load(open(similarityFile,'rb'))
        self.similarity[layer][i] = loadFile['simMeasure']

  def determineSimilarity(self, layer, activation, similarityFiles=None) :
  #determine similarity for a certain layer and save to similarityFiles
    numFilters = self.trained_nets[0].params[layer][0].data.shape[0]

    #check that similarityFiles are equal in number to models
    if similarityFiles:
      if not len(similarityFiles) == self.num_models:
        print 'Number of save files must be equal to number of models!'
        pass

    for n in range(0,self.num_models):
      from_net = n
      simMeasure_all = []
      for f in range(0,numFilters):
        print 'Computing similarity for filter %d, net %d.' %(f, n)
        simMeasure_nets = []
        for nn in range(0,self.num_models):
          if n == nn:
            pass
          else:
            onto_net = nn
            origFilter = f+from_net*32
            similarFilter, simMeasure = self.findSimilar(from_net, onto_net, layer, activation, f)
            simMeasure_nets.append(simMeasure)
      simMeasure_all.append(simMeasure_nets)
      if similarityFiles:
        save_data['simMeasure'] = simMeasure_all
        pickle.dump(save_data, open(similarityFiles[n],'wb'))
      self.similarity[layer][n] = simMeasure_all 
      

  def determineGraftDictV1(self, layer):
    #greedy search of filters    
    filters_per_model = self.netTRAIN.params[layer][0].data.shape[0]/self.num_models
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

          fillFilter = similar_filters[0]          
	  
          #difference in accuracy used to create first graft dict
          #fillFilter, acc = pickBestAccuracy(self.net, similar_filters,layer) #do based off similarity
          #print 'For filter %d accuracy is %f.' %(base_filter, acc)

          for sf in similar_filters:
            graft_dict[sf] = fillFilter
    self.graft_dict[layer] =  graft_dict
    #save_data = {}
    #save_data['graft_dict'] = self.graft_dict
    #pickle.dump(save_data, open('results/graft_dict_conv1_V1.p','wb'))

  def loadGraftDict(self, graft_file):
    f = pickle.load(open(graft_file,'rb'))
    self.graft_dict = f['graft_dict']

  def testGraftDict(self, layer):
    graft_dict = self.graft_dict[layer]
    for key in graft_dict.keys():
      self.netTEST.params[layer][0].data[key,...] = copy.deepcopy(self.netTEST.params[layer][0].data[graft_dict[key],...])
      self.netTEST.params[layer][1].data[key] = copy.deepcopy(self.netTEST.params[layer][1].data[graft_dict[key]])
    print 'Accuracy for graft dict on layer %s is %f.' %(layer, self.testNet())
    fNet.concatWeights(['conv1','conv2','conv3'],['ip1'])
  
  def testZipNet(self):
    num_videos = 0
    num_correct = 0
    for i in range(0,100):
      out = self.netTEST.forward()
      probs = out['probs']
      labels = out['label-out']

      labels_pred_conv1 = np.argmax(probs[:,0:10],1)
      num_correct += len(np.where(labels_pred_conv1 == labels)[0])

      num_videos += 100
    return float(num_correct)/num_videos

                     
  def modelZip(self, layer, protoLayer, tmp_save_proto):
    #create and return a new zippedModel which is zipped at indicated layer

    graft_dict = self.graft_dict[layer]
    zipNet = zippedModel(self.num_models, self.MODEL, self.PRETRAINED) 
    layer_index = self.layers.index(layer)
    next_layer = self.layers[layer_index+ 1] 
    layers_below = self.layers[:layer_index]
    layers_above = self.layers[layer_index+2:]
    
    #organize layer filters
    layer_filters = np.unique(np.array(graft_dict.values()))
    layer_filters_dict = {}
    for key in graft_dict.keys():
      layer_filters_dict[key] = np.where(layer_filters == graft_dict[key])[0][0]
    num_filters = len(layer_filters)
    origNumFilters = self.netTEST.params[layer][0].data.shape[0]/self.num_models

    #create new zipnet
    replace_dict = self.replace_dict 
    replace_dict[protoLayer] = str(num_filters)
    zipNet.initModel(self.template_proto, self.template_train_proto, replace_dict, tmp_save_proto)

    #Copy parameters into zip layer
    for i, lf in enumerate(layer_filters):
      zipNet.netTEST.params[layer][0].data[i,...] = copy.deepcopy(self.netTEST.params[layer][0].data[lf,...])
      zipNet.netTEST.params[layer][1].data[i] = copy.deepcopy(self.netTEST.params[layer][1].data[lf])
      zipNet.netTRAIN.params[layer][0].data[i,...] = copy.deepcopy(self.netTRAIN.params[layer][0].data[lf,...])
      zipNet.netTRAIN.params[layer][1].data[i] = copy.deepcopy(self.netTRAIN.params[layer][1].data[lf])

    #shuffle channels from layer up to correspond to zipped layers if next layer is a conv layer
    if next_layer in self.convlayers:
      for key in layer_filters_dict.keys():
        origNet = key/origNumFilters
        filterRange = range(origNet*origNumFilters,origNet*origNumFilters+origNumFilters)
        zipNet.netTEST.params[next_layer][0].data[filterRange,layer_filters_dict[key],...] = copy.deepcopy(self.netTEST.params[next_layer][0].data[filterRange, key,...])
        zipNet.netTRAIN.params[next_layer][0].data[filterRange,layer_filters_dict[key],...] = copy.deepcopy(self.netTEST.params[next_layer][0].data[filterRange, key,...])
      for i,b in enumerate(self.netTEST.params[next_layer][1].data):
        zipNet.netTEST.params[next_layer][1].data[i] = copy.deepcopy(b)
      for i,b in enumerate(self.netTRAIN.params[next_layer][1].data):
        zipNet.netTRAIN.params[next_layer][1].data[i] = copy.deepcopy(b)
    else:
      dim_filter = zipNet.netTest.params[layer][0].data.shape
      size_filter = dim_filter[1]*dim_filter[2]*dim_filter[3]
      for key in layer_filters_dict.keys():
        filterRangeFrom = range(key*size_filter, key*size_filter + size_filter)
        filterRangeTo = range(layer_filters_dict[key]*size_filter, layer_filters_dict[key]*size_filter + size_filter)
        zipNet.netTEST.params[next_layer][0].data[:,filterRangeTo] += self.netTEST.params[next_layer][0].data[:,filterRangeFrom]
        zipNet.netTRAIN.params[next_layer][0].data[:,filterRangeTo] += self.netTRAIN.params[next_layer][0].data[:,filterRangeFrom]
      zipNet.netTEST.params[next_layer][1].data[...] = copy.deepcopy(self.netTEST.params[next_layer][1].data)
      zipNet.netTRAIN.params[next_layer][1].data[...] = copy.deepcopy(self.netTRAIN.params[next_layer][1].data)
   
    other_layers = layers_below + layers_above
    for l in other_layers:
      zipNet.netTEST.params[l][0].data[...] = copy.deepcopy(self.netTEST.params[l][0].data[...])
      zipNet.netTEST.params[l][1].data[...] = copy.deepcopy(self.netTEST.params[l][1].data[...])
      zipNet.netTRAIN.params[l][0].data[...] = copy.deepcopy(self.netTRAIN.params[l][0].data[...])
      zipNet.netTRAIN.params[l][1].data[...] = copy.deepcopy(self.netTRAIN.params[l][1].data[...])


    return zipNet

