import numpy as np
caffe_root = '/home/lisa/caffe-LSTM-video/'
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
from multiprocessing import Pool
home_dir = '/home/lisa/caffe-LSTM-video/examples/analyze_weights/'
from timeit import default_timer as timer

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

def corrManyFunctions(Mats):
  return normxcorrNFastVector(Mats[0],Mats[1][0,...])

class corrMany(object):
  def __init__(self, A):
    self.A = A
  def __call__(B):
    return normxcorrNFastVector(self.A, B) 

class zippedModel(frankenNet.frankenNet):
  
  def __init__(self, num_models, MODEL, PRETRAINED, device_id=0):
     # init initializes attributes; most importantly the nets which will be combined
    self.num_models = num_models
    self.device_id = device_id
    caffe.set_device(device_id)
    self.home_dir = home_dir
    nets = []
    for model in PRETRAINED[0:self.num_models]:
       nets.append(caffe.Net(MODEL, model, caffe.TEST))

    self.trained_nets = nets
    self.MODEL = MODEL
    self.PRETRAINED = PRETRAINED

    self.grouping = {}
    self.grouping['conv1'] = 1
    self.grouping['conv2'] = 2
    self.grouping['conv3'] = 1
    self.grouping['conv4'] = 2
    self.grouping['conv5'] = 2
    

  def concatWeights(self,convlayers, iplayers):
    grouping = self.grouping
    nets = self.trained_nets
    #put conv parameters into zipNet   
    for l in convlayers + iplayers:
      self.netTEST.params[l][0].data[...] = 0
      self.netTRAIN.params[l][0].data[...] = 0
      self.netTEST.params[l][1].data[...] = 0
      self.netTRAIN.params[l][1].data[...] = 0
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
          for g in range(grouping[l]):
            gf = f/grouping[l] 
            gc = c
            c2 = self.netTEST.params[l][0].data.shape[1]/self.num_models 
            self.netTEST.params[l][0].data[i*f+g*gf:i*f+g*gf+gf,i*c2+g*gc:i*c2+g*gc+gc,...] = copy.deepcopy(net.params[l][0].data[g*gf:gf+g*gf,...])
            self.netTEST.params[l][1].data[i*f+g*gf:i*f+g*gf+gf] = copy.deepcopy(net.params[l][1].data[g*gf:gf+g*gf])
            self.netTRAIN.params[l][0].data[i*f+g*gf:i*f+g*gf+gf,i*c2+g*gc:i*c2+g*gc+gc,...] = copy.deepcopy(net.params[l][0].data[g*gf:gf+g*gf,...])
            self.netTRAIN.params[l][1].data[i*f+g*gf:i*f+g*gf+gf] = copy.deepcopy(net.params[l][1].data[g*gf:gf+g*gf])

    #put ip parameters into zipNet   
    for i, net in enumerate(nets[0:self.num_models]):
      for l in iplayers:
        if l == iplayers[-1]: #the output will be equal to the sum of all nets concatenated together
          f = net.params[l][0].data.shape[1] 
          self.netTEST.params[l][0].data[:,i*f:i*f+f,...] = copy.deepcopy(net.params[l][0].data)
          self.netTEST.params[l][1].data[...] += copy.deepcopy(net.params[l][1].data)
          self.netTRAIN.params[l][0].data[:,i*f:i*f+f,...] = copy.deepcopy(net.params[l][0].data)
          self.netTRAIN.params[l][1].data[...] += copy.deepcopy(net.params[l][1].data)
        else:
          f1 = net.params[l][0].data.shape[0] 
          f2 = net.params[l][0].data.shape[1] 
          self.netTEST.params[l][0].data[i*f1:i*f1+f1,i*f2:i*f2+f2,...] = copy.deepcopy(net.params[l][0].data)
          self.netTEST.params[l][1].data[i*f1:i*f1+f1] = copy.deepcopy(net.params[l][1].data)
          self.netTRAIN.params[l][0].data[i*f1:i*f1+f1,i*f2:i*f2+f2,...] = copy.deepcopy(net.params[l][0].data)
          self.netTRAIN.params[l][1].data[i*f1:i*f1+f1] = copy.deepcopy(net.params[l][1].data)
#    o_net0 = nets[0].forward()
#    o_net1 = nets[1].forward()
#    o_netC = self.netTEST.forward()
#    for l in nets[0].blobs.keys():
#      if not ((l == 'data') | (l == 'label') | (l=='fc8')):
#        f = nets[0].blobs[l].data.shape[1]
#        dif_net0 = nets[0].blobs[l].data - self.netTEST.blobs[l].data[:,0:f,...]
#        dif_net1 = nets[1].blobs[l].data - self.netTEST.blobs[l].data[:,f:,...]
#        print 'For layer %s, min/max difference from net0 is %f/%f.' %(l, np.min(dif_net0), np.max(dif_net0))
#        print 'For layer %s, min/max difference from net1 is %f/%f.' %(l, np.min(dif_net1), np.max(dif_net1))
    
      
    print 'Done concatenating model!'
  
  def initModelPartialNets(self, proto, replace_dict, tmp_save_proto):
    #proto is the template proto for the zipModel

    # write prototxts, load zip models, and sort out conv versus ip layers.
    fNetProto = open(home_dir + proto, 'rb')
    protoLines = fNetProto.readlines()
    fNetProto.close()
    for key in replace_dict.keys():
      protoLines = [x.replace(key,replace_dict[key]) for x in protoLines]

    fNetTmp = open(home_dir+tmp_save_proto, 'wb')
    for line in protoLines:
      fNetTmp.write(line)
    fNetTmp.close()

    self.netTEST = caffe.Net(home_dir + tmp_save_proto, caffe.TEST)
    self.netTRAIN = caffe.Net(home_dir + tmp_save_proto, caffe.TRAIN)
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
    self.save_proto = tmp_save_proto
    self.replace_dict = replace_dict
 
  def initModel(self, proto, trainProto, replace_dict, tmp_save_proto):
    # write prototxts, load zip models, and sort out conv versus ip layers.
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

  def findSimilar_indNets(self, refNet, compNet, layer, activation, f):
    #Find filter in compNet most similar to refNet.

    p = Pool(16)
    corr = []
    b_list = []
    for im in range(0,100):
      mats = [refNet.blobs[activation].data[im:im+1,f,...], compNet.blobs[activation].data[im:im+1,...]]
      b_list.append(mats)

    corrManyFunctions(b_list[0])
    res = np.array(p.map(corrManyFunctions, b_list))
    p.close()
    p.join()
  
    for im in range(0,100):
      maxes = res[im][:,int((res[im].shape[1]/2)+0.5), int((res[im].shape[2]/2)+0.5)]
      corr.append(maxes) 
    
    sums = np.zeros(corr[0].shape)
    for s in corr:
      sums += s
    sim_filter = np.argmax(sums)
    sim_measure = sums/100
    return sim_filter, sim_measure 
  
  def determineSimilarity_partialNet(self, layer, activation_in, activation_out, similarityFiles=None) :
    self.similarity[layer] = {}
    #determine similarity for a certain layer and save to similarityFiles
    numFilters = self.trained_nets[0].params[layer][0].data.shape[0]
    o_C = self.netTRAIN.forward()
    net_input = self.netTRAIN.blobs[activation_in].data
 
    #initialize net inputs as input activations from zipModel 
    for net in self.trained_nets:
      net.blobs['data'].data[...] = copy.deepcopy(net_input[0:100,...])
      net.forward()

    #check that similarityFiles are equal in number to models
    if similarityFiles:
      if not len(similarityFiles) == self.num_models:
        print 'Number of save files must be equal to number of models!'

    for n in range(0,self.num_models):
      from_net = self.trained_nets[n]
      simMeasure_all = []
      for f in range(0,numFilters):
        print 'Computing similarity for filter %d, net %d.' %(f, n)
        simMeasure_nets = []
        for nn in range(0,self.num_models):
          onto_net = self.trained_nets[nn]
          similarFilter, simMeasure = self.findSimilar_indNets(from_net, onto_net, layer, activation_out, f)
          simMeasure_nets.append(simMeasure)
        simMeasure_all.append(simMeasure_nets)
      if similarityFiles:
        save_data = {}
        save_data['simMeasure'] = simMeasure_all
        pickle.dump(save_data, open(similarityFiles[n],'wb'))
      self.similarity[layer][n] = simMeasure_all 

  def determineSimilarity(self, layer, activation, similarityFiles=None) :
    self.similarity[layer] = {}
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
          onto_net = nn
          origFilter = f+from_net*numFilters
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
  
  def testZipNet(self, iterations=100, display_iter = 10):
    num_videos = 0
    num_correct = 0
    for i in range(0,iterations):
      if i % display_iter == 0:
        print 'On iteration ', i
      out = self.netTEST.forward()
      probs = out['probs']
      labels = out['label-out']
      num_labels = probs.shape[1]
      labels_pred_conv1 = np.argmax(probs[:,0:num_labels],1)
      num_correct += len(np.where(labels_pred_conv1 == labels)[0])

      num_videos += probs.shape[0]
    return float(num_correct)/num_videos

  def netOutputs(self, iterations=100):
    num_videos = 0
    num_correct = 0
    
    for i in range(0,iterations):
      if i % 100 == 0:
        print 'On iteration ', i 
      out = self.netTEST.forward()
      probs = out['probs']
      labels = out['label-out']
      num_labels = probs.shape[1]
      if i == 0:
        labels_cat = labels
        probs_cat = probs
      else:
        labels_cat = np.concatenate((labels_cat, labels))
        probs_cat = np.concatenate((probs_cat, probs))
      labels_pred_conv1 = np.argmax(probs,1)
      num_correct += len(np.where(labels_pred_conv1 == labels)[0])

      num_videos += probs.shape[0]
    accuracy = float(num_correct)/num_videos
    return accuracy, labels_cat, probs_cat
                     
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

