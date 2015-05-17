import numpy as np
import matplotlib.pyplot as plt
import pickle

# Make sure that caffe is on the python path:
caffe_root = '/home/lisaanne/caffe-dev/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import scipy

import scipy.stats.mstats 

def normxcorrN(A,B):
    #assume input is N by x by y.  Assume correlating things that are the same dimension.
    channels = A.shape[0]
    pad_a = np.zeros((A.shape[1]+2*(A.shape[1]-1),A.shape[2]+2*(A.shape[2]-1)))
    pad_x = A.shape[1]
    pad_y = A.shape[2]
    result = np.zeros((channels, A.shape[1]*2-1, A.shape[2]*2-1))
    for channel in range(0,channels):
        pad_a[A.shape[1]-1:-(A.shape[1]-1),A.shape[2]-1:-(A.shape[2]-1)] = A[channel,:,:]
        b_mat = (B[channel,:,:] - np.mean(B[channel,:,:]))/np.std(B[channel,:,:])
        #b_mat = B[channel,:,:]
        for x in range(pad_x, 3*A.shape[1]-1):
            for y in range(pad_y, 3*A.shape[2]-1):
                a_mat = pad_a[x-pad_x:x,y-pad_y:y]
                a_mat = (a_mat-np.mean(a_mat))/np.std(a_mat)
                result[channel, x-pad_x, y-pad_y] = np.sum(a_mat*b_mat)/(A.shape[1]*A.shape[2])
                #result[channel, x-pad_x, y-pad_y] = np.sum(a_mat*b_mat)
    return result

def normxcorrNFast(A,B):
    #assume A is 1 by x by y and B is N by x by y.  Want to correlate N channels of B with 1 channel of A. Result will have N channels
    channels = B.shape[0]

    #create A vec which will be multiplied by each B vec
    count = 0
    pad_a = np.zeros((A.shape[1]+2*(A.shape[1]-1),A.shape[2]+2*(A.shape[2]-1)))
    pad_x = A.shape[1]
    pad_y = A.shape[2]
    result = np.zeros((channels, A.shape[1]*2-1, A.shape[2]*2-1))
    a_vec = np.zeros((A.shape[1]* A.shape[2],(A.shape[1]*2-1)*(A.shape[2]*2-1)))
    pad_a[A.shape[1]-1:-(A.shape[1]-1),A.shape[2]-1:-(A.shape[2]-1)] = A[0,:,:]
    for x in range(pad_x, 3*A.shape[1]-1):
      for y in range(pad_y, 3*A.shape[2]-1):
        a_mat = pad_a[x-pad_x:x,y-pad_y:y]
        a_mat = (a_mat-np.mean(a_mat))/np.std(a_mat)
        a_vec[:, count] = a_mat.reshape((a_mat.shape[0]*a_mat.shape[1],1))[:,0]
        count += 1

    for channel in range(0,channels):
        b_mat = (B[channel,:,:] - np.mean(B[channel,:,:]))/np.std(B[channel,:,:])
        b_vec = b_mat.reshape((1,B.shape[1]*B.shape[2]))
        result[channel, :, :] =  np.nan_to_num((np.dot(b_vec, a_vec).reshape((A.shape[1]*2-1, A.shape[2]*2-1)))/(A.shape[1]*A.shape[2]))
    return result

#set up nets:

caffe.set_mode_cpu()
home_dir = '/home/lisaanne/caffe-dev/examples/analyze_weights/'
MODEL = home_dir + 'cifar10_full_deploy.prototxt'
PRETRAINED = ['trained_models/cifar10_full_model1_iter_70000.caffemodel',
              'trained_models/snapshots_rs_107583_iter_70000.caffemodel', 
              'trained_models/snapshots_rs_11255_iter_70000.caffemodel', 
              'trained_models/snapshots_rs_52681_iter_70000.caffemodel', 
              'trained_models/snapshots_rs_80573_iter_70000.caffemodel']
nets = []
for model in PRETRAINED:
    nets.append(caffe.Net(MODEL, home_dir+model, caffe.TEST))
    
nets_out = []
for net in nets:
    nets_out.append(net.forward())

#Find order in reference to net 1
activations_net0 = nets[0].blobs['pool1'].data

#a = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
#b = np.array([[[1,2,1],[2,1,2],[1,1,1]]])
#outcheck = normxcorrN(a,b)
#outfast = normxcorrNFast(a,b)
print 'checking fast implemenation'
filter_map_all_nets = []
for net_num in range(1,5):
  print 'On net ', net_num
  activations_net1 = nets[net_num].blobs['pool1'].data
  filter_map = []
  filter_map_maxes = []
  for f in range(0,32):
    max_corr = []
    print 'On filter ', f
    for im in range(0,100):
      if im % 20 == 0:
        print 'On image ', im
      a = activations_net0[im:im+1,f,...]
      b = activations_net1[im:im+1,...]
      res = normxcorrNFast(a,b[0,...])
      max_corr.append(np.max(np.max(res,1),1))
    filter_map.append(max_corr)
  print 'For net %d, filter_map_maxes is %s' %(net_num, filter_map_maxes)
  #pickle.dump(filter_map, open('cifar_conv1_filter_maps_net%d.p' %net_num,'wb'))


