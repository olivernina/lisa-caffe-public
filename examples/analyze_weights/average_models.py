import numpy as np
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'cifar10_full_deploy.prototxt'
PRETRAINED = ['trained_models/cifar10_full_model1_iter_70000.caffemodel',
              'trained_models/snapshots_rs_107583_iter_70000.caffemodel', 
              'trained_models/snapshots_rs_11255_iter_70000.caffemodel', 
              'trained_models/snapshots_rs_52681_iter_70000.caffemodel', 
              'trained_models/snapshots_rs_80573_iter_70000.caffemodel', 
              'trained_models/snapshots_rs_1361_iter_70000.caffemodel', 
              'trained_models/snapshots_rs_4572_iter_70000.caffemodel',
              'trained_models/snapshots_rs_5916_iter_70000.caffemodel']
#PRETRAINED = ['cifar10_full_model1_iter_70000.caffemodel','snapshots_rs_107583_iter_70000.caffemodel', 'snapshots_rs_11255_iter_70000.caffemodel', 'snapshots_rs_52681_iter_70000.caffemodel', 'snapshots_rs_80573_iter_70000.caffemodel',]

out_probs = np.zeros((10000,10))
out_labels = np.zeros((10000,))
track_accuracy = []
track_accuracy_classes = {}
for i in range(0,10):
  track_accuracy_classes[i] = []
num_models = 2

for model in PRETRAINED[0:num_models]:

  caffe.set_mode_gpu()
  net = caffe.Net(MODEL_FILE, model, caffe.TEST)

  out_probs_temp = np.zeros((10000,10))
  out_labels_temp = np.zeros((10000,))

  for i in range(0,100):
    out = net.forward()
    out_probs_temp[i*100:i*100+100,:] = out['probs']
    out_labels_temp[i*100:i*100+100] = out['label']
  
  predicted_labels = np.argmax(out_probs_temp, 1)
  num_correct = np.where((out_labels_temp == predicted_labels))[0].shape[0]
  accuracy = float(num_correct)/10000.
  track_accuracy.append(accuracy)
  print "Accuracy for %s is %f." %(model, accuracy)
  for i in range(0,10):
    locs = np.where(out_labels_temp == i)
    pred_i = predicted_labels[locs]
    num_correct = np.where(pred_i == i)[0].shape[0]
    track_accuracy_classes[i].append(float(num_correct)/1000.)

  out_probs = out_probs + out_probs_temp
  out_labels = out_labels + out_labels_temp

out_probs /= num_models
out_labels /= num_models 
  
predicted_labels = np.argmax(out_probs, 1)
num_correct = np.where((out_labels == predicted_labels))[0].shape[0]
accuracy = float(num_correct)/10000.

track_accuracy_classes_average = []
for i in range(0,10):
  locs = np.where(out_labels == i)
  pred_i = predicted_labels[locs]
  num_correct = np.where(pred_i == i)[0].shape[0]
  track_accuracy_classes_average.append(float(num_correct)/1000.)

print track_accuracy
print "Accuracy for all is %f." %( accuracy)

