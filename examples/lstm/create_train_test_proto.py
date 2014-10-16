#!/usr/bin/python

import re
import sys
import getopt
import os.path

#TO DO: write code to check if level_dbs are currently in use

#function takes a template [train/test].prototxt file and creates a new [train/test].prototxt file based on input arguments. 
#Can change template by using argument template:my_template_file.prototxt.
#Can change [train/test] prototxt file name with the input output_file:output_file.prototxt. 
#include arguments as follows:
	#name:[path to param]:value to change to
#Examples
	#./create_train_test.prototxt data:data_param:clip_mode:LSTM fc6:blobs_lr:1 fc6:blobs_lr:2

#NOTES
#some paramters (such as blobs_lr) will have the same parameter names.  
#Put values in as follows:  
	#./create_train_test.prototxt conv1:blobs_lr:1 conv1:blobs_lr:2
#This will result in blobs_lr being set as below:
	#blobs_lr: 1
	#blobs_lr: 2

#unless specifically indicated in input arguments, train/test leveldb are assumed ot have the same name format

default_leveldb = 'hmdb_train_split_all_newDB_Get_1_leveldb'
default_template = 'hmdb_recurrent_train_ARCH2.prototxt' 
default_output_file = 'generated_hmdb_recurrent_train.prototxt'

def check_endswith_prototxt(item):
    item = item.replace('\n','')
    s = item.split('.')
    if (s[-1] != 'prototxt'):
      item = '%s.prototxt' %(item)
    return item

def format_proto_string(write_line, string):
  m = re.search(':.(("*[A-Z]*[a-z]*[1-9]*_*)*)',write_line)
  prefix = write_line.split(m.group())
  write_line = '%s: %s\n' %(prefix[0],string)
  return write_line

def check_equal(keys_a, keys_b, my_dict,l_a=0, l_b = 0):
  if check_keys(keys_a,my_dict):
    value_a = get_dict_value(keys_a, my_dict, l_a)
    if check_keys(keys_b,my_dict):
      value_b = get_dict_value(keys_b, my_dict, l_b)
      if value_a == value_b:
        return my_dict
      else: 
        return add_dict_value(keys_b,value_a,my_dict,True) 
    else:
      return add_dict_value(keys_b,value_a,my_dict)
  elif check_keys(keys_b,my_dict):
    value_b = get_dict_value(keys_b, my_dict, l_b)
    return add_dict_value(keys_a, value_b, my_dict)
  else:
    return my_dict 

def add_dict_value(keys,value,my_dict,rewrite=False):
  if len(keys) == 1:
    if keys[0] in my_dict.keys():
      if rewrite:
        my_dict[keys[0]] = value
      else:
        my_dict[keys[0]].extend(value)
    else:
        my_dict[keys[0]] = value
  else:
    if not keys[0] in my_dict.keys():
      my_dict[keys[0]] = {}
    my_dict[keys[0]] = add_dict_value(keys[1:],value,my_dict[keys[0]],rewrite)
  return my_dict

def check_keys(keys,my_dict):

  if len(keys) == 1:
    if keys[0] in my_dict.keys():
      return True
    else:
      return False
  else:
    if not keys[0] in my_dict.keys():
      return False
    else:
      return check_keys(keys[1:],my_dict[keys[0]])

def get_dict_value(keys,my_dict,l=-1):
  if len(keys) == 1:
    if l > -1:
      return [my_dict[keys[0]][l]]
    else:
      return my_dict[keys[0]]
  else:
    return get_dict_value(keys[1:],my_dict[keys[0]],l)

def check_leveldb(keys, my_dict):
  
  if not check_keys(keys, my_dict): 
    add_dict_value(keys, [default_leveldb], my_dict)

  leveldb = get_dict_value(keys, my_dict)
  
  if len(leveldb) < 2:
    if re.match('(.*)train(.*)',leveldb[0]):
      leveldb_train = leveldb[0]
      leveldb_test = leveldb_train.replace('train','test')
    elif re.match('(.*)test(.*)',leveldb[0]):
      leveldb_test = leveldb[0]
      leveldb_train = leveldb_test.replace('test','train') 
    else:
      print 'Do not indicate if leveldb is for test or train test!'
      return
    return add_dict_value(keys, [leveldb_train,leveldb_test],my_dict,True)
  else:
    return

def main(args):

  proto_dict = {}

  #create proto_dict based on arguments

  for arg in args:
    s = arg.split(':')
    value = s[-1:]
    keys = s[:-1]
    proto_dict = add_dict_value(keys,value,proto_dict) 

  #put in default values

  if not check_keys(['template'],proto_dict):
    proto_dict = add_dict_value(['template'],[default_template],proto_dict)
  if not check_keys(['output_file'],proto_dict):
    proto_dict = add_dict_value(['output_file'],[default_output_file],proto_dict)
  if not check_keys(['data','data_param','source'],proto_dict):
    proto_dict = add_dict_value(['data','data_param','source'], \
               [default_leveldb],proto_dict)

  #call on functions that do special things for certain parameters

  #Some variables need to be consistent across parameters. Can use function check_equal to check.

  proto_dict = check_equal(['data','data_param','sLSTM'],\
               ['lstm1','lstm_param','buffer_size'],proto_dict) 

  proto_dict = check_equal(['data','data_param','LSTM_clip_length'],\
               ['data','data_param','clip_length'], proto_dict)
                
  #LEVLEDB
  #make sure two values in leveldb field, and if not, assume test and train leveldb have the same name format
  proto_dict = check_leveldb(['data','data_param','source'],proto_dict)

  #Begin writing proto file

  f_template = open(proto_dict['template'][0],'rb')
  f_output = open(proto_dict['output_file'][0], 'wb')

  mini_dict = {}
  key_tracker = ['tmp']
  for line in f_template:
    if (not line.startswith('#')): #if not m, not a comment
      write_line = line
      #need to count leading whitspaces in line...
      line = line.replace(" ", "")
      line = line.replace("\n","")
      line = line.replace("{","")
      line = line.replace('"','')
      strings = line.split(':')

      if (strings[0] == 'name'):
        name = strings[1]
        key_tracker = []
        key_tracker.append(name)

      key_tracker.append(strings[0])
      if check_keys(key_tracker,proto_dict):
        tmp = get_dict_value(key_tracker,proto_dict)
        if not type(tmp) == dict:
          write_line = write_line.replace(strings[1],tmp[0])
          if len(tmp) > 1:
            proto_dict = add_dict_value(key_tracker,tmp[1:],proto_dict,True)
          key_tracker.pop()
      else:
        key_tracker.pop()

      if strings[0] == '}':
        if len(key_tracker) > 0:
          key_tracker.pop()
        else:
          key_tracker = ['tmp']
###This could be cleaner
#      if (strings[0] == 'name'):
#        name = strings[1]
#        key_tracker = []
#        key_tracker.append(name)  
#        if name in proto_dict.keys():
#          mini_dict = {}
#          mini_dict = proto_dict[name]    
#      elif strings[0] in mini_dict.keys():
#        if type(mini_dict[strings[0]]) == dict:
#          mini_dict = mini_dict[strings[0]]       
#        else:
#          write_line = write_line.replace(strings[1],mini_dict[strings[0]][0])      
#        key_tracker.append(strings[0])
#
#      #need to do something special when reading source field
#      print key_tracker
#      if check_keys(key_tracker,proto_dict): 
#        value = get_dict_value(key_tracker, proto_dict)
#        if not type(value) == dict:
#          if len(value) > 1:
#            value = value[1:]
#            proto_dict = add_dict_value(key_tracker, value, proto_dict,True) 
###


      f_output.write(write_line)       
        
  f_template.close()
  f_output.close()


  print 'Wrote file %s' %(proto_dict['output_file'])

if __name__ == '__main__':
  main(sys.argv[1:])


