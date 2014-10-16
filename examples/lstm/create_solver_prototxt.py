#!/usr/bin/python

import re
import sys
import getopt
import os.path
import glob

#function takes a template solver.prototxt file and creates a new solver.prototxt file
#based on input arguments.  A default template file and output file name are set
#but can be changed.  To set any parameter, input the following argument:
	# parameter:value_of_parameter
#To change the template, input template:my_template.prototxt
#To change the output file, input output_file:my_outputfile.prototxt
#Example use:
	# ./create_solver_prototxt.py template:my_template.prototxt gamma:0.1 
        #                             stepsize:1000 net:hmdb_LSTM.prototxt

default_template = 'solver_hmdb_lstm.prototxt';
default_output_file = 'generated_solver_hmdb_lstm.prototxt'
#used to check if we need to put quotes around parameter in prototxt
quote_pattern = '\s"(.*)"'

#check that template file exitsts.  If not uses default template.
def check_template(template):
  if not os.path.isfile(template):
     print 'File %s does not exist; using default template %s' %(template,default_template)
     template = default_template;
  return template

#check that snapshot file does not already exist
def check_snapshot(snapshot):
  base_snapshot = snapshot
  count = 0
  while len(glob.glob('%s*' %(snapshot))) > 0:
    count += 1
    snapshot = '%s_%d' %(base_snapshot,int(count))
    print '%s already exits.  Trying: %s' %(base_snapshot, snapshot)
  return snapshot

#checks that output file is the right format
def check_output_file(output_file):
  #don't really care if we overwrite an ouput file
  if not output_file.endswith('.prototxt'):      
    output_file = '%s.prototxt' %(output_file)
  return output_file 

#make the parameter have quotes
def make_quote(parameter):
  if re.match(quote_pattern,parameter) is None:
    parameter = '"%s"' %(parameter)
  return parameter
  

#creates dict containing all the parameters to change.
#reads template prototxt file and changes accordingly
def main(args):
  proto_dict = {};
  proto_dict['template'] = default_template;
  proto_dict['output_file'] = default_output_file;

  #put all arguments in dict structure
  for arg in args:
    s = arg.split(':')
    if s[0] in proto_dict.keys():
      proto_dict[s[0]] = s[1]
    else:
      proto_dict[s[0]] = {}
      proto_dict[s[0]] = s[1]

  #call on functions that do special things for certain parameters

  #make sure template exits.  
  proto_dict['template'] = check_template(proto_dict['template'])

  #make sure we do not over write snapshot
  if 'snapshot_prefix' in proto_dict.keys():
    proto_dict['snapshot_prefix'] = check_snapshot(proto_dict['snapshot_prefix'])

  #check output file has valid name
  proto_dict['output_file'] = check_output_file(proto_dict['output_file'])

  #read in template file and change parameters as needed

  f_template = open(proto_dict['template'],'rb')
  f_proto = open(proto_dict['output_file'],'wb')


  for line in f_template:
    m = line.startswith('#')
    if (not m): #if not m, not a comment
      strings = line.split(':')
      parameter = strings[1]
      for s in range(2,len(strings)):
        parameter = '%s:%s' %(parameter,strings[s])
      if strings[0] in proto_dict.keys():
        parameter = proto_dict[strings[0]]
        if re.match(quote_pattern,strings[0]) is not None:
          parameter = make_quote(parameter)
        parameter = ' %s\n' %(parameter)
      #create line to write
      write_line = '%s:%s' %(strings[0], parameter)
      f_proto.write(write_line)

  f_template.close()
  f_proto.close()

  print 'Finished writing generated prototext to: %s' %(proto_dict['output_file'])

if __name__ == '__main__':
  main(sys.argv[1:])


#  just to remember how to use optlist if i want to use it...
#  try:
#    optlist, args = getopt.getopt(argv, 'a',['test_iter=','device_id='])
#  except getopt.GetoptError:
#    print 'There is an issue.'
#    sys.exit(2)

