import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from timeit import default_timer as timer

#@cuda.jit(argtypes=[f4[:,:], f4[:,:,:], f4[:,:,:]])
#def corr_cuda(a_vec,B, result):
#  channel = cuda.grid(1)
#  b_mat = (B[channel,:,:] - np.mean(B[channel,:,:]))/np.std(B[channel,:,:])
#  b_vec = b_mat.reshape((1,B.shape[1]*B.shape[2]))
#  result[channel,:,:] =  np.nan_to_num((np.dot(b_vec, a_vec).reshape((B.shape[1]*2-1, B.shape[2]*2-1)))/(B.shape[1]*B.shape[2]))
  


def corr_python_parallel(a_vec,B,channel):
  b_mat = (B[channel,:,:] - np.mean(B[channel,:,:]))/np.std(B[channel,:,:])
  b_vec = b_mat.reshape((1,B.shape[1]*B.shape[2]))
  result =  np.nan_to_num((np.dot(b_vec, a_vec).reshape((B.shape[1]*2-1, B.shape[2]*2-1)))/(B.shape[1]*B.shape[2]))
  return result

#def corrFourier(a_vec, B, channel)

class corr_object(object):
  def __init__(self, A, B):
    self.A = A
    self.B = B
  def __call__(self, channel):
    return corr_python_parallel(self.A,self.B,channel)

def normxcorrNFastVector(A,B):
    #assume A is 1 by x by y and B is N by x by y.  Want to correlate N channels of B with 1 channel of A. Result will have N channels

    channels = B.shape[0]
    #create A vec which will be multiplied by each B vec
    count = 0
    pad_a = np.zeros((A.shape[1]+2*(A.shape[1]-1),A.shape[2]+2*(A.shape[2]-1)))
    pad_x = A.shape[1]
    pad_y = A.shape[2]
    a_vec = np.zeros((A.shape[1]* A.shape[2],(A.shape[1]*2-1)*(A.shape[2]*2-1)))
    pad_a[A.shape[1]-1:-(A.shape[1]-1),A.shape[2]-1:-(A.shape[2]-1)] = A[0,:,:]

    for x in range(pad_x, 3*A.shape[1]-1):
      for y in range(pad_y, 3*A.shape[2]-1):
        a_mat = pad_a[x-pad_x:x,y-pad_y:y]
        a_mat = (a_mat-np.mean(a_mat))/np.std(a_mat)
        a_vec[:, count] = a_mat.reshape((a_mat.shape[0]*a_mat.shape[1],1))[:,0]
        count += 1

    BR = np.reshape(B,(B.shape[0], B.shape[1]*B.shape[2]))
    B_N =  (BR-np.tile(np.mean(BR,1),(B.shape[1]*B.shape[2],1)).T)/np.tile(np.std(BR,1),(B.shape[1]*B.shape[2],1)).T
    AB = np.nan_to_num(np.dot(B_N, a_vec).reshape((B.shape[0],B.shape[1]*2-1,B.shape[2]*2-1)))/(B.shape[1]*B.shape[2])

    return AB

def normxcorrNFast(A,B):
    #assume A is 1 by x by y and B is N by x by y.  Want to correlate N channels of B with 1 channel of A. Result will have N channels
#    A = np.array([[[1,2,3],[4,5,6],[1,3,5]]])
#    B =  np.array([[[3,2,1],[5,8,7],[4,1,5]]])

    channels = B.shape[0]

    p = Pool(16)

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

    corr_objectAB = corr_object(a_vec, B)

    ts = timer()
    result_list = p.map(corr_objectAB, range(0,channels))
    p.close()
    p.join()
 
    for i,r in enumerate(result_list):
      result[i,:,:] = r
 
    return result

def main():
  n = 50
  A = np.array(np.random.random((1, n, n)), dtype=np.float32)
  B = np.array(np.random.random((n, n, n)), dtype=np.float32)
  ts = timer()
  res = normxcorrNFast(A,B)
  te = timer()
  print 'Old way took a total of %f seconds.' %(te-ts) 
  
  ts = timer()
  res2 = normxcorrNFastVector(A,B)
  te = timer()
  print 'New way takes a total of %f seconds.' %(te-ts) 

  print 'Min and max differences are %f/%f.' %(np.min(res-res2), np.max(res-res2))

if __name__ == '__main__':
  main()



