import numpy as np
import matplotlib.pyplot as plt
import pickle

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

