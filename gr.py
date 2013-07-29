import math as m
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import os
from pycuda.compiler import SourceModule

file = open(os.path.dirname(__file__) + '/gr.cu')
cudaCode = file.read()
file.close()

def gaussRegression(data, frequencies):
    '''
    data is a ndata*nfrequency array
    frequencies is a nfrequency array
    returns an ndata*3 array with height, center frequency, and standard deviation
    '''
    ndata, nfrequencies = data.shape
    out = np.empty((ndata, 3), 'float32')
    mod = SourceModule(cudaCode %(nfrequencies, ndata))

    blockSize = 512
    gridSize = (ndata + blockSize - 1) / blockSize
    greg = mod.get_function('gaussReg')

    greg(cuda.In(data.T.astype('float32')), cuda.Out(out), cuda.In(frequencies.astype('float32')), np.float32(0.5),
            block=(blockSize,1,1), grid=(gridSize,1,1))

    return out

