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

mod = SourceModule(cudaCode)

def gaussRegression(data, frequencies):
    '''
    data is a ndata*nfrequency array
    frequencies is a nfrequency array
    returns an ndata*3 array with height, center frequency, and standard deviation
    '''
    data = data.T #change to nfrequency * ndata
    nfrequencies, ndata = data.shape
    out = np.empty((ndata, 3), 'float32')
    mod = SourceModule(cudaCode %(nfrequencies, ndata))

    blockSize = 512
    gridSize = (ndata + blockSize - 1) / blockSize
    greg = mod.get_function('gaussReg')

    cuda.memcpy_htod(mod.get_global('numFreqs')[0], np.int32(nfrequencies))
    cuda.memcpy_htod(mod.get_global('numPoints')[0], np.int32(ndata))

    greg(cuda.In(data.astype('float32')), cuda.Out(out), cuda.In(frequencies.astype('float32')), np.float32(0.5),
            block=(blockSize,1,1), grid=(gridSize,1,1))

    return out

def findPeaks(data, frequencies):
    '''
    data is a ndata*frequency array
    '''
    data = data.T
    nfrequencies, ndata = data.shape
    out = np.empty(ndata, 'float32')

    blockSize = 512
    gridSize = (ndata + blockSize - 1) / blockSize

    cuda.memcpy_htod(mod.get_global('numFreqs')[0], np.int32(nfrequencies))
    cuda.memcpy_htod(mod.get_global('numPoints')[0], np.int32(ndata))

    fp = mod.get_function('findPeaks')

    fp(cuda.In(data.astype('float32')), cuda.Out(out), cuda.In(frequencies.astype('float32')),
            block=(blockSize,1,1), grid=(gridSize,1,1))

    return out

