import math as m
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import os
from pycuda.compiler import SourceModule

file = open(os.path.dirname(os.path.abspath(__file__)) + '/gr.cu')
cudaCode = file.read()
file.close()

CC = 3e8

mod = SourceModule(cudaCode)
blockSize = 256

def dopplerShift(velocity):
    '''
    converts a velocity to a doppler shift frequency
    velocity is in km/sec
    '''
    return centerFreq * velocity * 1e3 / CC

def gaussRegression(cutoff=0.5):
    '''
    Does a gaussian regression on the contiguous subset of data
    containing ymax where y>ymax*cutoff.
    Returns an array of height, center, standard deviation.
    '''
    __checksize()
    out = np.empty((ndata, 3), 'float32')

    gridSize = (ndata + blockSize - 1) / blockSize
    greg = mod.get_function('reg')

    cuda.memcpy_htod(mod.get_global('numFreqs')[0], np.int32(nfrequencies))
    cuda.memcpy_htod(mod.get_global('numPoints')[0], np.int32(ndata))
    cuda.memcpy_htod(mod.get_global('centerFreq')[0], np.float32(centerFreq))

    greg(cuda.In(data.astype('float32')), cuda.Out(out), cuda.In(frequencies.astype('float32')),
            np.float32(cutoff), np.int8(False),
            block=(blockSize,1,1), grid=(gridSize,1,1))

    return out.reshape(datashape + (3,))

def quadRegression(cutoff=0.5):
    '''
    Does a quadratic regression on the contiguous subset of data
    containing ymax where y>ymax*cutoff.
    Returns a, b, c, where y=a(x-b)^2+c
    '''
    __checksize()
    out = np.empty((ndata, 3), 'float32')

    gridSize = (ndata + blockSize - 1) / blockSize
    greg = mod.get_function('reg')

    cuda.memcpy_htod(mod.get_global('numFreqs')[0], np.int32(nfrequencies))
    cuda.memcpy_htod(mod.get_global('numPoints')[0], np.int32(ndata))
    cuda.memcpy_htod(mod.get_global('centerFreq')[0], np.float32(centerFreq))

    greg(cuda.In(data.astype('float32')), cuda.Out(out), cuda.In(frequencies.astype('float32')),
            np.float32(cutoff), np.int8(True),
            block=(blockSize,1,1), grid=(gridSize,1,1))

    return out.reshape(datashape + (3,))

def findPeaks():
    '''
    Returns an array with size of dataset, with locations of frequency peaks.
    '''
    __checksize()
    if (frequencies.size != nfrequencies):
        raise Exception('dFreq array must have same dimension as last dim of dat')
    out = np.empty(ndata, 'float32')

    gridSize = (ndata + blockSize - 1) / blockSize

    cuda.memcpy_htod(mod.get_global('numFreqs')[0], np.int32(nfrequencies))
    cuda.memcpy_htod(mod.get_global('numPoints')[0], np.int32(ndata))
    cuda.memcpy_htod(mod.get_global('centerFreq')[0], np.float32(centerFreq))

    fp = mod.get_function('findPeaks')

    fp(cuda.In(data.astype('float32')), cuda.Out(out), cuda.In(frequencies.astype('float32')),
            block=(blockSize,1,1), grid=(gridSize,1,1))

    return out.reshape(datashape)

def numPeaks():
    '''
    Returns an array with size of dataset, with numbers of peaks
    '''
    __checksize()
    if (frequencies.size != nfrequencies):
        raise Exception('dFreq array must have same dimension as last dim of dat')
    out = np.empty(ndata, 'int32')

    gridSize = (ndata + blockSize - 1) / blockSize

    cuda.memcpy_htod(mod.get_global('numFreqs')[0], np.int32(nfrequencies))
    cuda.memcpy_htod(mod.get_global('numPoints')[0], np.int32(ndata))
    cuda.memcpy_htod(mod.get_global('centerFreq')[0], np.float32(centerFreq))

    fp = mod.get_function('countPeaks')

    fp(cuda.In(data.astype('float32')), cuda.Out(out), cuda.In(frequencies.astype('float32')),
            block=(blockSize,1,1), grid=(gridSize,1,1))

    return out.reshape(datashape)

def splitIntegral(windowSeparation, windowWidth, nWindows):
    '''
    Integrates within a set of windows throughout the frequency range.
    windowSeparation is the distance between windows, in Hz
    windowWidth is the width of the window to integrate, in Hz
    nWindows is number of windows to test
    Returns an ndata*nWindows array
    '''
    __checksize()
    gridSize = (ndata + blockSize - 1) / blockSize

    cuda.memcpy_htod(mod.get_global('numFreqs')[0], np.int32(nfrequencies))
    cuda.memcpy_htod(mod.get_global('numPoints')[0], np.int32(ndata))
    cuda.memcpy_htod(mod.get_global('centerFreq')[0], np.float32(centerFreq))

    peakFreqs = findPeaks()
    lowerBounds = peakFreqs - windowSeparation * (nWindows - 1) / 2 - windowWidth / 2
    upperBounds = lowerBounds + windowWidth

    out = np.empty((ndata, 0), dtype='float32')
    temp = np.empty((ndata, 1), dtype='float32')
    integrate = mod.get_function('integrateBounds')

    for _ in range(nWindows):
        bounds = np.concatenate((lowerBounds, upperBounds))
        integrate(cuda.In(data.astype('float32')), cuda.Out(temp),
                cuda.In(bounds.astype('float32')), cuda.In(frequencies.astype('float32')),
                block=(blockSize,1,1), grid=(gridSize,1,1))

        out = np.concatenate((out, temp), 1)

        lowerBounds += windowSeparation
        upperBounds += windowSeparation

    return out.reshape(datashape + (nWindows,))

def splitIntegralV(vSeparation, vWidth, nWindows):
    '''
    same as splitIntegral, except parameters are in terms of velocities
    for doppler shift (rather than frequencies)
    '''
    return splitIntegral(dopplerShift(vSeparation), dopplerShift(vWidth), nWindows)

def setData(data, dFreqs, center):
    '''
    dat is an ndata*nfrequency array
    '''
    global dat, centerFreq, freqs, datashape
    datashape=data.shape[:-1]
    dat = data.reshape((-1, data.shape[-1])).T
    freqs = dFreqs
    centerFreq = center
    setSkipSize(1)

def setSkipSize(skipSize):
    global data, frequencies, nfrequencies, ndata
    nfreq, ndata = dat.shape
    nfrequencies = (nfreq + skipSize - 1) / skipSize
    data = dat[::skipSize]
    frequencies = freqs[::skipSize]

def __checksize():
    if (nfrequencies != frequencies.size):
        raise Exception('dFreq array must have same size as last dim of dat')
