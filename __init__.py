import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import os
from pycuda.compiler import SourceModule

KERNELFILE = open(os.path.dirname(os.path.abspath(__file__)) + '/sa.cu')
CUDACODE = KERNELFILE.read()
KERNELFILE.close()

CC = 3e8

BLOCKSIZE = 256


class Analyzer:
    dat = None  # raw data
    ndata = 0  # number of spectral lines
    data = None  # data, after frequency skips have been applied
    freqs = None  # all frequencies
    frequencies = None  # frequencies, after skips have been applied
    nfrequencies = 0
    center_freq = 0
    datashape = None

    def __init__(self, data, dfreqs, center):
        self.set_data(data, dfreqs, center)
        self.mod = SourceModule(CUDACODE)

    def dopp_shift(self, velocity):
        '''
        converts a velocity to a doppler shift frequency
        velocity is in km/sec
        '''
        return self.center_freq * velocity * 1e3 / CC

    def vel_shift(self, dfreqs):
        '''
        converts frequency shifts to relative velocities
        '''
        return dfreqs / self.center_freq / 1e3 * CC

    def gauss_reg(self, cutoff=0.25):
        '''
        Does a gaussian regression on the contiguous subset of data
        containing ymax where y>ymax*cutoff.
        Returns an array of height, center, standard deviation.
        '''
        self._checksize()
        out = np.empty((self.ndata, 3), 'float32')

        grid_size = (self.ndata + BLOCKSIZE - 1) / BLOCKSIZE
        greg = self.mod.get_function('reg')

        self._copy_size_to_gpu()

        greg(cuda.In(self.data), cuda.Out(out), cuda.In(self.frequencies),
             np.float32(cutoff), np.int8(False),
             block=(BLOCKSIZE, 1, 1), grid=(grid_size, 1, 1))

        return out.reshape(self.datashape + (3,))

    def quad_reg(self, cutoff=0.25):
        '''
        Does a quadratic regression on the contiguous subset of data
        containing ymax where y>ymax*cutoff.
        Returns a, b, c, where y=a(x-b)^2+c
        '''
        self._checksize()
        out = np.empty((self.ndata, 3), 'float32')

        grid_size = (self.ndata + BLOCKSIZE - 1) / BLOCKSIZE
        greg = self.mod.get_function('reg')

        self._copy_size_to_gpu()

        greg(cuda.In(self.data), cuda.Out(out), cuda.In(self.frequencies),
             np.float32(cutoff), np.int8(True),
             block=(BLOCKSIZE, 1, 1), grid=(grid_size, 1, 1))

        return out.reshape(self.datashape + (3,))

    def gauss_regc(self, cutoff=0.25):
        '''
        Returns the centers predicted by a gaussian regression.
        '''
        return self.gauss_reg(cutoff)[..., 1]

    def quad_regc(self, cutoff=0.25):
        '''
        Returns the centers predicted by a quadratic regression.
        '''
        return self.quad_reg(cutoff)[..., 1]

    def find_peaks(self):
        '''
        Returns an array with size of dataset, with locations of frequency peaks.
        '''
        self._checksize()
        out = np.empty(self.ndata, 'float32')

        grid_size = (self.ndata + BLOCKSIZE - 1) / BLOCKSIZE

        self._copy_size_to_gpu()

        findpeaks = self.mod.get_function('findPeaks')

        findpeaks(cuda.In(self.data), cuda.Out(out), cuda.In(self.frequencies),
                  block=(BLOCKSIZE, 1, 1), grid=(grid_size, 1, 1))

        return out.reshape(self.datashape)

    def num_peaks(self):
        '''
        Returns an array with size of dataset, with numbers of peaks
        '''
        self._checksize()
        out = np.empty(self.ndata, 'int32')

        grid_size = (self.ndata + BLOCKSIZE - 1) / BLOCKSIZE

        self._copy_size_to_gpu()

        countpeaks = self.mod.get_function('countPeaks')

        countpeaks(cuda.In(self.data), cuda.Out(out), cuda.In(self.frequencies),
                   block=(BLOCKSIZE, 1, 1), grid=(grid_size, 1, 1))

        return out.reshape(self.datashape)

    def fwhm(self):
        '''
        Returns an array with size of dataset, with width of each peak
        '''
        self._checksize()
        out = np.empty(self.ndata, 'float32')

        grid_size = (self.ndata + BLOCKSIZE - 1) / BLOCKSIZE

        self._copy_size_to_gpu()

        calcfwhm = self.mod.get_function('fwhm')

        calcfwhm(cuda.In(self.data), cuda.Out(out), cuda.In(self.frequencies),
                 block=(BLOCKSIZE, 1, 1), grid=(grid_size, 1, 1))

        return out.reshape(self.datashape)

    def split_integral(self, window_separation, window_width, nwindows):
        '''
        Integrates within a set of windows throughout the frequency range.
        window_separation is the distance between windows, in Hz
        window_width is the width of the window to integrate, in Hz
        nwindows is number of windows to test
        Returns an ndata*nwindows array
        '''
        self._checksize()
        grid_size = (self.ndata + BLOCKSIZE - 1) / BLOCKSIZE

        self._copy_size_to_gpu()

        peak_freqs = self.find_peaks()
        lower_bounds = peak_freqs - window_separation * (nwindows - 1) / 2 - window_width / 2
        upper_bounds = lower_bounds + window_width

        out = np.empty((self.ndata, 0), dtype='float32')
        temp = np.empty((self.ndata, 1), dtype='float32')
        integrate = self.mod.get_function('integrateBounds')

        for _ in range(nwindows):
            bounds = np.concatenate((lower_bounds, upper_bounds))
            integrate(cuda.In(self.data), cuda.Out(temp),
                      cuda.In(bounds), cuda.In(self.frequencies),
                      block=(BLOCKSIZE, 1, 1), grid=(grid_size, 1, 1))

            out = np.concatenate((out, temp), 1)

            lower_bounds += window_separation
            upper_bounds += window_separation

        return out.reshape(self.datashape + (nwindows,))

    def split_integral_vel(self, vel_separation, vel_width, nwindows):
        '''
        same as split_integral, except parameters are in terms of velocities
        for doppler shift (rather than frequencies)
        '''
        return self.split_integral(self.dopp_shift(vel_separation), self.dopp_shift(vel_width), nwindows)

    def lr_integral(self, left, limit=.1):
        '''
        Performs an integral from the peak to the left/rightmost bound of the gaussian
        or to the first instance of a 0/negative number, whichever comes first.
        If left is False, does the integral on the right;
        if True does the integral on the left.
        '''
        self._checksize()
        grid_size = (self.ndata + BLOCKSIZE - 1) / BLOCKSIZE

        out = np.empty((self.ndata), dtype='float32')
        integrate = self.mod.get_function('integrateLR')

#     peak_freqs = self.find_peaks()
        peak_freqs = self.quad_reg()

        integrate(cuda.In(self.data), cuda.Out(out),
                  cuda.In(np.ascontiguousarray(peak_freqs)), cuda.In(self.frequencies),
                  np.float32(limit), np.int8(left),
                  block=(BLOCKSIZE, 1, 1), grid=(grid_size, 1, 1))

        return out.reshape(self.datashape)

    def set_data(self, data, dfreqs, center):
        '''
        dat is an ndata*nfrequency array
        '''
        self.datashape = data.shape[:-1]
        self.dat = data.reshape((-1, data.shape[-1])).astype('float32')
        self.freqs = dfreqs.astype('float32')
        self.center_freq = center
        self.skip_size(1)

    def skip_size(self, skipsize):
        '''
        Only takes every n data points.
        For testing purposes, do not use on actual data.
        '''
        self.ndata, nfreq = self.dat.shape
        self.nfrequencies = (nfreq + skipsize - 1) / skipsize
        self.data = np.ascontiguousarray(self.dat[:, ::skipsize])
        self.frequencies = np.ascontiguousarray(self.freqs[::skipsize])

    def _copy_size_to_gpu(self):
        cuda.memcpy_htod(self.mod.get_global('numFreqs')[0], np.int32(self.nfrequencies))
        cuda.memcpy_htod(self.mod.get_global('numPoints')[0], np.int32(self.ndata))
        cuda.memcpy_htod(self.mod.get_global('centerFreq')[0], np.float32(self.center_freq))

    def _checksize(self):
        if (self.nfrequencies != self.frequencies.size):
            raise Exception('dFreq array must have same size as last dim of dat')
