import math
from numba import *
from numba import cuda, f4, i4, int32
import numpy as np

__author__ = 'christopher'


@jit(target='cpu', nopython=True)
def ij_to_k(i, j):
    return int(j + i * (i - 1) / 2)


@jit(target='cpu', nopython=True)
def k_to_ij(k):
    i = math.floor(float((1 + math.sqrt(1 + 8. * k))) / 2.)
    j = k - i * (i - 1) / 2.
    return i4(i), i4(j)


def symmetric_reshape(in_data):
    im, jm = k_to_ij(in_data.shape[0])
    out_data = np.zeros((im, im) + in_data.shape[1:])
    for k in xrange(in_data.shape[0]):
        i, j = k_to_ij(k)
        out_data[i, j] = in_data[k]
        out_data[j, i] = in_data[k]
    return out_data


def antisymmetric_reshape(in_data):
    im, jm = k_to_ij(in_data.shape[0])
    out_data = np.zeros((im, im) + in_data.shape[1:])
    for k in xrange(in_data.shape[0]):
        i, j = k_to_ij(k)
        out_data[i, j] = -1 * in_data[k]
        out_data[j, i] = in_data[k]
    return out_data


@cuda.jit(device=True)
def cuda_k_to_ij(k):
    i = math.floor((f4(1) + f4(math.sqrt(f4(1) + f4(8.) * f4(k)))) * f4(.5))
    j = f4(k) - f4(i) * (f4(i) - f4(1)) * f4(.5)
    return i4(i), i4(j)


@cuda.jit(device=True)
def cuda_ij_to_k(i, j):
    return int32(j + i * (i - 1) / 2)


@jit(target='cpu', nopython=True)
def index1d_to_3d(i, v):
    xm, ym, zm = v
    return i / ym / zm, (i / zm) % ym, i % zm