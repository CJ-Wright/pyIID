__author__ = 'christopher'
from numba import *
import math
import numpy as np


@autojit()
def get_ij_lists(n):
    i_list = []
    j_list = []
    for i in xrange(n):
        for j in xrange(n):
            if j > i:
                i_list.append(i)
                j_list.append(j)
    return np.asarray(i_list, dtype=np.uint32), np.asarray(j_list, dtype=np.uint32)


def symmetric_reshape(out_data, in_data, i_list, j_list):
    out_data[j_list, i_list] = in_data
    out_data[i_list, j_list] = in_data


def antisymmetric_reshape(out_data, in_data, i_list, j_list):
    out_data[j_list, i_list] = in_data
    out_data[i_list, j_list] = -1 * in_data


def asr_gpu(new_grad, grad, il, jl):
    k, qx = cuda.grid(2)

    if k > len(il):
        return
    for tz in range(3):
        new_grad[jl[k], il[k], tz, qx] = grad[k, tz, qx]
        new_grad[il[k], jl[k], tz, qx] = -1*grad[k, tz, qx]


@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :]])
def get_d_array(d, qi, qj):
    k = cuda.grid(1)
    if k >= len(d):
        return
    for tz in range(3):
        d[k, tz] = qi[k, tz] - qj[k, tz]


@cuda.jit(argtypes=[f4[:], f4[:, :]])
def get_r_array(r, d):
    k = cuda.grid(1)

    if k >= len(r):
        return
    r[k] = math.sqrt(d[k, 0] ** 2 + d[k, 1] ** 2 + d[k, 2] ** 2)


@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :]])
def get_normalization_array(norm_array, scati, scatj):
    """
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: NxNx3 array
        Normalization array
    scatter_array: NxM array
        The scatter factor array
    """

    k, qx = cuda.grid(2)

    n = norm_array.shape[0]
    qmax_bin = norm_array.shape[1]
    if k >= n or qx >= qmax_bin:
        return
    norm_array[k, qx] = scati[k, qx] * scatj[k, qx]


@cuda.jit(argtypes=[f4[:, :], f4[:], f4[:, :], f4])
def get_fq(fq, r, norm, qbin):
    k, qx = cuda.grid(2)
    n = fq.shape[0]
    qmax_bin = fq.shape[1]
    if k >= n or qx >= qmax_bin:
        return
    fq[k, qx] = norm[k, qx] * math.sin(qbin * qx * r[k]) / r[k]

@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4[:], f4[:, :], f4[:, :], f4])
def get_grad_fq(grad, fq, r, d, norm, qbin):
    k, qx = cuda.grid(2)

    if k >= len(grad) or qx > norm.shape[1]:
        return
    for w in range(3):
        grad[k, w, qx] = (norm[k, qx] * qx * qbin * math.cos(qx * qbin * r[k]) - fq[k, qx]) / r[k] * d[k, w] / r[k]


@cuda.jit(argtypes=[f4[:], f4[:, :]])
def d2_to_d1_sum(d1, d2):
    qx = cuda.grid(1)

    if qx >= len(d1):
        return
    tmp = d2[:, qx].sum()
    d1[qx] = tmp


def d4_to_d2_sum(d3, d4):
    imax, N, jmax, kmax = d4.shape
    i, j, k = cuda.grid(3)
    if i >= imax or j >= jmax or k >= kmax:
        return
    for l in range(N):
        d3[i, j, k] += d4[i, l, j, k]

@cuda.jit(argtypes=[f4[:,:], f4[:,:], f4[:,:], u4[:], u4[:]])
def construct_qij(qi, qj, q, il, jl):

    k = cuda.grid(1)

    if k >= len(il):
        return
    for tz in range(3):
        qi[k, tz] = q[il[k], tz]
        qj[k, tz] = q[jl[k], tz]

@cuda.jit(argtypes=[f4[:,:], f4[:,:], f4[:,:], u4[:], u4[:]])
def construct_scatij(scati, scatj, scat, il, jl):

    k, qx = cuda.grid(2)

    if k >= len(il) or qx >= scat.shape[1]:
        return
    scati[k, qx] = scat[il[k], qx]
    scatj[k, qx] = scat[jl[k], qx]


@cuda.jit(argtypes=[f4[:,:,:]])
def zero_pseudo_3D(A):
    tx, kq = cuda.grid(2)

    n = A.shape[0]
    qmax_bin = A.shape[2]

    if tx >= n or kq >= qmax_bin:
        return
    for ty in range(3):
        A[tx, ty, kq] = 0.0


@cuda.jit(argtypes=[f4[:,:,:], f4[:,:,:], u4[:], u4[:]])
def flat_sum(new_grad, grad, il, jl):
    # Something goes wrong here, a race condition or something
    # the CPU version of this code works fine.  I may need to move to atomic
    # addition, although it is not currently supported by numba.

    qx = cuda.grid(1)
    # k = cuda.grid(1)

    if qx >= grad.shape[2]:
    # if k >= len(il):
        return

    for k in range(len(il)):
    # for qx in range(grad.shape[2]):
        ik = il[k]
        jk = jl[k]
        for tz in range(3):
            new_grad[ik, tz, qx] -= grad[k, tz, qx]
            new_grad[jk, tz, qx] += grad[k, tz, qx]
