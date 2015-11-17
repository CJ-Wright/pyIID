import math
from numba import *
from numba import cuda, f4, i4
from pyiid.experiments.elasticscatter.kernels import (cuda_k_to_ij,
                                                      cuda_ij_to_k,
                                                      cuda_index1d_to_3d)

__author__ = 'christopher'


# F(Q) kernels ---------------------------------------------------------------
@cuda.jit(argtypes=[f4[:, :], f4[:, :], i4])
def get_d_array(d, q, offset):
    """
    Generate the kx3 array which holds the pair displacements

    Parameters
    ----------
    d: NxNx3 array
        The displacement array
    q: Nx3 array
        The atomic positions
    offset: int
        The amount of previously covered pairs
    """
    k = cuda.grid(1)
    if k >= len(d):
        return
    i, j = cuda_k_to_ij(i4(k + offset))
    for w in range(3):
        d[k, w] = q[i, w] - q[j, w]


@cuda.jit(argtypes=[f4[:], f4[:, :]])
def get_r_array(r, d):
    """
    Generate the k array which holds the pair distances

    Parameters
    ----------
    r: k array
        The pair distances
    d: kx3 array
        The pair displacements
    """
    k = cuda.grid(1)
    if k >= len(r):
        return
    a, b, c = d[k, :]
    r[k] = math.sqrt(a * a + b * b + c * c)


@cuda.jit(argtypes=[f4[:, :], f4[:, :], i4])
def get_normalization_array(norm_array, scat, offset):
    """
    Generate the sv dependant normalization factors for the F(sv) array

    Parameters
    -----------
    norm_array: kxQ array
        Normalization array
    scat: NxQ array
        The scatter factor array
    offset: int
        The amount of previously covered pairs
    """
    k, qx = cuda.grid(2)
    if k >= norm_array.shape[0] or qx >= norm_array.shape[1]:
        return
    i, j = cuda_k_to_ij(i4(k + offset))
    norm_array[k, qx] = scat[i, qx] * scat[j, qx]


@cuda.jit(argtypes=[f4[:, :], f4[:], f4])
def get_omega(omega, r, qbin):
    """
    Generate Omega

    Parameters
    ---------
    omega: kxQ array
    r: k array
        The pair distance array
    qbin: float
        The qbin size
    """
    kmax, qmax_bin = omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    sv = qbin * f4(qx)
    rk = r[k]
    omega[k, qx] = math.sin(sv * rk) / rk


@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :]])
def get_fq(fq, omega, norm):
    """
    Get the reduced structure factor F(sv) via the Debye Sum

    Parameters
    ----------
    fq: kxQ array
    omega: kxQ array
        Debye sum
    norm: kxQ
        Outer Product of the scatter factors
    """
    kmax, qmax_bin = omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    fq[k, qx] = norm[k, qx] * omega[k, qx]


# Gradient test_kernels -------------------------------------------------------
@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4[:], f4[:, :], f4])
def get_grad_omega(grad_omega, omega, r, d, qbin):
    """
    Get the gradient of the Debye sum with respect to atomic positions

    Parameters
    ----------
    grad_omega: kx3xQ array
        The gradient
    omega: kxQ array
        Debye sum
    r: k array
        The pair distance array
    d: kx3 array
        The pair displacements
    qbin: float
        The qbin size
    """
    kmax, _, qmax_bin = grad_omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    sv = f4(qx) * qbin
    rk = r[k]
    a = (sv * math.cos(sv * rk)) - omega[k, qx]
    a /= rk * rk
    for w in xrange(i4(3)):
        grad_omega[k, w, qx] = a * d[k, w]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], f4[:, :]])
def get_grad_fq(grad, grad_omega, norm):
    """
    Generate the gradient F(sv) for an atomic configuration

    Parameters
    ------------
    grad: kx3xQ numpy array
        The array which will store the FQ gradient
    grad_omega: kx3xQ array
        The gradient of the Debye sum
    norm: kxQ
        Outer Product of the scatter factors
    """
    kmax, _, qmax_bin = grad.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    a = norm[k, qx]
    for w in xrange(i4(3)):
        grad[k, w, qx] = a * grad_omega[k, w, qx]


@cuda.jit(argtypes=[f4[:, :, :]])
def zero3d(a):
    """
    Zero out a 3D array on the GPU
    
    Parameters
    ----------
    a: Mx3xQ array
    """
    i, qx = cuda.grid(2)
    if i >= a.shape[0] or qx >= a.shape[2]:
        return
    for tz in range(3):
        a[i, tz, qx] = float32(0.)


@cuda.jit(argtypes=[f4[:], f4[:, :]])
def d2_to_d1_sum(d1, d2):
    qx = cuda.grid(1)

    if qx >= len(d1):
        return
    tmp = d2[:, qx].sum()
    d1[qx] = tmp


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], i4])
def fast_fast_flat_sum(new_grad, grad, k_cov):
    i, j, qx = cuda.grid(3)
    n = len(new_grad)
    if i >= n or j >= n or qx >= grad.shape[2] or i == j:
        return
    if j < i:
        k = cuda_ij_to_k(i, j)
        alpha = float32(-1)
    else:
        k = cuda_ij_to_k(j, i)
        alpha = float32(1)
    k -= k_cov
    if 0 <= k < len(grad):
        for tz in xrange(i4(3)):
            cuda.atomic.add(new_grad, (i, tz, qx), grad[k, tz, qx] * alpha)


@cuda.jit(argtypes=[f4[:, :], f4[:, :], i4])
def experimental_sum_fq(g_odata, g_idata, n):
    _, qx = cuda.grid(2)
    sdata = cuda.shared.array(512, f4)

    tid = cuda.threadIdx.x
    bd = cuda.blockDim.x
    bid = cuda.blockIdx.x
    i = bid * bd * 2 + tid
    gridsize = bd * 2 * cuda.gridDim.x

    sdata[tid] = 0.
    while i < n:
        if i + bd >= len(g_idata):
            sdata[tid] += g_idata[i, qx]
        else:
            sdata[tid] += g_idata[i, qx] + g_idata[i + bd, qx]
        i += gridsize
    cuda.syncthreads()

    if bd >= 512:
        if tid < 256:
            sdata[tid] += sdata[tid + 256]
        cuda.syncthreads()

    if bd >= 256:
        if tid < 128:
            sdata[tid] += sdata[tid + 128]
        cuda.syncthreads()

    if bd >= 128:
        if tid < 64:
            sdata[tid] += sdata[tid + 64]
        cuda.syncthreads()

    if tid < 32:
        if bd >= 64:
            sdata[tid] += sdata[tid + 32]
        if bd >= 32:
            sdata[tid] += sdata[tid + 16]
        if bd >= 16:
            sdata[tid] += sdata[tid + 8]
        if bd >= 8:
            sdata[tid] += sdata[tid + 4]
        if bd >= 4:
            sdata[tid] += sdata[tid + 2]
        if bd >= 2:
            sdata[tid] += sdata[tid + 1]

    if tid == 0:
        g_odata[cuda.blockIdx.x, qx] = sdata[0]


@cuda.jit(argtypes=[f4[:], f4[:, :]])
def d2_to_d1_cleanup_kernel(out_data, in_data):
    i = cuda.grid(1)
    if i >= len(out_data):
        return
    out_data[i] = in_data[0, i]


@cuda.jit(argtypes=[f4[:, :]])
def d2_zero(a):
    i, j = cuda.grid(2)
    if i >= a.shape[0] or j >= a.shape[1]:
        return
    a[i, j] = f4(0.)


@cuda.jit(argtypes=[f4[:, :], f4[:, :]])
def get_fq_inplace(norm, omega):
    k, qx = cuda.grid(2)
    if k >= norm.shape[0] or qx >= norm.shape[1]:
        return
    norm[k, qx] *= omega[k, qx]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], i4])
def experimental_sum_grad_fq1(new_grad, grad, k_cov):
    k, qx = cuda.grid(2)
    if k >= len(grad) or qx >= grad.shape[2]:
        return
    i, j = cuda_k_to_ij(i4(k + k_cov))
    for tz in range(3):
        a = grad[k, tz, qx]
        cuda.atomic.add(new_grad, (j, tz, qx), a)
        cuda.atomic.add(new_grad, (i, tz, qx), f4(-1.) * a)


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :]])
def get_grad_fq_inplace(grad_omega, norm):
    """
    Generate the gradient F(sv) for an atomic configuration

    Parameters
    ------------
    grad_omega: Kx3xQ numpy array
        The array which will store the FQ gradient
    norm: kxQ array
        The normalization array
    """
    kmax, _, qmax_bin = grad_omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    a = norm[k, qx]
    for w in xrange(i4(3)):
        grad_omega[k, w, qx] *= a


@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:], i4[:]])
def get_voxel_distances(r, q, resolution, v):
    om, n = r.shape
    o, l = cuda.grid(2)
    if o >= om or l >= n:
        return
    i, j, k = cuda_index1d_to_3d(o, v)
    x = (i + .5) * resolution[0]
    y = (j + .5) * resolution[1]
    z = (k + .5) * resolution[2]
    a = x - q[l, 0]
    b = y - q[l, 1]
    c = z - q[l, 2]
    r[o, l] = math.sqrt(a * a + b * b + c * c)


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4])
def get_voxel_omega(omega, r, qbin):
    """
    Generate F(sv), not normalized, via the Debye sum

    Parameters:
    ---------
    fq: Nd array
        The reduced scatter pattern
    r: NxN array
        The pair distance array
    scatter_array: NxM array
        The scatter factor array
    qbin: float
        The qbin size
    """
    om, n, qmax_bin = omega.shape
    o, l, qx = cuda.grid(3)
    if o >= om or l >= n or qx >= qmax_bin:
        return
    sv = f4(qx) * qbin
    rij = r[o, l]
    if rij != 0.0:
        omega[o, l, qx] = math.sin(sv * rij) / rij


@cuda.jit(argtypes=[f4[:, :], f4[:, :, :], f4[:, :]])
def get_voxel_fq(fq, omega, norm):
    """
    Generate F(sv), not normalized, via the Debye sum

    Parameters:
    ---------
    fq: Nd array
        The reduced scatter pattern
    r: NxN array
        The pair distance array
    scatter_array: NxM array
        The scatter factor array
    qbin: float
        The qbin size
    """
    om, n, qmax_bin = omega.shape
    o, l, qx = cuda.grid(3)
    if o >= om or l >= n or qx >= qmax_bin:
        return
    fq[o, qx] += omega[o, l, qx] * norm[l, qx]