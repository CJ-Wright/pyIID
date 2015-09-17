import math

from numba import *
from numba import cuda, f4, i4

from pyiid.experiments.elasticscatter.kernels import cuda_k_to_ij, cuda_ij_to_k

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
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: kxQ array
        Normalization array
    scatter_array: NxQ array
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

    Parameters:
    ---------
    omega: kxQ array
    r: k array
        The pair distance array
    scatter_array: kxQ array
        The scatter factor array
    qbin: float
        The qbin size
    """
    kmax, qmax_bin = omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    Q = f4(qbin) * f4(qx)
    rk = r[k]
    omega[k, qx] = math.sin(Q * rk) / rk


@cuda.jit(argtypes=[f4[:], f4[:, :], f4[:], f4[:, :], i4])
def get_sigma_from_adp(sigma, adps, r, d, offset):
    """
    Get the thermal broadening standard deviation from atomic displacement
    parameters

    Parameters
    ----------
    :param sigma: k array
    :param adps: nx3 array
        The atomic displacment parameters
    :param r: k array
        The interatomic distances
    :param d:
        The interatomic displacements
    :param offset:
        Atomic pair offset for parallelization
    """
    k = cuda.grid(1)
    if k >= len(sigma):
        return
    i, j = cuda_k_to_ij(i4(k + offset))
    tmp = 0.
    for w in xrange(3):
        tmp += (adps[i, w] - adps[j, w]) * d[k, w] / r[k]
    sigma[k] = tmp


@cuda.jit(argtypes=[f4[:, :], f4[:], f4])
def get_tau(dw_factor, sigma, qbin):
    kmax, qmax_bin = dw_factor.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    Q = f4(qx) * f4(qbin)
    dw_factor[k, qx] = math.exp(f4(-.5) * f4(sigma[k] * sigma[k]) * f4(Q * Q))


@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :]])
def get_fq(fq, omega, norm):
    kmax, qmax_bin = omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    fq[k, qx] = norm[k, qx] * omega[k, qx]


@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :], f4[:, :]])
def get_adp_fq(fq, omega, tau, norm):
    kmax, qmax_bin = omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    fq[k, qx] = norm[k, qx] * omega[k, qx] * tau[k, qx]


# Gradient test_kernels -------------------------------------------------------
@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4[:], f4[:, :], f4])
def get_grad_omega(grad_omega, omega, r, d, qbin):
    kmax, _, qmax_bin = grad_omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    Q = f4(qx) * f4(qbin)
    rk = r[k]
    a = (Q * math.cos(Q * rk)) - omega[k, qx]
    a /= f4(rk * rk)
    for w in xrange(3):
        grad_omega[k, w, qx] = a * d[k, w]


@cuda.jit(
    argtypes=[f4[:, :, :], f4[:, :], f4[:], f4[:, :], f4[:], f4[:, :], f4, i4])
def get_grad_tau(grad_tau, tau, r, d, sigma, adps, qbin, offset):
    kmax, _, qmax_bin = grad_tau.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    Q = f4(qx) * f4(qbin)
    i, j = cuda_k_to_ij(i4(k + offset))
    rk = r[k]
    tmp = f4(sigma[k] * f4(Q * Q) * tau[k, qx]) / f4(rk * rk * rk)
    for w in xrange(3):
        grad_tau[k, w, qx] = f4(tmp) * f4(
            d[k, w] * sigma[k] - f4(adps[i, w] - adps[j, w]) * f4(rk * rk))


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], f4[:, :]])
def get_grad_fq(grad, grad_omega, norm):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ------------
    grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    d: NxNx3 array
        The distance array for the configuration
    r: NxN array
        The inter-atomic distances
    scatter_array: NxQ array
        The scatter factor array
    qbin: float
        The size of the Q bins
    """
    kmax, _, qmax_bin = grad.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    a = norm[k, qx]
    for w in xrange(3):
        grad[k, w, qx] = a * grad_omega[k, w, qx]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4[:, :], f4[:, :, :], f4[:, :, :],
                    f4[:, :]])
def get_adp_grad_fq(grad, omega, tau, grad_omega, grad_tau, norm):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ------------
    grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    d: NxNx3 array
        The distance array for the configuration
    r: NxN array
        The inter-atomic distances
    scatter_array: NxQ array
        The scatter factor array
    qbin: float
        The size of the Q bins
    """
    kmax, _, qmax_bin = grad.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    a = norm[k, qx]
    b = tau[k, qx]
    c = omega[k, qx]
    for w in xrange(3):
        grad[k, w, qx] = a * (b * grad_omega[k, w, qx] +
                              c * grad_tau[k, w, qx])


@cuda.jit(argtypes=[f4[:, :, :]])
def zero3d(A):
    i, qx = cuda.grid(2)
    if i >= A.shape[0] or qx >= A.shape[2]:
        return
    for tz in range(3):
        A[i, tz, qx] = float32(0.)


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
        for tz in xrange(3):
            cuda.atomic.add(new_grad, (i, tz, qx), grad[k, tz, qx] * alpha)


@cuda.jit(argtypes=[f4[:, :], f4[:, :], i4])
def experimental_sum_fq(g_odata, g_idata, n):
    _, qx = cuda.grid(2)
    sdata = cuda.shared.array(512, f4)

    tid = cuda.threadIdx.x
    bd = cuda.blockDim.x
    bid = cuda.blockIdx.x
    i = bid * bd * 2 + tid
    gridSize = bd * 2 * cuda.gridDim.x

    sdata[tid] = 0.
    while i < n:
        if i + bd >= len(g_idata):
            sdata[tid] += g_idata[i, qx]
        else:
            sdata[tid] += g_idata[i, qx] + g_idata[i + bd, qx]
        i += gridSize
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
def d2_zero(A):
    i, j = cuda.grid(2)
    if i >= A.shape[0] or j >= A.shape[1]:
        return
    A[i, j] = f4(0.)


@cuda.jit(argtypes=[f4[:, :], f4[:, :]])
def get_fq_inplace(norm, omega):
    k, qx = cuda.grid(2)
    if k >= norm.shape[0] or qx >= norm.shape[1]:
        return
    norm[k, qx] *= omega[k, qx]


@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :]])
def get_adp_fq_inplace(norm, omega, tau):
    kmax, qmax_bin = omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    norm[k, qx] *= omega[k, qx] * tau[k, qx]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], i4],
          debug=True
          )
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
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ------------
    grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    d: NxNx3 array
        The distance array for the configuration
    r: NxN array
        The inter-atomic distances
    scatter_array: NxQ array
        The scatter factor array
    qbin: float
        The size of the Q bins
    """
    kmax, _, qmax_bin = grad_omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    a = norm[k, qx]
    for w in xrange(3):
        grad_omega[k, w, qx] *= a
