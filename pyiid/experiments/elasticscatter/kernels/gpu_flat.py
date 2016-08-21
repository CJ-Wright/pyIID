import math

from numba import *
from numba import cuda, f4, i4

from pyiid.experiments.elasticscatter.kernels import cuda_k_to_ij, cuda_ij_to_k
from builtins import range
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


@cuda.jit(argtypes=[f4[:], f4[:, :], f4[:], f4[:, :], i4])
def get_sigma_from_adp(sigma, adps, r, d, offset):
    """
    Get the thermal broadening standard deviation from atomic displacement
    parameters

    Parameters
    ----------
    sigma: k array
    adps: nx3 array
        The atomic displacment parameters
    r: k array
        The interatomic distances
    d:
        The interatomic displacements
    offset:
        Atomic pair offset for parallelization
    """
    k = cuda.grid(1)
    if k >= len(sigma):
        return
    i, j = cuda_k_to_ij(i4(k + offset))
    tmp = f4(0.)
    for w in range(i4(3)):
        tmp += (adps[i, w] - adps[j, w]) * d[k, w] / r[k]
    sigma[k] = tmp


@cuda.jit(argtypes=[f4[:, :], f4[:], f4])
def get_tau(dw_factor, sigma, qbin):
    """
    Get the Debye-Waller factor from the thermal broadening

    Parameters
    -----------
    dw_factor: kxQ array
    sigma: k array
        Thermal broadening standard deviation
    qbin: float
        The sv resolution of the experiment
    """
    kmax, qmax_bin = dw_factor.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    sv = f4(qx) * qbin
    dw_factor[k, qx] = math.exp(f4(-.5) * sigma[k] * sigma[k] * sv * sv)

@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :]])
def get_fq(fq, omega, norm):
    """
    Get the reduced structure factor F(Q) via the Debye Sum

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


@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :], f4[:, :]])
def get_adp_fq(fq, omega, tau, norm):
    """
    Get the reduced structure factor F(sv) via the Debye Sum

    Parameters
    ----------
    fq: kxQ array
    omega: kxQ array
        Debye sum
    tau: kxQ array
        Debye-Waller factor
    norm: kxQ
        Outer Product of the scatter factors
    """
    kmax, qmax_bin = omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    fq[k, qx] = norm[k, qx] * omega[k, qx] * tau[k, qx]

# Gradient kernels -------------------------------------------------------
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
    for w in range(i4(3)):
        grad_omega[k, w, qx] = a * d[k, w]


@cuda.jit(
    argtypes=[f4[:, :, :], f4[:, :], f4[:], f4[:, :], f4[:], f4[:, :], f4, i4])
def get_grad_tau(grad_tau, tau, r, d, sigma, adps, qbin, offset):
    """
    Get the gradient of the Debye-Waller factors with respect to atomic position
    grad_tau: kx3xQ array
     The gradient
    tau: kxQ array
        The Debye-Waller factor
    r:k array
        The interatomic distances
    d: kx3 array
        The interatomic displacements
    sigma: k array
        Thermal broadenings
    adps: Nx3 array
        The atomic displacement parameters
    qbin: float
        The experimental resolution
    offset: int
        The parallelization offset
    """
    kmax, _, qmax_bin = grad_tau.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    sv = f4(qx) * qbin
    i, j = cuda_k_to_ij(i4(k + offset))
    rk = r[k]
    tmp = sigma[k] * sv * sv * tau[k, qx] / (rk * rk * rk)
    for w in range(i4(3)):
        grad_tau[k, w, qx] = tmp * (
            d[k, w] * sigma[k] - (adps[i, w] - adps[j, w]) * (rk * rk))

@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], f4[:, :]])
def get_grad_fq(grad, grad_omega, norm):
    """
    Generate the gradient F(Q) for an atomic configuration

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
    for w in range(i4(3)):
        grad[k, w, qx] = a * grad_omega[k, w, qx]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4[:, :], f4[:, :, :], f4[:, :, :],
                    f4[:, :]])
def get_adp_grad_fq(grad, omega, tau, grad_omega, grad_tau, norm):
    """
    Generate the gradient F(sv) for an atomic configuration
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
        The size of the sv bins
    """
    kmax, _, qmax_bin = grad.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    a = norm[k, qx]
    b = tau[k, qx]
    c = omega[k, qx]
    for w in range(i4(3)):
        grad[k, w, qx] = a * (b * grad_omega[k, w, qx] +
                              c * grad_tau[k, w, qx])

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
        for tz in range(i4(3)):
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


@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :]])
def get_adp_fq_inplace(norm, omega, tau):
    kmax, qmax_bin = omega.shape
    k, qx = cuda.grid(2)
    if k >= kmax or qx >= qmax_bin:
        return
    norm[k, qx] *= omega[k, qx] * tau[k, qx]

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
    Generate the gradient F(Q) for an atomic configuration

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
    for w in range(i4(3)):
        grad_omega[k, w, qx] *= a


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4[:], f4[:], f4[:, :], f4])
def get_dtau_dadp(dtau_dadp, tau, sigma, r, d, qbin):
    kmax, _, qmax_bin = dtau_dadp.shape
    k, qx = cuda.grid(2)
    sv = qx * qbin
    tmp = f4(-1.) * sigma[k] * sv * sv * tau[k, qx] / r[k]
    for w in range(i4(3)):
        dtau_dadp[k, w, qx] = tmp * d[k, w]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4[:, :]])
def get_dfq_dadp_inplace(dtau_dadp, omega, norm):
    kmax, _, qmax_bin = dtau_dadp.shape
    k, qx = cuda.grid(2)
    for w in range(i4(3)):
        dtau_dadp[k, w, qx] *= norm[k, qx] * omega[k, qx]