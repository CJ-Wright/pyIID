import math
from numba import *
import mkl
import os

__author__ = 'christopher'

cache = True
if bool(os.getenv('NUMBA_DISABLE_JIT')):
    cache = False
processor_target = 'cpu'


# F(Q) test_kernels ----------------------------------------------------------

@jit(void(f4[:, :, :], f4[:, :]), target=processor_target, nopython=True,
     cache=cache)
def get_d_array(d, q):
    """
    Generate the NxNx3 array which holds the coordinate pair distances

    Parameters
    ----------
    d: NxNx3 array
    q: Nx3 array
        The atomic positions
    """
    n = len(q)
    for i in xrange(i4(n)):
        for j in xrange(i4(n)):
            for w in xrange(i4(3)):
                d[i, j, w] = q[j, w] - q[i, w]


@jit(void(f4[:, :], f4[:, :, :]), target=processor_target, nopython=True,
     cache=cache)
def get_r_array(r, d):
    """
    Generate the Nx3 array which holds the pair distances

    Parameters
    ----------
    r: Nx3 array
    d: NxNx3 array
        The coordinate pair distances
    """
    n = len(r)
    for i in xrange(i4(n)):
        for j in xrange(i4(n)):
            tmp = f4(0.)
            for w in xrange(i4(3)):
                tmp += d[i, j, w] * d[i, j, w]
            r[i, j] = math.sqrt(tmp)


@jit(void(f4[:, :, :], f4[:, :]), target=processor_target, nopython=True,
     cache=cache)
def get_normalization_array(norm_array, scatter_array):
    """
    Generate the sv dependant normalization factors for the F(sv) array

    Parameters
    -----------
    norm_array: NxNxQ array
        Normalization array
    scatter_array: NxQ array
        The scatter factor array
    """
    n, _, qmax_bin = norm_array.shape
    for qx in xrange(i4(qmax_bin)):
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                if i != j:
                    norm_array[i, j, qx] = scatter_array[i, qx] * \
                                           scatter_array[j, qx]


@jit(void(f4[:, :, :], f4[:, :], f4), target=processor_target, nopython=True,
     cache=cache)
def get_omega(omega, r, qbin):
    """
    Generate F(Q), not normalized, via the Debye sum

    Parameters
    ---------
    omega: Nd array
        The reduced scatter pattern
    r: NxN array
        The pair distance array
    qbin: float
        The qbin size
    """
    n, _, qmax_bin = omega.shape
    for qx in xrange(i4(qmax_bin)):
        sv = f4(qx) * qbin
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                if i != j:
                    rij = r[i, j]
                    omega[i, j, qx] = math.sin(sv * rij) / rij


@jit(void(f4[:], f4[:, :, :], f4[:, :, :]), target=processor_target,
     nopython=True, cache=cache)
def get_fq(fq, omega, norm):
    n, _, qmax_bin = omega.shape
    for qx in xrange(i4(qmax_bin)):
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                fq[qx] += norm[i, j, qx] * omega[i, j, qx]


@jit(void(f4[:, :, :], f4[:, :, :], f4[:, :, :], f4[:, :, :]),
     target=processor_target, nopython=True, cache=cache)
def get_adp_fq(fq, omega, tau, norm):
    n, _, qmax_bin = omega.shape
    for qx in xrange(i4(qmax_bin)):
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                fq[i, j, qx] = norm[i, j, qx] * omega[i, j, qx] * tau[i, j, qx]


@jit(void(f4[:, :, :], f4[:, :, :]), target=processor_target, nopython=True,
     cache=cache)
def get_fq_inplace(omega, norm):
    n, _, qmax_bin = omega.shape
    for qx in xrange(i4(qmax_bin)):
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                omega[i, j, qx] *= norm[i, j, qx]


# Gradient test_kernels -------------------------------------------------------


@jit(void(f4[:, :, :, :], f4[:, :, :], f4[:, :], f4[:, :, :], f4),
     target=processor_target, nopython=True, cache=cache)
def get_grad_omega(grad_omega, omega, r, d, qbin):
    n, _, _, qmax_bin = grad_omega.shape
    for qx in xrange(i4(qmax_bin)):
        sv = f4(qx) * qbin
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                if i != j:
                    rij = r[i, j]
                    a = sv * math.cos(sv * rij) - omega[i, j, qx]
                    a /= rij * rij
                    for w in xrange(i4(3)):
                        grad_omega[i, j, w, qx] = a * d[i, j, w]


@jit(void(f4[:, :, :, :], f4[:, :, :, :], f4[:, :, :]),
     target=processor_target, nopython=True, cache=cache)
def get_grad_fq(grad, grad_omega, norm):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ------------
    grad: NxNx3xQ numpy array
        The array which will store the FQ gradient
    grad_omega: NxNx3xQ numpy array
        The gradient of omega
    norm: NxNxQ array
        The normalization array
    """
    n, _, _, qmax_bin = grad.shape
    for i in xrange(i4(n)):
        for w in xrange(i4(3)):
            for j in xrange(i4(n)):
                if i != j:
                    for qx in xrange(grad.shape[3]):
                        grad[i, j, w, qx] = norm[i, j, qx] * grad_omega[
                            i, j, w, qx]


@jit(void(f4[:, :, :, :], f4[:, :, :]), target=processor_target, nopython=True,
     cache=cache)
def get_grad_fq_inplace(grad_omega, norm):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ------------
    grad_omega: NxNx3xQ numpy array
        The gradient of omega
    norm: NxNxQ array
        The normalization array
    """
    n, _, _, qmax_bin = grad_omega.shape
    for i in xrange(i4(n)):
        for w in xrange(i4(3)):
            for j in xrange(i4(n)):
                if i != j:
                    for qx in xrange(i4(qmax_bin)):
                        grad_omega[i, j, w, qx] *= norm[i, j, qx]

@jit(void(f4[:, :, :, :, :], f4[:, :], f4), target=processor_target,
     nopython=True, cache=cache)
def get_voxel_displacements(d, q, resolution):
    im, jm, km, n, _ = d.shape
    for i in xrange(im):
        x = (i + .5) * resolution
        for j in xrange(jm):
            y = (j + .5) * resolution
            for k in xrange(km):
                z = (k + .5) * resolution
                for l in xrange(n):
                    d[i, j, k, l, 0] = x - q[l, 0]
                    d[i, j, k, l, 1] = y - q[l, 1]
                    d[i, j, k, l, 2] = z - q[l, 2]


@jit(void(f4[:, :, :, :, :], f4[:, :, :, :], f4), target=processor_target,
     nopython=True, cache=cache)
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
    im, jm, km, n, qmax_bin = omega.shape
    for i in xrange(im):
        for j in xrange(jm):
            for k in xrange(km):
                for qx in xrange(i4(qmax_bin)):
                    sv = f4(qx) * qbin
                    for l in xrange(n):
                        rij = r[i, j, k, l]
                        if rij != 0.0:
                            omega[i, j, k, l, qx] = math.sin(sv * rij) / rij

@jit(void(f4[:, :, :, :], f4[:, :, :, :, :], f4[:, :]), target=processor_target, nopython=True,
     cache=cache)
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
    im, jm, km, n, qmax_bin = omega.shape
    for i in xrange(im):
        for j in xrange(jm):
            for k in xrange(km):
                for qx in xrange(i4(qmax_bin)):
                    for l in xrange(n):
                        fq[i, j, k, qx] += omega[i, j, k, l, qx] * norm[l, qx]