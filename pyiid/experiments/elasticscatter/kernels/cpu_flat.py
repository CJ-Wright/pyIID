from pyiid.experiments.elasticscatter.kernels import *
import math
import os

__author__ = 'christopher'
cache = True
if bool(os.getenv('NUMBA_DISABLE_JIT')):
    cache = False
processor_target = 'cpu'


# F(sv) kernels ---------------------------------------------------------------
@jit(void(f4[:, :], f4[:, :], i4), target=processor_target, nopython=True,
     cache=cache)
def get_d_array(d, q, offset):
    """
    Generate the kx3 array which holds the pair displacements

    Parameters
    ----------
    d: kx3 array
        The displacement array
    q: Nx3 array
        The atomic positions
    offset: int
        The amount of previously covered pairs
    """
    for k in xrange(i4(len(d))):
        i, j = k_to_ij(i4(k + offset))
        for tz in xrange(i4(3)):
            d[k, tz] = q[i, tz] - q[j, tz]


@jit(void(f4[:], f4[:, :]), target=processor_target, nopython=True,
     cache=cache)
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
    for k in xrange(i4(len(r))):
        tmp = f4(0.)
        for w in xrange(i4(3)):
            tmp += d[k, w] * d[k, w]
        r[k] = math.sqrt(tmp)


@jit(void(f4[:, :], f4[:, :], i4), target=processor_target, nopython=True,
     cache=cache)
def get_normalization_array(norm, scat, offset):
    """
    Generate the sv dependant normalization factors for the F(sv) array

    Parameters
    -----------
    norm: kxQ array
        Normalization array
    scat: NxQ array
        The scatter factor array
    offset: int
        The amount of previously covered pairs
    """
    kmax, qmax_bin = norm.shape
    for k in xrange(i4(kmax)):
        i, j = k_to_ij(i4(k + offset))
        for qx in xrange(i4(qmax_bin)):
            norm[k, qx] = scat[i, qx] * scat[j, qx]


@jit(void(f4[:, :], f4[:], f4), target=processor_target, nopython=True,
     cache=cache)
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
    for qx in xrange(i4(qmax_bin)):
        sv = qbin * f4(qx)
        for k in xrange(i4(kmax)):
            rk = r[k]
            omega[k, qx] = math.sin(sv * rk) / rk


@jit(void(f4[:, :], f4[:, :], f4[:, :]), target=processor_target,
     nopython=True, cache=cache)
def get_fq(fq, omega, norm):
    kmax, qmax_bin = omega.shape
    for qx in xrange(i4(qmax_bin)):
        for k in xrange(kmax):
            fq[k, qx] = norm[k, qx] * omega[k, qx]


@jit(void(f4[:, :], f4[:, :]), target=processor_target, nopython=True,
     cache=cache)
def get_fq_inplace(omega, norm):
    kmax, qmax_bin = omega.shape
    for qx in xrange(i4(qmax_bin)):
        for k in xrange(kmax):
            omega[k, qx] *= norm[k, qx]


# Gradient test_kernels -------------------------------------------------------


@jit(void(f4[:, :, :], f4[:, :], f4[:], f4[:, :], f4),
     target=processor_target, nopython=True, cache=cache)
def get_grad_omega(grad_omega, omega, r, d, qbin):
    kmax, _, qmax_bin = grad_omega.shape
    for qx in xrange(i4(qmax_bin)):
        sv = f4(qx) * qbin
        for k in xrange(i4(kmax)):
            rk = r[k]
            a = sv * math.cos(sv * rk) - omega[k, qx]
            a /= rk * rk
            for w in xrange(i4(3)):
                grad_omega[k, w, qx] = a * d[k, w]


@jit(void(f4[:, :, :], f4[:, :, :], f4[:, :]),
     target=processor_target, nopython=True, cache=cache)
def get_grad_fq(grad, grad_omega, norm):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ------------
    grad: Kx3xQ numpy array
        The array which will store the FQ gradient
    grad_omega: Kx3xQ numpy array
        The gradient of omega
    norm: KxQ array
        The normalization array
    """
    kmax, _, qmax_bin = grad.shape
    for k in xrange(kmax):
        for w in xrange(i4(3)):
            for qx in xrange(i4(qmax_bin)):
                grad[k, w, qx] = norm[k, qx] * grad_omega[k, w, qx]


@jit(void(f4[:, :, :], f4[:, :], f4[:, :], f4[:, :, :], f4[:, :, :],
          f4[:, :]),
     target=processor_target, nopython=True, cache=cache)
def get_adp_grad_fq(grad, omega, tau, grad_omega, grad_tau, norm):
    """
    Generate the gradient F(sv) for an atomic configuration

    Parameters
    ------------
    grad: Nx3xQ numpy array
        The array which will store the FQ gradient
    """
    kmax, _, qmax_bin = grad.shape
    for k in xrange(kmax):
        for w in xrange(i4(3)):
            for qx in xrange(i4(qmax_bin)):
                grad[k, w, qx] = norm[k, qx] * \
                                 (tau[k, qx] * grad_omega[k, w, qx] +
                                  omega[k, qx] * grad_tau[k, w, qx])


@jit(void(f4[:, :, :], f4[:, :]), target=processor_target, nopython=True,
     cache=cache)
def get_grad_fq_inplace(grad_omega, norm):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ------------
    grad_omega: Kx3xQ numpy array
        The gradient of omega
    norm: KxQ array
        The normalization array
    """
    kmax, _, qmax_bin = grad_omega.shape
    for k in xrange(kmax):
        for w in xrange(i4(3)):
            for qx in xrange(i4(qmax_bin)):
                grad_omega[k, w, qx] *= norm[k, qx]


@jit(target=processor_target, nopython=True, cache=cache)
def fast_fast_flat_sum(new_grad, grad, k_cov):
    n = len(new_grad)
    k_max = len(grad)
    for i in xrange(n):
        for j in xrange(n):
            alpha = 0
            if j < i:
                k = j + i * (i - 1) / 2 - k_cov
                alpha = -1
            elif i < j:
                k = i + j * (j - 1) / 2 - k_cov
                alpha = 1
            else:
                k = -1
            if 0 <= k < k_max:
                for qx in xrange(grad.shape[2]):
                    for tz in xrange(i4(3)):
                        new_grad[i, tz, qx] += grad[k, tz, qx] * alpha


@jit(void(f4[:, :], f4[:, :], f4[:], i4[:], i4),
     target=processor_target,
     nopython=True, cache=cache)
def get_voxel_distances(r, q, resolution, v, offset):
    om, n = r.shape
    for o in xrange(om + offset):
        i, j, k = index1d_to_3d(o, v)
        x = (i + .5) * resolution[0]
        y = (j + .5) * resolution[1]
        z = (k + .5) * resolution[2]
        for l in xrange(n):
            a = x - q[l, 0]
            b = y - q[l, 1]
            c = z - q[l, 2]
            r[o, l] = math.sqrt(a * a + b * b + c * c)


@jit(void(f4[:, :, :], f4[:, :], f4), target=processor_target,
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
    om, n, qmax_bin = omega.shape
    for o in xrange(om):
        for qx in xrange(i4(qmax_bin)):
            sv = f4(qx) * qbin
            for l in xrange(n):
                rij = r[o, l]
                if rij != 0.0:
                    omega[o, l, qx] = math.sin(sv * rij) / rij


@jit(void(f4[:, :], f4[:, :, :], f4[:, :]),
     target=processor_target, nopython=True,
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
    om, n, qmax_bin = omega.shape
    for o in xrange(om):
        for qx in xrange(i4(qmax_bin)):
            for l in xrange(n):
                fq[o, qx] += omega[o, l, qx] * norm[l, qx]
