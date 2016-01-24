import numpy as np

from pyiid.experiments.elasticscatter.kernels.cpu_flat import *
from pyiid.experiments.elasticscatter.kernels.cpu_experimental import \
    experimental_sum_grad_cpu

__author__ = 'christopher'


def cpu_k_space_fq_allocation(n, sv, mem):
    """
    Determine the maximum amount of atoms which can be placed on a gpu for a
    computation of F(Q).  This depends on how exactly the F(Q) function makes
    arrays on the GPU.

    Parameters
    ----------
    n: int
        Number of atoms
    sv:
        Size of the scatter vector
    mem: int
        Size of the GPU memory

    Returns
    -------
    int:
        The number of atom pairs which can go on the GPU
    """
    return int(math.floor(
        float(.8 * mem - 4 * sv * n - 12 * n) / (4 * (3 * sv + 4))
    ))


def k_space_grad_fq_allocation(n, qmax_bin, mem):
    """
    Determine the maximum amount of atoms which can be placed on a gpu for a
    computation of grad F(Q).  This depends on how exactly the grad F(Q)
    function makes arrays on the GPU.

    Parameters
    ----------
    n: int
        Number of atoms
    qmax_bin: int
        Size of the scatter vector
    mem: int
        Size of the GPU memory

    Returns
    -------
    int:
        The number of atom pairs which can go on the
    """
    return int(math.floor(
        float(.8 * mem - 16 * qmax_bin * n - 12 * n) / (
            16 * (2 * qmax_bin + 1))))


def atomic_fq(task):
    q, adps, scatter_array, qbin, k_max, k_cov = task
    n, qmax_bin = scatter_array.shape

    d = np.zeros((k_max, 3), np.float32)
    get_d_array(d, q, k_cov)

    r = np.zeros(k_max, np.float32)
    get_r_array(r, d)

    norm = np.zeros((k_max, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array, k_cov)

    omega = np.zeros((k_max, qmax_bin), np.float32)
    get_omega(omega, r, qbin)
    fq = np.zeros((k_max, qmax_bin), np.float32)
    get_fq(fq, omega, norm)
    del q, d, scatter_array, norm, r, omega
    return fq.sum(axis=0, dtype=np.float64)


def atomic_grad_fq(task):
    q, adps, scatter_array, qbin, k_max, k_cov = task
    n, qmax_bin = scatter_array.shape
    d = np.empty((k_max, 3), np.float32)
    get_d_array(d, q, k_cov)
    r = np.empty(k_max, np.float32)
    get_r_array(r, d)
    norm = np.empty((k_max, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array, k_cov)
    omega = np.zeros((k_max, qmax_bin), np.float32)
    get_omega(omega, r, qbin)
    grad_omega = np.zeros((k_max, 3, qmax_bin), np.float32)
    get_grad_omega(grad_omega, omega, r, d, qbin)

    grad = np.empty((k_max, 3, qmax_bin), np.float32)
    get_grad_fq(grad, grad_omega, norm)

    rtn = np.zeros((n, 3, qmax_bin), np.float32)
    experimental_sum_grad_cpu(rtn, grad, k_cov)

    del grad, q, scatter_array, omega, r, d, norm
    return rtn