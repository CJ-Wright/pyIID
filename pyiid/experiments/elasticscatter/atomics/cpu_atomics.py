import numpy as np

from pyiid.experiments.elasticscatter.kernels.cpu_flat import *
from pyiid.experiments.elasticscatter.kernels.cpu_experimental import \
    experimental_sum_grad_cpu

__author__ = 'christopher'


def cpu_k_space_fq_allocation(n, Q, mem):
    """
    Determine the maximum amount of atoms which can be placed on a gpu for a
    computation of F(Q).  This depends on how exactly the F(Q) function makes
    arrays on the GPU.

    Parameters
    ----------
    n: int
        Number of atoms
    Q: int
        Size of the scatter vector
    mem: int
        Size of the GPU memory

    Returns
    -------
    int:
        The number of atom pairs which can go on the GPU
    """
    return int(math.floor(
        float(.8 * mem - 4 * Q * n - 12 * n) / (4 * (3 * Q + 4))
    ))


def cpu_k_space_fq_adp_allocation(n, Q, mem):
    """
    Determine the maximum amount of atoms which can be placed on a gpu for a
    computation of F(Q).  This depends on how exactly the F(Q) function makes
    arrays on the GPU.

    Parameters
    ----------
    n: int
        Number of atoms
    Q: int
        Size of the scatter vector
    mem: int
        Size of the GPU memory

    Returns
    -------
    int:
        The number of atom pairs which can go on the GPU
    """
    return int(math.floor(
        float(.8 * mem - 4 * Q * n - 24 * n) / (4 * (3 * Q + 5))
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
    Q: int
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


def k_space_grad_adp_fq_allocation(n, qmax_bin, mem):
    """
    Determine the maximum amount of atoms which can be placed on a gpu for a
    computation of grad F(Q).  This depends on how exactly the grad F(Q)
    function makes arrays on the GPU.

    Parameters
    ----------
    n: int
        Number of atoms
    Q: int
        Size of the scatter vector
    mem: int
        Size of the GPU memory

    Returns
    -------
    int:
        The number of atom pairs which can go on the
    """
    return int(math.floor(
        float(.8 * mem - 16 * qmax_bin * n - 24 * n) / (
            4 * (12 * qmax_bin + 5))))


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
    if adps is None:
        fq = np.zeros((k_max, qmax_bin), np.float32)
        get_fq(fq, omega, norm)
    else:
        print type(adps)
        sigma = np.zeros(k_max, np.float32)
        get_sigma_from_adp(sigma, adps, r, d, k_cov)

        tau = np.zeros((k_max, qmax_bin), np.float32)
        get_tau(tau, sigma, qbin)

        fq = np.zeros((k_max, qmax_bin), np.float32)
        get_adp_fq(fq, omega, tau, norm)
        del tau, sigma, adps
    del q, d, scatter_array, norm, r, omega
    return fq.sum(axis=0)


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

    grad_omega = np.zeros((k_max, 3, qmax_bin))
    get_grad_omega(grad_omega, omega, r, d, qbin)

    if adps is None:
        grad = np.empty((k_max, 3, qmax_bin), np.float32)
        get_grad_fq(grad, grad_omega, norm)

    else:
        sigma = np.zeros(k_max, np.float32)
        get_sigma_from_adp(sigma, adps, r, d)

        tau = np.zeros((k_max, qmax_bin), np.float32)
        get_tau(tau, sigma, qbin)

        grad_tau = np.zeros((k_max, 3, qmax_bin))
        get_grad_tau(grad_tau, tau, r, d, sigma, adps, qbin, k_cov)

        grad = np.empty((k_max, 3, qmax_bin), np.float32)
        get_adp_grad_fq(grad, omega, tau, grad_omega, grad_tau, norm)

    # rtn = np.empty((n, 3, qmax_bin), np.float32)
    # fast_fast_flat_sum(rtn, grad, k_cov)
    rtn = np.zeros((n, 3, qmax_bin), np.float32)
    experimental_sum_grad_cpu(rtn, grad, k_cov)

    del grad, q, scatter_array, omega, r, d, norm
    return rtn
