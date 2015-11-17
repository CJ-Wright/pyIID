import numpy as np
from pyiid.experiments.elasticscatter.kernels.cpu_flat import *
from pyiid.experiments.elasticscatter.kernels.cpu_experimental import \
    experimental_sum_grad_cpu

__author__ = 'christopher'


def cpu_k_space_fq_allocation(mem, n, sv):
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


def k_space_grad_fq_allocation(mem, n, qmax_bin):
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


def voxel_fq_allocation(mem, n, qmax_bin, v):
    # return 2
    return int(math.floor(
        float(.8 * mem / 4 - 3 * n - qmax_bin * n) / (
        qmax_bin * n + qmax_bin + 1)))


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


def atomic_voxel_fq(task):
    q, norm, fq, na, qbin, resolution, v, v_per_thread, v_cov = task
    n, qmax_bin = norm.shape

    r = np.zeros((v_per_thread, n), np.float32)
    get_voxel_distances(r, q, resolution, v, np.int32(v_cov))

    # Get omega
    omega = np.zeros(r.shape + (qmax_bin,), np.float32)
    get_voxel_omega(omega, r, qbin)

    vfq = np.zeros((r.shape[0], qmax_bin), np.float32)
    # get non-normalized fq
    get_voxel_fq(vfq, omega, norm)
    vfq *= 2
    vfq += fq
    vfq = np.nan_to_num(vfq / na)
    return vfq
