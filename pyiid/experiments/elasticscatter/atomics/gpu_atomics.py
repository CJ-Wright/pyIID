import math
import numpy as np
from numba import *
from pyiid.experiments import generate_grid

__author__ = 'christopher'


def gpu_fq_atoms_allocation(mem, n, sv):
    """
    Determine the maximum amount of atoms which can be placed on a gpu for a
    computation of F(sv).  This depends on how exactly the F(sv) function makes
    arrays on the GPU.

    Parameters
    ----------
    n: int
        Number of atoms
    sv: int
        Size of the scatter vector
    mem: int
        Size of the GPU memory

    Returns
    -------
    int:
        The number of atom pairs which can go on the GPU
    """
    k = n * (n - 1) / 2
    summation_mem = 0
    k_count = k
    i = 1
    while k_count > 1:
        k_count = math.ceil(k * (512 ** -i))
        summation_mem += k_count
        i += 1

    return int(math.floor(
        float(.8 * mem / 4 - 3 * n - summation_mem - sv * n) / (sv + 4)
    ))


def atoms_per_gpu_grad_fq(mem, n, qmax_bin):
    """
    Determine the maximum amount of atoms which can be placed on a gpu for a
    computation of grad F(sv).  This depends on how exactly the grad F(sv)
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
            4 * (5 * qmax_bin + 4))))


def gpu_k_space_fq_allocation(mem, n, sv):
    return int(math.floor(
        float(.8 * mem - 4 * sv * n - 4 * sv - 12 * n) / (16 * (sv + 1))))


def gpu_k_space_grad_fq_allocation(mem, n, sv):
    return int(math.floor(
        float(.8 * mem - 16 * sv * n - 12 * n) / (16 * (2 * sv + 1))))


def voxel_fq_allocation(mem, n, qmax_bin, v):
    return int(math.floor(
        float(.8 * mem / 4 - 3 * n - qmax_bin * n) / (
        qmax_bin * n + qmax_bin + 1)))

def atomic_fq(q, adps, scatter_array, qbin, k_cov, k_per_thread):
    """
    Calculate a portion of F(sv).  This is the smallest division of the F(sv)
    function.

    Parameters
    ----------
    q: Nx3 array
        The atomic positions
    scatter_array: NxQ array
        The atomic scatter factors
    qbin: float
        The sv resolution of the experiment
    k_cov: int
        The number of atoms pairs previously run, used as an offset
    k_per_thread: int
        The number of atoms pairs to be run in this chunk

    Returns
    -------
    1darray:
        The calculated chunk of F(sv)

    """
    qmax_bin = scatter_array.shape[1]
    # load kernels
    from pyiid.experiments.elasticscatter.kernels.gpu_flat import \
        (get_d_array, get_r_array, get_normalization_array, get_omega,
         get_fq_inplace, d2_zero, d2_to_d1_cleanup_kernel, experimental_sum_fq)
    # generate grids for the GPU
    elements_per_dim_1 = [k_per_thread]
    tpb_k = [32]
    bpg_k = generate_grid(elements_per_dim_1, tpb_k)

    elements_per_dim_q = [qmax_bin]
    tpb_q = [4]
    bpg_q = generate_grid(elements_per_dim_q, tpb_q)

    elements_per_dim_2 = [k_per_thread, qmax_bin]
    tpb_kq = [1, 128]
    bpg_kq = generate_grid(elements_per_dim_2, tpb_kq)

    # generate GPU streams
    stream = cuda.stream()
    stream2 = cuda.stream()

    # calculate kernels
    dq = cuda.to_device(q, stream=stream)
    dd = cuda.device_array((k_per_thread, 3), dtype=np.float32, stream=stream)
    dr = cuda.device_array(k_per_thread, dtype=np.float32, stream=stream)

    get_d_array[bpg_k, tpb_k, stream](dd, dq, k_cov)
    get_r_array[bpg_k, tpb_k, stream](dr, dd)

    dscat = cuda.to_device(scatter_array.astype(np.float32), stream=stream2)
    dnorm = cuda.device_array((k_per_thread, qmax_bin), dtype=np.float32,
                              stream=stream2)

    get_normalization_array[bpg_kq, tpb_kq, stream2](dnorm, dscat, k_cov)

    domega = cuda.device_array((k_per_thread, qmax_bin), dtype=np.float32,
                               stream=stream2)
    get_omega[bpg_kq, tpb_kq, stream2](domega, dr, qbin)

    get_fq_inplace[bpg_kq, tpb_kq, stream2](dnorm, domega)

    dfq = dnorm

    # pseudo-recursive summation kernel
    tpb_sum = [512, 1]
    bpg_sum = generate_grid(elements_per_dim_2, tpb_sum)

    # Generate array to hold summation
    dsum = cuda.device_array((bpg_sum[0], qmax_bin), np.float32)

    # Generate grid to zero the summation array, which is smaller
    bpg_sum2 = generate_grid(bpg_sum, tpb_sum)

    # Zero the summation array
    d2_zero[bpg_sum2, tpb_sum, stream2](dsum)

    # Sum
    experimental_sum_fq[bpg_sum, tpb_sum, stream2](dsum, dfq, k_per_thread)

    # We may need to sum more than once so sum as much as needed
    while bpg_sum[0] > 1:
        bpg_sum = generate_grid(bpg_sum, tpb_sum)
        dsum2 = cuda.device_array((bpg_sum[0], qmax_bin), np.float32,
                                  stream=stream2)
        d2_zero[bpg_sum, tpb_sum, stream2](dsum2)
        experimental_sum_fq[bpg_sum, tpb_sum, stream2](dsum2, dsum,
                                                       dsum.shape[0])
        dsum = dsum2

    dfinal = cuda.device_array(qmax_bin, np.float32, stream=stream2)
    d2_to_d1_cleanup_kernel[bpg_q, tpb_q, stream2](dfinal, dsum)
    final = dfinal.copy_to_host(stream=stream2)
    # remove from memory
    del dq, dd, dscat, dr, dnorm, dfq, dfinal, dsum
    cuda.current_context().trashing.clear()
    return final


def atomic_grad_fq(q, adps, scatter_array, qbin, k_cov, k_per_thread):
    """
    Calculate a portion of the gradient of F(Q).  This is the smallest division
    of the grad F(Q) function.

    Parameters
    ----------
    q: Nx3 array
        The atomic positions
    scatter_array: NxQ array
        The atomic scatter factors
    qbin: float
        The sv resolution of the experiment
    k_cov: int
        The number of atoms pairs previously run, used as an offset
    k_per_thread: int
        The number of atoms pairs to be run in this chunk

    Returns
    -------
    1darray:
        The calculated chunk of grad F(sv)

    """
    qmax_bin = scatter_array.shape[1]

    # load kernels
    from pyiid.experiments.elasticscatter.kernels.gpu_flat import \
        (get_d_array,
         get_r_array,
         get_normalization_array,
         get_omega,
         get_grad_omega,
         zero3d,
         experimental_sum_grad_fq1, get_grad_fq)

    n = len(q)
    # generate GPU grids
    elements_per_dim_1 = [k_per_thread]
    tpb_k = [64]
    bpg_k = generate_grid(elements_per_dim_1, tpb_k)

    elements_per_dim_2 = [k_per_thread, qmax_bin]
    tpb_kq = [1, 64]
    bpg_kq = generate_grid(elements_per_dim_2, tpb_kq)

    elements_per_dim_nq = [n, qmax_bin]
    tpb_nq = [1, 64]
    bpg_nq = generate_grid(elements_per_dim_nq, tpb_nq)

    # generate GPU streams
    stream = cuda.stream()
    stream2 = cuda.stream()

    # Start calculations
    dq = cuda.to_device(q, stream=stream)
    dscat = cuda.to_device(scatter_array, stream=stream2)

    dd = cuda.device_array((k_per_thread, 3), dtype=np.float32, stream=stream)
    dr = cuda.device_array(k_per_thread, dtype=np.float32, stream=stream)
    domega = cuda.device_array((k_per_thread, qmax_bin), dtype=np.float32,
                               stream=stream2)
    dnorm = cuda.device_array((k_per_thread, qmax_bin), dtype=np.float32,
                              stream=stream2)
    dgrad_omega = cuda.device_array((k_per_thread, 3, qmax_bin),
                                    dtype=np.float32, stream=stream2)
    dgrad = cuda.device_array((k_per_thread, 3, qmax_bin), dtype=np.float32,
                              stream=stream2)
    dnew_grad = cuda.device_array((n, 3, qmax_bin), dtype=np.float32,
                                  stream=stream2)
    cuda.synchronize()

    zero3d[bpg_nq, tpb_nq, stream2](dnew_grad)
    get_d_array[bpg_k, tpb_k, stream](dd, dq, k_cov)
    get_r_array[bpg_k, tpb_k, stream](dr, dd)

    get_normalization_array[bpg_kq, tpb_kq, stream2](dnorm, dscat, k_cov)
    get_omega[bpg_kq, tpb_kq, stream](domega, dr, qbin)

    get_grad_omega[bpg_kq, tpb_kq, stream](dgrad_omega, domega, dr, dd, qbin)

    cuda.synchronize()
    get_grad_fq[bpg_kq, tpb_kq, stream2](dgrad, dgrad_omega, dnorm)

    experimental_sum_grad_fq1[bpg_kq, tpb_kq, stream2](dnew_grad, dgrad, k_cov)
    rtn = dnew_grad.copy_to_host(stream=stream2)
    del dq, dscat, dd, dr, domega, dnorm, dgrad_omega, dgrad, dnew_grad
    cuda.current_context().trashing.clear()
    return rtn


def atomic_voxel_fq(q, norm, qbin, resolution, v, v_per_thread, v_cov):
    n, qmax_bin = norm.shape
    from ..kernels.gpu_flat import (get_voxel_omega,
                                    get_voxel_fq,
                                    get_voxel_distances)
    elements_per_dim_2 = [v_per_thread, n]
    tpb_vn = [16, 16]
    bpg_vn = generate_grid(elements_per_dim_2, tpb_vn)

    elements_per_dim_3 = [v_per_thread, n, qmax_bin]
    tpb_vnq = [4, 4, 4]
    bpg_vnq = generate_grid(elements_per_dim_3, tpb_vnq)

    s1 = cuda.stream()
    s2 = cuda.stream()

    domega = cuda.device_array((v_per_thread, n, qmax_bin),
                               dtype=np.float32, stream=s2)
    dr = cuda.device_array((v_per_thread, n), dtype=np.float32, steram=s1)
    dq = cuda.to_device(q, s1)
    dresolution = cuda.to_device(resolution, s1)
    dv = cuda.to_device(v, s1)
    vfq = np.zeros((v_per_thread, qmax_bin), np.float32)
    dvfq = cuda.to_device(vfq, s2)
    get_voxel_distances[bpg_vn, tpb_vn, s1](dr, dq, dresolution, dv)
    cuda.synchronize()

    # Get omega
    get_voxel_omega[bpg_vnq, tpb_vnq, s1](domega, dr, qbin)
    # get non-normalized fq
    get_voxel_fq[bpg_vnq, tpb_vnq, s1](dvfq, domega, norm)
    dvfq.to_host()

    return vfq
