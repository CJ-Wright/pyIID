import numpy as np
from pyiid.experiments.elasticscatter.kernels.cpu_flat import *
from pyiid.experiments.elasticscatter.kernels.cpu_experimental import \
    experimental_sum_grad_cpu
from pyiid.experiments.elasticscatter.kernels import antisymmetric_reshape, \
    symmetric_reshape
from ..kernels.master_kernel import get_single_scatter_array

__author__ = 'christopher'


def wrap_fq(atoms, qbin=.1, sum_type='fq', normalization=True):
    """
    Generate the reduced structure function

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration
    qbin: float
        The size of the scatter vector increment
    sum_type: {'fq', 'pdf'}
        Which scatter array should be used for the calculation
    normalization: bool
        If True normalize F(Q) else don't
    Returns
    -------
    fq:1darray
        The reduced structure function
    """
    q = atoms.get_positions().astype(np.float32)

    # get scatter array
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')
    # define scatter_q information and initialize constants

    n, qmax_bin = scatter_array.shape
    k_max = n * (n - 1) / 2.
    k_cov = i4(0)

    d = np.zeros((k_max, 3), np.float32)
    get_d_array(d, q, k_cov)

    r = np.zeros(k_max, np.float32)
    get_r_array(r, d)

    norm = np.zeros((k_max, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array, k_cov)

    omega = np.zeros((k_max, qmax_bin), np.float32)
    get_omega(omega, r, qbin)

    get_fq_inplace(omega, norm)
    fq = omega
    # Normalize fq
    fq = np.sum(fq, axis=0, dtype=np.float64).astype(np.float32)
    if normalization:
        na = np.mean(norm, axis=0, dtype=np.float32) * np.float32(n)
        old_settings = np.seterr(all='ignore')
        fq = np.nan_to_num(fq / na)
        np.seterr(**old_settings)
        del na
    del q, d, r, norm, omega
    return fq * 2.


def wrap_fq_grad(atoms, qbin=.1, sum_type='fq'):
    """
    Generate the reduced structure function gradient

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration
    qbin: float
        The size of the scatter vector increment
    sum_type: {'fq', 'pdf'}
        Which scatter array should be used for the calculation

    Returns
    -------

    dfq_dq:ndarray
        The reduced structure function gradient
    """
    q = atoms.get_positions().astype(np.float32)
    qbin = np.float32(qbin)

    # get scatter array
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')

    # define scatter_q information and initialize constants
    qmax_bin = scatter_array.shape[1]
    n = len(q)
    k_max = n * (n - 1) / 2.
    k_cov = 0

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

    get_grad_fq_inplace(grad_omega, norm)
    grad = grad_omega

    rtn = np.zeros((n, 3, qmax_bin), np.float32)
    experimental_sum_grad_cpu(rtn, grad, k_cov)
    # '''
    # Normalize FQ
    na = np.mean(norm, axis=0) * np.float32(n)
    old_settings = np.seterr(all='ignore')
    rtn = np.nan_to_num(rtn / na)
    np.seterr(**old_settings)
    del d, r, scatter_array, norm, omega, grad_omega
    return rtn

'''
def wrap_voxel_fq(atoms, new_atom, resolution, fq, qbin=.1, sum_type='fq'):
    """
    Generate the reduced structure function

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration
    qbin: float
        The size of the scatter vector increment
    sum_type: 'fq' or 'pdf'
        Determines the type of scatter array to use
    Returns
    -------
    fq:1darray
        The reduced structure function
    """
    # Pre-pool
    # get scatter array
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')
    new_scatter = np.zeros(scatter_array.shape[1], np.float32)
    get_single_scatter_array(new_scatter, new_atom.number, qbin)

    # define scatter_q information and initialize constants
    # Get normalization array
    norm = np.float32(scatter_array * new_scatter)
    # Set up all the variables of interest
    qbin = np.float32(qbin)
    q = atoms.get_positions().astype(np.float32)
    resolution = np.float32(resolution)
    v = tuple(np.int32(np.ceil(np.diagonal(atoms.get_cell()) / resolution)))
    n, qmax_bin = scatter_array.shape

    # Inside pool

    # Get pair coordinate distance array
    r = np.zeros((n, np.product(v)), np.float32)
    get_voxel_distances(r, q, resolution, v)

    # Get omega
    omega = np.zeros((n, qmax_bin, np.product(v)), np.float32)
    get_voxel_omega(omega, r, qbin)

    vfq = np.zeros((qmax_bin, np.product(v)), np.float32)
    # get non-normalized fq
    get_voxel_fq(vfq, omega, norm)

    # Post-Pool
    # Normalize fq
    vfq = np.reshape(vfq, (qmax_bin, ) + v)
    norm2 = np.zeros((n * (n - 1) / 2., qmax_bin), np.float32)
    flat_norm(norm2, np.vstack((scatter_array, new_scatter)), 0)
    na = np.mean(norm2, axis=0, dtype=np.float32) * np.float32(n + 1)
    im, jm, km = v
    vfq *= 2
    for i in xrange(im):
        for j in xrange(jm):
            for k in xrange(km):
                old_settings = np.seterr(all='ignore')
                vfq[:, i, j, k] += fq
                vfq[:, i, j, k] = np.nan_to_num(vfq[:, i, j, k] / na)
                np.seterr(**old_settings)
    del q, r, norm, omega, na
    return vfq
'''
