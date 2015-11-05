import numpy as np

from pyiid.experiments.elasticscatter.kernels.cpu_flat import *
from pyiid.experiments.elasticscatter.kernels.cpu_experimental import \
    experimental_sum_grad_cpu

from pyiid.experiments.elasticscatter.kernels import antisymmetric_reshape, \
    symmetric_reshape

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
    sum_type: 'fq' or 'pdf'
        Determines the type of scatter array to use
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

    adps = None
    if hasattr(atoms, 'adp'):
        adps = atoms.adp.get_position().astype(np.float32)
    elif hasattr(atoms, 'adps'):
        adps = atoms.adps.get_position().astype(np.float32)
    # get non-normalized fq
    if adps is None:
        get_fq_inplace(omega, norm)
        fq = omega
    else:
        sigma = np.zeros(k_max, np.float32)
        get_sigma_from_adp(sigma, adps, r, d, k_cov)

        tau = np.zeros((k_max, qmax_bin), np.float32)
        get_tau(tau, sigma, qbin)

        fq = np.zeros((k_max, qmax_bin), np.float32)
        get_adp_fq(fq, omega, tau, norm)
        del tau, sigma, adps

    # Normalize fq
    # '''
    fq = np.sum(fq, 0, dtype=np.float32)
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
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment

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

    adps = None
    if hasattr(atoms, 'adp'):
        adps = atoms.adp.get_position().astype(np.float32)
    elif hasattr(atoms, 'adps'):
        adps = atoms.adps.get_position().astype(np.float32)
    if adps is None:
        get_grad_fq_inplace(grad_omega, norm)
        grad = grad_omega
    else:
        sigma = np.zeros(k_max, np.float32)
        get_sigma_from_adp(sigma, adps, r, d, k_cov)

        tau = np.zeros((k_max, qmax_bin), np.float32)
        get_tau(tau, sigma, qbin)

        grad_tau = np.zeros((k_max, 3, qmax_bin), np.float32)
        get_grad_tau(grad_tau, tau, r, d, sigma, adps, qbin, k_cov)

        grad = np.empty((k_max, 3, qmax_bin), np.float32)
        get_adp_grad_fq(grad, omega, tau, grad_omega, grad_tau, norm)

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
