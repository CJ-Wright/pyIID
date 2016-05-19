import numpy as np
from pyiid.experiments.elasticscatter.kernels.cpu_nxn import *
from ..kernels.cpu_flat import get_normalization_array as flat_norm
from pyiid.experiments.elasticscatter.atomics import pad_pdf
from pyiid.adp import has_adp
__author__ = 'christopher'


def wrap_fq(atoms, qbin=.1, sum_type='fq'):
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
    # Get pair coordinate distance array
    d = np.zeros((n, n, 3), np.float32)
    get_d_array(d, q)

    # Get pair distance array
    r = np.zeros((n, n), np.float32)
    get_r_array(r, d)

    # Get normalization array
    norm = np.zeros((n, n, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array)

    # Get omega
    omega = np.zeros((n, n, qmax_bin), np.float32)
    get_omega(omega, r, qbin)

    adps = has_adp(atoms)
    if adps:
        adps = adps.get_positions().astype(np.float32)
    # get non-normalized fq
    if adps is None:
        get_fq_inplace(omega, norm)
        fq = omega
    else:
        sigma = np.zeros((n, n), np.float32)
        get_sigma_from_adp(sigma, adps, r, d)

        tau = np.zeros((n, n, qmax_bin), np.float32)
        get_tau(tau, sigma, qbin)

        fq = np.zeros((n, n, qmax_bin), np.float32)
        get_adp_fq(fq, omega, tau, norm)
        del tau, sigma, adps

    # Normalize fq
    fq = np.sum(fq, axis=(0, 1), dtype=np.float64)
    fq = fq.astype(np.float32)

    norm2 = np.zeros((int(n * (n - 1) / 2.), qmax_bin), np.float32)
    flat_norm(norm2, scatter_array, 0)
    na = np.mean(norm2, axis=0, dtype=np.float32) * np.float32(n)
    # na = np.mean(norm2, axis=0, dtype=np.float64) * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(fq / na)
    np.seterr(**old_settings)
    del q, d, r, norm, omega, na
    return fq


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


    # Get pair coordinate distance array
    d = np.zeros((n, n, 3), np.float32)
    get_d_array(d, q)

    # Get pair distance array
    r = np.zeros((n, n), np.float32)
    get_r_array(r, d)

    # Get normalization array
    norm = np.zeros((n, n, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array)

    # Get omega
    omega = np.zeros((n, n, qmax_bin), np.float32)
    get_omega(omega, r, qbin)

    # Get grad omega
    grad_omega = np.zeros((n, n, 3, qmax_bin), np.float32)
    get_grad_omega(grad_omega, omega, r, d, qbin)

    adps = has_adp(atoms)
    if adps:
        adps = adps.get_positions().astype(np.float32)

    if adps is None:
        # Get grad FQ
        get_grad_fq_inplace(grad_omega, norm)
        grad_fq = grad_omega
    else:
        sigma = np.zeros((n, n), np.float32)
        get_sigma_from_adp(sigma, adps, r, d)

        tau = np.zeros((n, n, qmax_bin), np.float32)
        get_tau(tau, sigma, qbin)

        grad_tau = np.zeros((n, n, 3, qmax_bin), np.float32)
        get_grad_tau(grad_tau, tau, r, d, sigma, adps, qbin)

        grad_fq = np.zeros((n, n, 3, qmax_bin), np.float32)
        get_adp_grad_fq(grad_fq, omega, tau, grad_omega, grad_tau, norm)
        del tau, sigma, adps

    # Normalize FQ
    grad_fq = grad_fq.sum(1)
    # '''
    norm = np.zeros((int(n * (n - 1) / 2.), qmax_bin), np.float32)
    flat_norm(norm, scatter_array, 0)
    na = np.mean(norm, axis=0) * np.float32(n)
    old_settings = np.seterr(all='ignore')
    grad_fq = np.nan_to_num(grad_fq / na)
    np.seterr(**old_settings)
    del d, r, scatter_array, norm, omega, grad_omega
    return grad_fq


def wrap_fq_dadp(atoms, qbin=.1, sum_type='fq'):
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

    adps = has_adp(atoms)
    if adps:
        adps = adps.get_positions().astype(np.float32)
    if adps is None:
        return np.zeros((n, 3, qmax_bin), np.float32)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3), np.float32)
    get_d_array(d, q)

    # Get pair distance array
    r = np.zeros((n, n), np.float32)
    get_r_array(r, d)

    # Get normalization array
    norm = np.zeros((n, n, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array)

    # Get omega
    omega = np.zeros((n, n, qmax_bin), np.float32)
    get_omega(omega, r, qbin)

    sigma = np.zeros((n, n), np.float32)
    get_sigma_from_adp(sigma, adps, r, d)

    tau = np.zeros((n, n, qmax_bin), np.float32)
    get_tau(tau, sigma, qbin)

    dtau_dadp = np.zeros((n, n, 3, qmax_bin), np.float32)
    get_dtau_dadp(dtau_dadp, tau, sigma, r, d, qbin)

    get_dfq_dadp_inplace(dtau_dadp, omega, norm)
    grad_fq = dtau_dadp
    del tau, sigma, adps

    # Normalize FQ
    grad_fq = grad_fq.sum(1)
    # '''
    norm = np.zeros((int(n * (n - 1) / 2.), qmax_bin), np.float32)
    flat_norm(norm, scatter_array, 0)
    na = np.mean(norm, axis=0) * np.float32(n)
    old_settings = np.seterr(all='ignore')
    grad_fq = np.nan_to_num(grad_fq / na)
    np.seterr(**old_settings)
    del d, r, scatter_array, norm, omega
    return grad_fq