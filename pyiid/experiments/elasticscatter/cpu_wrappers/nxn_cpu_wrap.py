import numpy as np
from pyiid.experiments.elasticscatter.kernels.cpu_nxn import *
from ..kernels.cpu_flat import get_normalization_array as flat_norm
from pyiid.experiments.elasticscatter.atomics import pad_pdf
from itertools import product
from ase import Atoms

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

    get_fq_inplace(omega, norm)
    fq = omega

    # Normalize fq
    fq = np.sum(fq, axis=(0, 1), dtype=np.float64)
    fq = fq.astype(np.float32)
    # fq = np.sum(fq, axis=0, dtype=np.float32)
    # fq = np.sum(fq, axis=0, dtype=np.float32)
    norm2 = np.zeros((n * (n - 1) / 2., qmax_bin), np.float32)
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

    # Get grad FQ
    get_grad_fq_inplace(grad_omega, norm)
    grad_fq = grad_omega

    # Normalize FQ
    grad_fq = grad_fq.sum(1)
    # '''
    norm = np.zeros((n * (n - 1) / 2., qmax_bin), np.float32)
    flat_norm(norm, scatter_array, 0)
    na = np.mean(norm, axis=0) * np.float32(n)
    old_settings = np.seterr(all='ignore')
    grad_fq = np.nan_to_num(grad_fq / na)
    np.seterr(**old_settings)
    del d, r, scatter_array, norm, omega, grad_omega
    return grad_fq


def wrap_pbc_fq(atoms, qbin=.1, sum_type='fq', pbc_iterations=2):
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
    pbc = atoms.pbc
    cell = atoms.cell

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
    get_periodic_normalization_array(norm, scatter_array)

    fq = np.zeros((n, n, qmax_bin), np.float32)

    diag = np.diagonal(atoms.cell).astype(np.float32)
    a = np.asarray([np.asarray(u, dtype=np.float32) * diag for u in
                    product(range(pbc_iterations), repeat=3)])
    a *= pbc
    b = np.ascontiguousarray(a).view(
        np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    us = a[idx]

    # Loop through all the possible extra boxes
    for u in us:
        d_u = d + u
        r_u = np.zeros((n, n), np.float32)
        get_r_array(r_u, d_u)
        # Get omega
        omega = np.zeros((n, n, qmax_bin), np.float32)
        if np.all(u == np.zeros(3, dtype=np.float32)):
            get_omega(omega, r_u, qbin)
        else:
            get_periodic_omega(omega, r_u, qbin)
        get_fq_inplace(omega, norm)
        fq += omega
    # Normalize fq
    fq = np.sum(fq, axis=(0, 1), dtype=np.float64)
    fq = fq.astype(np.float32)
    # fq = np.sum(fq, axis=0, dtype=np.float32)
    # fq = np.sum(fq, axis=0, dtype=np.float32)
    norm2 = np.zeros((n * (n - 1) / 2., qmax_bin), np.float32)
    flat_norm(norm2, scatter_array, 0)
    na = np.mean(norm2, axis=0, dtype=np.float32) * np.float32(n)
    # na = np.mean(norm2, axis=0, dtype=np.float64) * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(fq / na)
    np.seterr(**old_settings)
    del q, d, r, norm, omega, na
    return fq


def wrap_2d_fq_slice(atoms, qbin=.1, sum_type='fq', slices=10):
    q = atoms.get_positions().astype(np.float32)

    # get scatter array
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')
    # define scatter_q information and initialize constants

    n, qmax_bin = scatter_array.shape
    s = np.linspace(0, atoms.get_cell()[-1, -1] + 1e-4, slices)
    for i in range(len(s) - 1):
        sub_atoms = Atoms([atom for atom in atoms if
                           s[i + 1] > atom.position[2] >= s[i]])
        sub_q = sub_atoms.get_positions().astype(np.float32)
        n = len(sub_q)
        # Get pair coordinate distance array
        d = np.zeros((n, n, 3), np.float32)
        get_d_array(d, sub_q)

        for i in range(2):
            # Get pair distance array
            r = d[:, :, i]
            print r.shape

            # Get normalization array
            norm = np.zeros((n, n, qmax_bin), np.float32)
            get_normalization_array(norm, scatter_array)

            # Get omega
            # FIXME: Problem with the some of the rx values zero off axis
            omega = np.zeros((n, n, qmax_bin), np.float32)
            get_omega(omega, r, qbin)

            get_fq_inplace(omega, norm)
            fq = omega

        # Normalize fq
        fq = np.sum(fq, axis=(0, 1), dtype=np.float64)
        fq = fq.astype(np.float32)
        # fq = np.sum(fq, axis=0, dtype=np.float32)
        # fq = np.sum(fq, axis=0, dtype=np.float32)
        norm2 = np.zeros((n * (n - 1) / 2., qmax_bin), np.float32)
        flat_norm(norm2, scatter_array, 0)
        na = np.mean(norm2, axis=0, dtype=np.float32) * np.float32(n)
        # na = np.mean(norm2, axis=0, dtype=np.float64) * n
        old_settings = np.seterr(all='ignore')
        fq = np.nan_to_num(fq / na)
        np.seterr(**old_settings)
        yield fq
        # del q, d, r, norm, omega, na
        # return fq
