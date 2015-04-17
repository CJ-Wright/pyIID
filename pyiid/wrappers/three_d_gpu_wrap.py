__author__ = 'christopher'
import numpy as np

from pyiid.kernels.three_d_cuda import *
from pyiid.kernels.serial_kernel import get_pdf_at_qmin, grad_pdf, get_rw, \
    get_grad_rw, get_chi_sq, get_grad_chi_sq


def wrap_fq(atoms, qmax=25., qbin=.1):
    # get information for FQ transformation

    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    # build the empty arrays

    d = np.zeros((n, n, 3), dtype=np.float32)
    r = np.zeros((n, n), dtype=np.float32)
    super_fq = np.zeros((n, n, qmax_bin), dtype=np.float32)
    norm_array = np.zeros((n, n, qmax_bin), dtype=np.float32)

    # cuda kernel information

    stream = cuda.stream()
    stream2 = cuda.stream()

    # two kinds of test_kernels; NxN or NxNxQ

    # NXN
    elements_per_dim_2 = [n, n]
    tpb_l_2 = [32, 32]
    bpg_l_2 = []
    for e_dim, tpb in zip(elements_per_dim_2, tpb_l_2):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_2.append(bpg)

    # NxNxQ
    elements_per_dim_3 = [n, n, qmax_bin]
    tpb_l_3 = [16, 16, 4]
    bpg_l_3 = []
    for e_dim, tpb in zip(elements_per_dim_3, tpb_l_3):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_3.append(bpg)

    # start calculations

    dscat = cuda.to_device(scatter_array, stream2)
    dnorm = cuda.to_device(norm_array, stream2)
    get_normalization_array[bpg_l_3, tpb_l_3, stream2](dnorm, dscat)

    dd = cuda.to_device(d, stream)
    dq = cuda.to_device(q, stream)
    dr = cuda.to_device(r, stream)
    dfq = cuda.to_device(super_fq, stream)

    get_d_array[bpg_l_2, tpb_l_2, stream](dd, dq)

    get_r_array[bpg_l_2, tpb_l_2, stream](dr, dd)

    get_fq_p0_1_2[bpg_l_3, tpb_l_3, stream](dfq, dr, qbin, dnorm)
    get_fq_p3[bpg_l_3, tpb_l_3, stream](dfq, dnorm)
    dnorm.to_host(stream2)


    # cuda.synchronize()



    dfq.to_host(stream)
    fq = super_fq.sum(axis=(0, 1))
    na = norm_array.sum(axis=(0, 1))

    na *= 1. / (scatter_array.shape[0] ** 2)
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / (n * na) * fq)
    np.seterr(**old_settings)
    del dfq, dnorm, dscat, dd, dq, dr
    return fq


def wrap_pdf(atoms, qmax=25., qmin=0.0, qbin=.1, rmax=40., rstep=.01):
    """
    Generate the atomic pair distribution function

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------

    pdf0:1darray
        The atomic pair distributuion function
    fq:1darray
        The reduced structure function
    """
    qmin_bin = int(qmin / qbin)
    fq = wrap_fq(atoms, qmax, qbin)
    fq[:qmin_bin] = 0
    pdf0 = get_pdf_at_qmin(fq, rstep, qbin, np.arange(0, rmax, rstep))
    return pdf0, fq


def wrap_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40., rstep=.01):
    """
    Generate the Rw value

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration
    gobs: 1darray
        The observed atomic pair distributuion function
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------

    rw: float
        The Rw value in percent
    scale: float
        The scale factor between the observed and calculated PDF
    pdf0:1darray
        The atomic pair distributuion function
    fq:1darray
        The reduced structure function
    """
    g_calc, fq = wrap_pdf(atoms, qmax, qmin, qbin, rmax, rstep)
    rw, scale = get_rw(gobs, g_calc, weight=None)
    return rw, scale, g_calc, fq


def wrap_chi_sq(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmax=40., rstep=.01):
    """
    Generate the Rw value

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration
    gobs: 1darray
        The observed atomic pair distributuion function
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------

    rw: float
        The Rw value in percent
    scale: float
        The scale factor between the observed and calculated PDF
    pdf0:1darray
        The atomic pair distributuion function
    fq:1darray
        The reduced structure function
    """
    g_calc, fq = wrap_pdf(atoms, qmax, qmin, qbin, rmax, rstep)
    rw, scale = get_chi_sq(gobs, g_calc)
    return rw, scale, g_calc, fq


def wrap_fq_grad_gpu(atoms, qmax=25., qbin=.1):
    # atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    # build empty arrays
    d = np.zeros((n, n, 3), dtype=np.float32)
    r = np.zeros((n, n), dtype=np.float32)
    norm_array = np.zeros((n, n, qmax_bin), dtype=np.float32)
    super_fq = np.zeros((n, n, qmax_bin), dtype=np.float32)
    grad_p = np.zeros((n, n, 3, qmax_bin), dtype=np.float32)
    cos_term = np.zeros((n, n, qmax_bin), dtype=np.float32)

    # cuda info
    stream = cuda.stream()
    stream2 = cuda.stream()

    # two kinds of test_kernels; NxN or NxNxQ

    # NXN
    elements_per_dim_2 = [n, n]
    tpb_l_2 = [32, 32]
    bpg_l_2 = []
    for e_dim, tpb in zip(elements_per_dim_2, tpb_l_2):
        bpg_l_2.append(int(math.ceil(float(e_dim) / tpb)))

    # NxNxQ
    elements_per_dim_3 = [n, n, qmax_bin]
    tpb_l_3 = [16, 16, 4]
    bpg_l_3 = []
    for e_dim, tpb in zip(elements_per_dim_3, tpb_l_3):
        bpg_l_3.append(int(math.ceil(float(e_dim) / tpb)))

    # START CALCULATIONS-----------------------------------------------------------

    dscat = cuda.to_device(scatter_array, stream2)
    dnorm = cuda.to_device(norm_array, stream2)
    '--------------------------------------------------------------'
    get_normalization_array[bpg_l_3, tpb_l_3, stream2](dnorm, dscat)
    '--------------------------------------------------------------'
    dd = cuda.to_device(d, stream)
    dr = cuda.to_device(r, stream)
    dfq = cuda.to_device(super_fq, stream)
    dq = cuda.to_device(q, stream)

    get_d_array[bpg_l_2, tpb_l_2, stream](dd, dq)
    # cuda.synchronize()

    get_r_array[bpg_l_2, tpb_l_2, stream](dr, dd)

    '--------------------------------------------------------------'
    get_fq_p0_1_2[bpg_l_3, tpb_l_3, stream](dfq, dr, qbin, dnorm)
    '--------------------------------------------------------------'
    dcos_term = cuda.to_device(cos_term, stream2)
    # cuda.synchronize()


    get_fq_p3[bpg_l_3, tpb_l_3, stream](dfq, dnorm)
    fq_grad_position3[bpg_l_3, tpb_l_3, stream](dcos_term, dr, qbin)  # OK
    dgrad_p = cuda.to_device(grad_p, stream2)
    # cuda.synchronize()


    # cuda.synchronize()

    fq_grad_position_final1[bpg_l_3, tpb_l_3, stream](dgrad_p, dd, dr)  # OK
    fq_grad_position5[bpg_l_3, tpb_l_3, stream2](dcos_term, dnorm)  # OK
    # cuda.synchronize()

    fq_grad_position7[bpg_l_3, tpb_l_3, stream](dcos_term, dfq, dr)  # OK
    # cuda.synchronize()

    fq_grad_position_final2[bpg_l_3, tpb_l_3, stream](dgrad_p, dcos_term)  # Ok
    dnorm.to_host(stream2)
    dgrad_p.to_host(stream)



    # sum down to 1D array
    grad_p = grad_p.sum(axis=1)
    # print grad_p


    # sum reduce to 1D
    na = norm_array.sum(axis=(0, 1))
    na *= 1. / (scatter_array.shape[0] ** 2)

    old_settings = np.seterr(all='ignore')
    # print na
    for tx in range(n):
        for tz in range(3):
            # grad_p[tx, tz, :qmin_bin] = 0.0
            grad_p[tx, tz] = np.nan_to_num(
                1 / (n * na) * grad_p[tx, tz])
    np.seterr(**old_settings)
    return grad_p


def wrap_grad_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40., rstep=.01,
                 rw=None, gcalc=None, scale=None):
    """
    Generate the Rw value gradient

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration
    gobs: 1darray
        The observed atomic pair distributuion function
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------

    grad_rw: float
        The gradient of the Rw value with respect to the atomic positions,
        in percent

    """
    if rw is None:
        rw, scale, gcalc, fq = wrap_rw(atoms, gobs, qmax, qmin, qbin, rmin, rmax,
                                       rstep)
    fq_grad = wrap_fq_grad_gpu(atoms, qmax, qbin)
    qmin_bin = int(qmin / qbin)
    for tx in range(len(atoms)):
        for tz in range(3):
            fq_grad[tx, tz, :qmin_bin] = 0.
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep))
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_rw(grad_rw, pdf_grad, gcalc, gobs, rw, scale, weight=None)
    return grad_rw


def wrap_grad_chi_sq(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmax=40.,
                     rstep=.01,
                     rw=None, gcalc=None, scale=None):
    """
    Generate the Rw value gradient

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration
    gobs: 1darray
        The observed atomic pair distributuion function
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------

    grad_rw: float
        The gradient of the Rw value with respect to the atomic positions,
        in percent

    """
    if rw is None:
        rw, scale, gcalc, fq = wrap_rw(atoms, gobs, qmax, qmin, qbin, rmax,
                                       rstep)
    fq_grad = wrap_fq_grad_gpu(atoms, qmax, qbin)
    qmin_bin = int(qmin / qbin)
    for tx in range(len(atoms)):
        for tz in range(3):
            fq_grad[tx, tz, :qmin_bin] = 0.
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep))
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_chi_sq(grad_rw, pdf_grad, gcalc, gobs, None)
    return grad_rw


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('''
    import ase.io as aseio
    import os
    from pyiid.wrappers.serial_cpu_wrap import wrap_atoms
    import matplotlib.pyplot as plt

    atoms_file = '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/2-AuNP-DFT.d/SizeVariation.d/Au55.xyz'
    atoms_file_no_ext = os.path.splitext(atoms_file)[0]
    atomsio = aseio.read(atoms_file)
    wrap_atoms(atomsio)
    atomsio *= (10, 1, 1)

    fq = wrap_fq(atomsio)
    plt.plot(fq), plt.show()
    # for i in range(10):
    # gfq = wrap_fq_grad_gpu(atomsio)
    # ''', sort='tottime')