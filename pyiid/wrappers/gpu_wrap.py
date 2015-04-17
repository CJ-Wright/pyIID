__author__ = 'christopher'
import numpy as np

from pyiid.kernels.numbapro_cuda_kernels import *
from pyiid.kernels.serial_kernel import get_pdf_at_qmin, grad_pdf, get_rw, \
    get_grad_rw


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

    #cuda kernel information

    stream = cuda.stream()
    tpb = 32
    bpg = int(math.ceil(float(n) / tpb))


    #start calculations

    dscat = cuda.to_device(scatter_array, stream)
    dnorm = cuda.to_device(norm_array)
    get_normalization_array[(bpg, bpg), (tpb, tpb), stream](dnorm, dscat)

    dd = cuda.to_device(d, stream)
    dq = cuda.to_device(q, stream)
    dr = cuda.to_device(r, stream)
    dfq = cuda.to_device(super_fq, stream)

    get_d_array[(bpg, bpg), (tpb, tpb), stream](dd, dq)
    cuda.synchronize()

    get_r_array[(bpg, bpg), (tpb, tpb), stream](dr, dd)

    cuda.synchronize()
    get_fq_p0[(bpg, bpg), (tpb, tpb), stream](dfq, dr, qbin)
    cuda.synchronize()

    get_fq_p1[(bpg, bpg), (tpb, tpb), stream](dfq)
    cuda.synchronize()
    dnorm.to_host()

    get_fq_p2[(bpg, bpg), (tpb, tpb), stream](dfq, dr)
    cuda.synchronize()

    get_fq_p3[(bpg, bpg), (tpb, tpb), stream](dfq, dnorm)
    cuda.synchronize()
    dfq.to_host(stream)
    fq = super_fq.sum(axis=(0, 1))
    na = norm_array.sum(axis=(0, 1))

    na *= 1. / (scatter_array.shape[0] ** 2)
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / (n * na) * fq)
    np.seterr(**old_settings)
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


def wrap_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmax=40., rstep=.01):
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

    #cuda info
    stream = cuda.stream()
    tpb = 32
    bpg = int(math.ceil(float(n) / tpb))

    #start calculations
    dscat = cuda.to_device(scatter_array, stream)
    dnorm = cuda.to_device(norm_array)
    get_normalization_array[(bpg, bpg), (tpb, tpb), stream](dnorm, dscat)

    dd = cuda.to_device(d, stream)
    dq = cuda.to_device(q, stream)
    dr = cuda.to_device(r, stream)
    dfq = cuda.to_device(super_fq, stream)

    get_d_array[(bpg, bpg), (tpb, tpb), stream](dd, dq)
    cuda.synchronize()

    get_r_array[(bpg, bpg), (tpb, tpb), stream](dr, dd)

    cuda.synchronize()
    dr.to_host(stream)
    get_fq_p0[(bpg, bpg), (tpb, tpb), stream](dfq, dr, qbin)
    cuda.synchronize()

    cuda.synchronize()

    get_fq_p1[(bpg, bpg), (tpb, tpb), stream](dfq)
    cuda.synchronize()

    get_fq_p2[(bpg, bpg), (tpb, tpb), stream](dfq, dr)
    cuda.synchronize()
    get_fq_p3[(bpg, bpg), (tpb, tpb), stream](dfq, dnorm)
    cuda.synchronize()
    dfq.to_host(stream)

    rgrad = np.zeros((n, n, 3), dtype=np.float32)
    q_over_r = np.zeros((n, n, qmax_bin), dtype=np.float32)
    cos_term = np.zeros((n, n, qmax_bin), dtype=np.float32)
    grad_p = np.zeros((n, n, 3, qmax_bin), dtype=np.float32)
    kqr = np.zeros((n, n, qmax_bin), dtype=np.float32)

    drgrad = cuda.to_device(rgrad, stream)
    dq_over_r = cuda.to_device(q_over_r, stream)
    dcos_term = cuda.to_device(cos_term, stream)
    dgrad_p = cuda.to_device(grad_p, stream)
    dkqr = cuda.to_device(kqr, stream)

    get_fq_p0[(bpg, bpg), (tpb, tpb), stream](dkqr, dr, qbin)
    cuda.synchronize()

    fq_grad_position1[(bpg, bpg), (tpb, tpb), stream](dq_over_r, dr, qbin)  #OK
    cuda.synchronize()

    fq_grad_position0[(bpg, bpg), (tpb, tpb), stream](drgrad, dd, dr)  #OK
    fq_grad_position3[(bpg, bpg), (tpb, tpb), stream](dcos_term, dkqr)  #OK
    cuda.synchronize()

    fq_grad_position4[(bpg, bpg), (tpb, tpb), stream](dcos_term,
                                                      dq_over_r)  #OK
    cuda.synchronize()

    fq_grad_position5[(bpg, bpg), (tpb, tpb), stream](dcos_term, dnorm)  #OK
    cuda.synchronize()

    fq_grad_position6[(bpg, bpg), (tpb, tpb), stream](dfq, dr)  #OK
    cuda.synchronize()

    fq_grad_position7[(bpg, bpg), (tpb, tpb), stream](dcos_term, dfq)  #OK
    cuda.synchronize()

    fq_grad_position_final1[(bpg, bpg), (tpb, tpb), stream](dgrad_p,
                                                            drgrad)  #OK
    cuda.synchronize()

    fq_grad_position_final2[(bpg, bpg), (tpb, tpb), stream](dgrad_p,
                                                            dcos_term)  #ok
    dgrad_p.to_host(stream)


    #sum down to 1D array
    grad_p = grad_p.sum(axis=1)
    # print grad_p
    dnorm.to_host(stream)

    #sum reduce to 1D
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


def wrap_grad_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmax=40., rstep=.01,
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
            fq_grad[tx, tz, :qmin_bin] = 0
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep))
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_rw(grad_rw, pdf_grad, gcalc, gobs, rw, scale, weight=None)
    # print 'scale:', scale
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
