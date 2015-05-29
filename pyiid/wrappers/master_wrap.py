from pyiid.wrappers.scatter import wrap_atoms

__author__ = 'christopher'
from numba import cuda
import numpy as np
import math

from pyiid.kernels.master_kernel import get_pdf_at_qmin, \
    get_rw, grad_pdf, get_grad_rw, get_chi_sq, get_grad_chi_sq


def set_processor(processor=None):
    avail_pro = ['MPI-GPU', 'Multi-GPU', 'Serial-CPU']
    if processor == 'MPI-GPU':
        # TEST FOR MPI
        assert os.getenv('MPI_COMM_WORLD_SIZE') is not None
        from pyiid.wrappers.mpi_gpu_wrap import \
            wrap_fq as low_wrap_fq
        from pyiid.wrappers.mpi_gpu_wrap import \
            wrap_fq_grad as low_wrap_fq_grad

        return processor, low_wrap_fq, low_wrap_fq_grad

    elif processor == 'Multi-GPU':
        cuda.gpus.lst
        # cuda.get_current_device()
        from pyiid.wrappers.multi_gpu_wrap import \
            wrap_fq as low_wrap_fq
        from pyiid.wrappers.multi_gpu_wrap import \
            wrap_fq_grad as low_wrap_fq_grad

        return processor, low_wrap_fq, low_wrap_fq_grad

        # cuda.close()

    elif processor == 'Serial-CPU':
        from pyiid.wrappers.cpu_wrap import wrap_fq as low_wrap_fq
        from pyiid.wrappers.cpu_wrap import \
            wrap_fq_grad as low_wrap_fq_grad

        return processor, low_wrap_fq, low_wrap_fq_grad
    else:
        # Test which kernel to use: MPI, Single Node-MultiGPU, Serial CPU
        for i in avail_pro:
            try:
                if i == 'MPI-GPU':
                    # TEST FOR MPI
                    assert os.getenv('MPI_COMM_WORLD_SIZE') is not None
                    from pyiid.wrappers.mpi_gpu_wrap import \
                        wrap_fq as low_wrap_fq
                    from pyiid.wrappers.mpi_gpu_wrap import \
                        wrap_fq_grad as low_wrap_fq_grad

                    break
                elif i == 'Multi-GPU':
                    cuda.gps.lst
                    # cuda.get_current_device()
                    from pyiid.wrappers.multi_gpu_wrap import \
                        wrap_fq as low_wrap_fq
                    from pyiid.wrappers.multi_gpu_wrap import \
                        wrap_fq_grad as low_wrap_fq_grad

                    # cuda.close()
                    break
                elif i == 'Serial-CPU':
                    from pyiid.wrappers.cpu_wrap import wrap_fq as low_wrap_fq
                    from pyiid.wrappers.cpu_wrap import \
                        wrap_fq_grad as low_wrap_fq_grad

                    break
            except:
                continue
    return i, low_wrap_fq, low_wrap_fq_grad


processor, low_wrap_fq, low_wrap_fq_grad = set_processor()


def wrap_fq(atoms, qmax=25, qbin=.1):
    return low_wrap_fq(atoms, qmax, qbin)


def wrap_pdf(atoms, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40.,
             rstep=.01):
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
    fq = wrap_fq(atoms, qmax, qbin)
    fq[:int(qmin / qbin)] = 0
    pdf0 = get_pdf_at_qmin(fq, rstep, qbin, np.arange(0, rmax, rstep), qmin)
    pdf = pdf0[int(rmin / rstep):int(rmax / rstep)]
    return pdf


def wrap_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40.,
            rstep=.01):
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
    g_calc = wrap_pdf(atoms, qmax, qmin, qbin, rmin, rmax, rstep)
    rw, scale = get_rw(gobs, g_calc, weight=None)
    return rw, scale, g_calc


def wrap_chi_sq(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40.,
                rstep=.01):
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
    g_calc = wrap_pdf(atoms, qmax, qmin, qbin, rmin, rmax, rstep)
    rw, scale = get_chi_sq(gobs, g_calc)
    return rw, scale, g_calc


def wrap_fq_grad(atoms, qmax=25., qbin=.1):
    return low_wrap_fq(atoms, qmax, qbin)


def wrap_grad_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0,
                 rmax=40., rstep=.01):
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
    rw, scale, gcalc = wrap_rw(atoms, gobs, qmax, qmin, qbin, rmin,
                                   rmax,
                                   rstep)
    fq_grad = low_wrap_fq_grad(atoms, qmax, qbin)
    qmin_bin = int(qmin / qbin)
    for tx in range(len(atoms)):
        for tz in range(3):
            fq_grad[tx, tz, :qmin_bin] = 0.
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep))
    pdf_grad = pdf_grad[:,:, math.floor(rmin/rstep):]
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_rw(grad_rw, pdf_grad, gcalc, gobs, rw, scale, weight=None)
    return grad_rw


def wrap_grad_chi_sq(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0,
                     rmax=40.,
                     rstep=.01,):
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
    rw, scale, gcalc = wrap_rw(atoms, gobs, qmax, qmin, qbin, rmax=rmax,
                               rstep=rstep)
    fq_grad = wrap_fq_grad(atoms, qmax, qbin)
    qmin_bin = int(qmin / qbin)
    for tx in range(len(atoms)):
        for tz in range(3):
            fq_grad[tx, tz, :qmin_bin] = 0.
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep))
    pdf_grad = pdf_grad[:, :, math.floor(rmin / rstep):]
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_chi_sq(grad_rw, pdf_grad, gcalc, gobs, scale)
    return grad_rw


if __name__ == '__main__':
    print processor
    from ase.atoms import Atoms
    import os
    import matplotlib.pyplot as plt
    atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    wrap_atoms(atoms)

    fq = wrap_fq(atoms)
    pdf = wrap_pdf(atoms)
    grad_fq = wrap_fq_grad(atoms)
    print grad_fq
    plt.plot(pdf), plt.show()