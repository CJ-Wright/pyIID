__author__ = 'christopher'
from pyiid.kernels.serial_kernel import *


def wrap_atoms(atoms, qmax=25., qbin=.1):
    """
    Call this function before applying calculator, it will generate static
    arrays for the scattering, preventing recalculation
    :param atoms:
    :param qmax:
    :param qbin:
    :return:
    """

    n = len(atoms)
    qmax_bin = int(qmax / qbin)
    e_num = atoms.get_atomic_numbers()

    scatter_array = np.zeros((n, qmax_bin), dtype=np.float32)
    get_scatter_array(scatter_array, e_num, qbin)
    atoms.set_array('scatter', scatter_array)


def wrap_fq(atoms, qmax=25., qbin=.1):
    """
    Generate the reduced structure function

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

    fq:1darray
        The reduced structure function
    """
    fq_func = atoms.fq_calc
    return fq_func(atoms, qmax, qbin)


def wrap_pdf(atoms, qmax=25., qmin=0.0, qbin=.1, rmax=40., rstep=.01):
    """
    Generate the atomic pair distribution function

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic config   uration
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


def wrap_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40.,
            rstep=.01):
    """
    Generate the Rw value

    Parameters
    -----------
    :param rmin:
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
    g_calc = g_calc[math.floor(rmin/rstep):]
    rw, scale = get_rw(gobs, g_calc, weight=None)
    return rw, scale, g_calc, fq


def wrap_chi_sq(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40., rstep=.01):
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
    g_calc = g_calc[math.floor(rmin/rstep):]
    rw, scale = get_chi_sq(gobs, g_calc)
    return rw, scale, g_calc, fq


def wrap_fq_grad(atoms, qmax=25., qbin=.1):
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
    return atoms.fq_grad_calc(atoms, qmax, qbin)


def wrap_grad_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40.,
                 rstep=.01, rw=None, gcalc=None, scale=None):
    """
    Generate the Rw value gradient

    Parameters
    -----------
    :param rmin:
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
        rw, scale, gcalc, fq = wrap_rw(atoms, gobs, qmax, qmin, qbin,
                                       rmin = rmin, rmax=rmax, rstep=rstep)
    fq_grad = wrap_fq_grad(atoms, qmax, qbin)
    qmin_bin = int(qmin / qbin)
    for tx in range(len(atoms)):
        for tz in range(3):
            fq_grad[tx, tz, :qmin_bin] = 0.
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep))
    pdf_grad = pdf_grad[:,:,math.floor(rmin/rstep):]
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_rw(grad_rw, pdf_grad, gcalc, gobs, rw, scale, weight=None)
    return grad_rw


def wrap_grad_chi_sq(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40.,
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
        rw, scale, gcalc, fq = wrap_rw(atoms, gobs, qmax, qmin, qbin,
                                       rmax=rmax, rstep=rstep)
    fq_grad = wrap_fq_grad(atoms, qmax, qbin)
    qmin_bin = int(qmin / qbin)
    for tx in range(len(atoms)):
        for tz in range(3):
            fq_grad[tx, tz, :qmin_bin] = 0.
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep))
    pdf_grad = pdf_grad[:,:,math.floor(rmin/rstep):]
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_chi_sq(grad_rw, pdf_grad, gcalc, gobs, scale)
    return grad_rw