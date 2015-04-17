__author__ = 'christopher'
from pyiid.kernels.serial_kernel import *
import numpy
from numbapro import autojit


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


def wrap_fq(atoms, fq_func,  qmax=25., qbin=.1):
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
    q = atoms.get_positions()

    # define scatter_q information and initialize constants
    qmax_bin = int(qmax / qbin)
    n = len(q)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3))
    get_d_array(d, q)

    # Get pair distance array
    r = np.zeros((n, n))
    get_r_array(r, d)

    # get scatter array
    scatter_array =atoms.get_array('scatter')

    # get non-normalized fq
    fq = np.zeros(qmax_bin)
    get_fq_array(fq, r, scatter_array, qbin)


    #Normalize fq
    norm_array = np.zeros((n, n, qmax_bin), dtype=np.float32)
    get_normalization_array(norm_array, scatter_array)

    norm_array = norm_array.sum(axis=(0, 1))
    norm_array *= 1. / (scatter_array.shape[0] ** 2)

    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / (n * norm_array) * fq)
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
    fq = wrap_fq(atoms, qmax, qbin)
    fq[:int(qmin/qbin)] = 0
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
    g_calc = g_calc[math.floor(rmin/rstep):]
    rw, scale = get_rw(gobs, g_calc, weight=None)
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
    q = atoms.get_positions()

    # define scatter_q information and initialize constants
    scatter_q = np.arange(0, qmax, qbin)
    n = len(q)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3))
    get_d_array(d, q)

    # Get pair distance array
    r = np.zeros((n, n))
    get_r_array(r, d)

    # get scatter array
    scatter_array = atoms.get_array('scatter')

    # get non-normalized FQ

    #Normalize FQ
    norm_array = np.zeros((n, n, len(scatter_q)))
    get_normalization_array(norm_array, scatter_array)
    norm_array = norm_array.sum(axis=(0, 1))
    norm_array *= 1. / (scatter_array.shape[0] ** 2)

    dfq_dq = np.zeros((n, 3, len(scatter_q)))
    fq_grad_position(dfq_dq, d, r, scatter_array, qbin)
    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            dfq_dq[tx, tz] = np.nan_to_num(
                1 / (n * norm_array) * dfq_dq[tx, tz])
    np.seterr(**old_settings)
    return dfq_dq


def wrap_grad_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1,rmin=0.0,
                 rmax=40., rstep=.01, rw=None, gcalc=None, scale=None):
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
    fq_grad = wrap_fq_grad(atoms, qmax, qbin)
    qmin_bin = int(qmin / qbin)
    for tx in range(len(atoms)):
        for tz in range(3):
            fq_grad[tx, tz, :qmin_bin] = 0.
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep))
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_rw(grad_rw, pdf_grad, gcalc, gobs, rw, scale, weight=None)
    return grad_rw

@autojit
def spring_nrg(atoms, k, rt):
    q = atoms.positions
    n = len(atoms)
    d = numpy.zeros((n, n, 3))
    get_d_array(d, q)
    r = numpy.zeros((n, n))
    get_r_array(r, d)

    thresh = numpy.less(r, rt)
    for i in range(len(thresh)):
        thresh[i,i] = False

    mag = numpy.zeros(r.shape)
    mag[thresh] = k * (r[thresh]-rt)

    energy = numpy.sum(mag[thresh]/2.*(r[thresh]-rt))
    return energy

@autojit
def spring_force(atoms, k, rt):
    q = atoms.positions
    n = len(atoms)
    d = numpy.zeros((n, n, 3))
    get_d_array(d, q)
    r = numpy.zeros((n, n))
    get_r_array(r, d)

    thresh = numpy.less(r, rt)
    for i in range(len(thresh)):
        thresh[i,i] = False

    mag = numpy.zeros(r.shape)
    mag[thresh] = k * (r[thresh]-rt)
    # print mag
    direction = numpy.zeros(q.shape)
    spring_force_kernel(direction, d, r, mag)
    return direction
