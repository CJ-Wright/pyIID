import numpy as np

from pyiid.experiments.elasticscatter.kernels.master_kernel import get_rw, \
    get_chi_sq, get_grad_rw, \
    get_grad_chi_sq

__author__ = 'christopher'


def wrap_rw(gcalc, gobs):
    """
    Generate the Rw value

    Parameters
    -----------
    gcalc: 1darray
        The calculated 1D data
    gobs: 1darray
        The observed 1D data

    Returns
    -------

    rw: float
        The Rw value in percent
    scale: float
        The scale factor between the observed and calculated PDF
    """
    rw, scale = get_rw(gobs, gcalc, weight=None)
    return rw, scale


def wrap_chi_sq(gcalc, gobs):
    """
    Generate the Rw value

    Parameters
    -----------
    gcalc: 1darray
        The calculated 1D data
    gobs: 1darray
        The observed 1D data

    Returns
    -------

    rw: float
        The Rw value in percent
    scale: float
        The scale factor between the observed and calculated PDF
    """
    rw, scale = get_chi_sq(gobs, gcalc)
    return rw, scale


def wrap_grad_rw(grad_gcalc, gcalc, gobs):
    """
    Generate the Rw value gradient

    Parameters
    -----------
    grad_gcalc: ndarray
        The gradient of the 1D data
    gcalc: 1darray
        The calculated 1D data
    gobs: 1darray
        The observed 1D data

    Returns
    -------

    grad_rw: ndarray
        The gradient of the Rw value with respect to the atomic positions,
        in percent
    """
    rw, scale = wrap_rw(gcalc, gobs)
    grad_rw = np.zeros((len(grad_gcalc), 3))
    get_grad_rw(grad_rw, grad_gcalc, gcalc, gobs, rw, scale)
    return grad_rw


def wrap_grad_chi_sq(grad_gcalc, gcalc, gobs):
    """
    Generate the Rw value gradient

    Parameters
    -----------
    grad_gcalc: ndarray
        The gradient of the 1D data
    gcalc: 1darray
        The calculated 1D data
    gobs: 1darray
        The observed 1D data

    Returns
    -------

    grad_chi_sq: ndarray
        The gradient of the chi squared value with respect to the atomic
        positions, in percent
    """
    chi_sq, scale = wrap_chi_sq(gcalc, gobs)
    grad_chi_sq = np.zeros((len(grad_gcalc), 3))
    get_grad_chi_sq(grad_chi_sq, grad_gcalc, gcalc, gobs, scale)
    return grad_chi_sq
