import numpy as np

from pyiid.experiments.elasticscatter.kernels.master_kernel import get_rw, \
    get_chi_sq, get_grad_rw, \
    get_grad_chi_sq
from ase.calculators.calculator import Calculator

__author__ = 'christopher'


def wrap_rw(gcalc, gobs):
    """
    Generate the Rw value

    Parameters
    -----------
    gcalc: np.ndarray
        The calculated 1D data
    gobs: np.ndarray
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
    gcalc: np.ndarray
        The calculated 1D data
    gobs: np.ndarray
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
    gcalc: np.ndarray
        The calculated 1D data
    gobs: np.ndarray
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
    gcalc: np.ndarray
        The calculated 1D data
    gobs: np.ndarray
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


class Null(Calculator):
    """
    Returns zero forces or energy
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)

    def calculate(self, atoms=None, properties=None,
                  system_changes=None):
        """PDF Calculator
        Parameters
        ----------
        atoms: Atoms object
            Contains positions, unit-cell, ...
        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces'
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these five: 'positions', 'numbers', 'cell',
            'pbc', 'charges' and 'magmoms'.
        """

        if system_changes is None:
            system_changes = ['positions', 'numbers', 'cell',
                              'pbc', 'charges', 'magmoms']
        if properties is None:
            properties = ['energy']
        Calculator.calculate(self, atoms, properties, system_changes)

        # we shouldn't really recalc if charges or magmos change
        if len(system_changes) > 0:  # something wrong with this way
            if 'energy' in properties:
                self.calculate_energy(self.atoms)

            if 'forces' in properties:
                self.calculate_forces(self.atoms)
        for calc_property in properties:
            if calc_property not in self.results:
                if calc_property is 'energy':
                    self.calculate_energy(self.atoms)

                if calc_property is 'forces':
                    self.calculate_forces(self.atoms)

    def calculate_energy(self, atoms):
        """
        Calculate energy
        :param atoms:
        :return:
        """
        self.results['energy'] = 0.0

    def calculate_forces(self, atoms):
        forces = np.zeros((len(atoms), 3))
        self.results['forces'] = forces
