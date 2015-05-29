__author__ = 'christopher'
from ase.calculators.calculator import Calculator
import numpy as np

from pyiid.wrappers.scatter import Scatter, wrap_atoms
from pyiid.kernels.master_kernel import get_rw, grad_pdf, get_grad_rw, \
    get_chi_sq, get_grad_chi_sq


def wrap_rw(gcalc, gobs):
    """
    Generate the Rw value

    Parameters
    -----------
    :param gcalc:
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
    """
    rw, scale = get_rw(gobs, gcalc, weight=None)
    return rw, scale


def wrap_chi_sq(gcalc, gobs):
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
    rw, scale = get_chi_sq(gobs, gcalc)
    return rw, scale


def wrap_grad_rw(grad_gcalc, gcalc, gobs):
    """
    Generate the Rw value gradient

    Parameters
    -----------
    :param grad_gcalc:
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
    rw, scale = wrap_rw(gcalc, gobs)
    grad_rw = np.zeros((len(grad_gcalc), 3))
    get_grad_rw(grad_rw, grad_gcalc, gcalc, gobs, rw, scale, weight=None)
    return grad_rw


def wrap_grad_chi_sq(grad_gcalc, gcalc, gobs):
    """
    Generate the Rw value gradient

    Parameters
    -----------
    :param gcalc:
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

    grad_chi_sq: float
        The gradient of the Rw value with respect to the atomic positions,
        in percent

    """
    chi_sq, scale = wrap_chi_sq(gcalc, gobs)
    grad_chi_sq = np.zeros((len(grad_gcalc), 3))
    get_grad_chi_sq(grad_chi_sq, grad_gcalc, gcalc, gobs, scale)
    return grad_chi_sq


class PDFCalc(Calculator):
    """
    Class for doing PDF based RW/chi**2 calculations
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, gobs=None, scatter=Scatter(), conv=1.,
                 potential='chi_sq', **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        self.scatter = scatter
        self.rw_to_eV = conv

        if gobs is not None:
            self.gobs = gobs
        else:
            raise NotImplementedError('Need an experimental PDF')

        if potential == 'chi_sq':
            self.potential = wrap_chi_sq
            self.grad = wrap_grad_chi_sq
        elif potential == 'rw':
            self.potential = wrap_rw
            self.grad = wrap_grad_rw
        else:
            raise NotImplementedError('Potential not implemented')

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        """PDF Calculator

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

        Calculator.calculate(self, atoms, properties, system_changes)

        # we shouldn't really recalc if charges or magmos change
        if len(system_changes) > 0:  # something wrong with this way
            if 'energy' in properties:
                self.calculate_energy(self.atoms)

            if 'forces' in properties:
                self.calculate_forces(self.atoms)
        for property in properties:
            if property not in self.results:
                if property is 'energy':
                    self.calculate_energy(self.atoms)

                if property is 'forces':
                    self.calculate_forces(self.atoms)

    def calculate_energy(self, atoms):
        """
        Calculate energy
        :param atoms:
        :return:
        """
        energy, scale = self.potential(self.scatter.get_pdf(atoms), self.gobs)
        self.energy_free = energy * self.rw_to_eV
        self.energy_zero = energy * self.rw_to_eV

        self.results['energy'] = energy * self.rw_to_eV

    def calculate_forces(self, atoms):
        # self.results['forces'] = np.zeros((len(atoms), 3))
        forces = self.grad(self.scatter.get_grad_pdf(atoms),
                           self.scatter.get_pdf(atoms),
        self.gobs) * self.rw_to_eV

        self.results['forces'] = forces


if __name__ == '__main__':
    from ase.atoms import Atoms
    from ase.visualize import view
    from pyiid.wrappers.scatter import wrap_atoms
    from ase.cluster.octahedron import Octahedron
    import numpy as np
    import matplotlib.pyplot as plt

    ideal_atoms = Octahedron('Au', 2)
    ideal_atoms.pbc = False
    wrap_atoms(ideal_atoms)

    scat = Scatter()
    print scat.get_grad_pdf(ideal_atoms)
    gobs = scat.get_pdf(ideal_atoms)

    # calc1 = PDFCalc(gobs=gobs, scatter=scat)
    calc1 = PDFCalc(gobs=gobs, scatter=scat, potential='rw')
    ideal_atoms.set_calculator(calc1)
    ideal_atoms.positions *= 1.5
    print ideal_atoms.get_potential_energy()
    print ideal_atoms.get_forces()