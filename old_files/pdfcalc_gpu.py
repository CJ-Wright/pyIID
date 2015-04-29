__author__ = 'christopher'
from ase.calculators.calculator import Calculator
import numpy as np

from old_files.three_d_gpu_wrap import wrap_rw, wrap_grad_rw


class PDFCalc(Calculator):
    """
    Class for doing PDF based RW calculations
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, gobs=None, qmin=0.0, qmax=25.0, qbin=None,
                 rmin=0.0, rmax=40.0, rbin=.01, conv=1., **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        self.qmin = qmin
        self.qmax = qmax
        if qbin is None:
            self.qbin = np.pi / (rmax + 6 * 2 * np.pi / qmax)
        else:
            self.qbin = qbin
        self.rmin = rmin
        self.rmax = rmax
        self.rmin = rmin
        self.rbin = rbin
        self.rw_to_eV = conv
        if gobs is not None:
            self.gobs = gobs
        else:
            raise NotImplementedError('Need an experimental PDF')

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
        energy, scale, gcalc, fq = wrap_rw(atoms, self.gobs, self.qmax,
                                           self.qmin, self.qbin, self.rmax,
                                           self.rbin)
        self.energy_free = energy * self.rw_to_eV
        self.energy_zero = energy * self.rw_to_eV
        self.results['energy'] = energy * self.rw_to_eV

    def calculate_forces(self, atoms):
        self.results['forces'] = np.zeros((len(atoms), 3))
        forces = wrap_grad_rw(atoms, self.gobs, self.qmax, self.qmin, self.qbin,
                              self.rmax, self.rbin) * self.rw_to_eV
        self.results['forces'] = forces