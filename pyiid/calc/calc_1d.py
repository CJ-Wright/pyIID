import numpy as np
from ase.calculators.calculator import Calculator

from pyiid.calc import *

__author__ = 'christopher'


class Calc1D(Calculator):
    """
    Class for doing PDF based RW/chi**2 calculations
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None,
                 target_data=None,
                 exp_function=None, exp_grad_function=None,
                 exp_voxel_function=None, exp_atomwise_function=None,
                 conv=1., potential='rw', **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        # Check calculator kwargs for all the needed info
        if target_data is None or len(target_data.shape) != 1:
            raise NotImplementedError('Need a 1d array target data set')
        if exp_function is None or exp_grad_function is None:
            raise NotImplementedError('Need functions which return the '
                                      'simulated data associated with the '
                                      'experiment and its gradient')
        self.target_data = target_data
        self.exp_function = exp_function
        self.exp_grad_function = exp_grad_function
        self.exp_voxel_function = exp_voxel_function
        self.exp_atomwise_function = exp_atomwise_function
        self.scale = 1
        self.rw_to_eV = conv
        if potential == 'chi_sq':
            self.potential = wrap_chi_sq
            self.grad = wrap_grad_chi_sq
        elif potential == 'rw':
            self.potential = wrap_rw
            self.grad = wrap_grad_rw
            self.voxel = wrap_voxel_rw
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
        Calculate the energy of the atomic configuration
        Parameters
        ----------
        atoms: ASE Atoms object
            The atomic configuration

        Returns
        -------

        """
        energy, scale = self.potential(self.exp_function(atoms),
                                       self.target_data)
        self.scale = scale
        self.results['energy'] = energy * self.rw_to_eV

    def calculate_forces(self, atoms):
        """
        Calculate the forces on the atomic configuration
        Parameters
        ----------
        atoms: ASE Atoms object
            The atomic configuration

        Returns
        -------

        """
        # self.results['forces'] = np.zeros((len(atoms), 3))
        forces = self.grad(self.exp_grad_function(atoms),
                           self.exp_function(atoms),
                           self.target_data) * self.rw_to_eV

        self.results['forces'] = forces

    def calculate_voxel_energy(self, atoms, new_atom, resolution):
        voxels = self.exp_voxel_function(atoms, new_atom, resolution)
        rw, scale = self.voxel(voxels, self.target_data)
        return rw

    def calculate_atomwise_energy(self, atoms):
        atomwise = self.exp_atomwise_function(atoms)
