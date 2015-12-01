import numpy as np
from ase.calculators.calculator import Calculator
from pyiid.experiments.elasticscatter.kernels.cpu_flat import *

__author__ = 'christopher'


class LJ(Calculator):
    """
    Class for doing PDF based RW/chi**2 calculations
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None,
                 param_dict=None,
                 **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        self.param_dict = param_dict
        # Check calculator kwargs for all the needed info

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
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
        a = lj_energy(atoms, self.param_dict)
        self.results['energy'] = a
        return a

    def calculate_forces(self, atoms):
        # self.results['forces'] = lj_forces(atoms, self.param_dict)
        self.results['forces'] = None

    def calculate_voxel_energy(self, atoms, new_atom, resolution):
        if type(resolution) == float:
            resolution = np.ones(3, np.float32) * resolution
        elif not isinstance(resolution, np.ndarray):
            raise NotImplementedError
        return lj_voxel_energy(atoms, new_atom, self.param_dict, resolution)

    def calculate_atomwise_energy(self, atoms):
        return None


def lj_energy(atoms, param_dict):
    q = atoms.get_positions().astype(np.float32)
    n = len(atoms)
    k_max = int(n * (n - 1) / 2.)
    k_cov = i4(0)

    d = np.zeros((k_max, 3), np.float32)
    get_d_array(d, q, k_cov)

    r = np.zeros(k_max, np.float32)
    get_r_array(r, d)

    rm = np.zeros(k_max)
    epsilon = np.zeros(k_max)
    symbols = atoms.get_chemical_symbols()
    for k in xrange(k_max):
        i, j = k_to_ij(k)
        ei = symbols[i]
        ej = symbols[j]
        for key in param_dict.keys():
            if ei in key and ej in key:
                rm[k] = param_dict[key][0]
                epsilon[k] = param_dict[key][1]
                break
    c6 = (rm / r) ** 6
    c12 = c6 ** 2
    lj = epsilon * (c12 - 2 * c6)
    return np.sum(lj)

def lj_voxel_energy(atoms, new_atom, param_dict, resolution):
    v = np.int32(np.ceil(np.diagonal(atoms.get_cell()) / resolution))
    q = atoms.get_positions().astype(np.float32)
    n = len(atoms)

    resolution = np.float32(resolution)
    r = np.zeros((np.product(v), n), np.float32)

    get_voxel_distances(r, q, resolution, v, i4(0))
    rm = np.zeros((np.product(v), n), np.float32)
    epsilon = np.zeros((np.product(v), n), np.float32)

    symbols = atoms.get_chemical_symbols()
    ei = new_atom.symbol

    for j in xrange(n):
        ej = symbols[j]
        for key in param_dict.keys():
            if ei in key and ej in key:
                rm[:, j] = param_dict[key][0]
                epsilon[:, j] = param_dict[key][1]
                break
    c6 = (rm / r) ** 6
    del rm, r
    c12 = c6 ** 2
    lj = epsilon * (c12 - 2 * c6)
    lj = np.sum(lj, axis=-1)
    lj = np.reshape(lj, tuple(v))
    return lj