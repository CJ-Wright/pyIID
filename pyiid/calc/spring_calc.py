from ase.calculators.calculator import Calculator
import numpy as np
from pyiid.experiments.elasticscatter.kernels.cpu_nxn import get_d_array, \
    get_r_array

__author__ = 'christopher'


class Spring(Calculator):
    """
    Spring Repulsion Potential Energy Surface
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, k=10, rt=1.5, sp_type='rep', **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        self.nrg_func = spring_nrg
        self.f_func = spring_force
        if sp_type == 'com':
            self.nrg_func = com_spring_nrg
            self.f_func = com_spring_force
        if sp_type == 'att':
            self.nrg_func = att_spring_nrg
            self.f_func = att_spring_force

        self.sp_type = sp_type
        self.k = k
        self.rt = rt

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        """Spring Calculator
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
        energy = self.nrg_func(atoms, self.k, self.rt)
        self.results['energy'] = energy

    def calculate_forces(self, atoms):
        self.results['forces'] = np.zeros((len(atoms), 3))

        forces = self.f_func(atoms, self.k, self.rt)

        self.results['forces'] = forces

    def calculate_voxel_energy(self, atoms, resolution):
        pass


def spring_nrg(atoms, k, rt):
    q = atoms.get_positions().astype(np.float32)
    n = len(atoms)
    d = np.zeros((n, n, 3), dtype=np.float32)
    get_d_array(d, q)
    r = np.zeros((n, n), dtype=np.float32)
    get_r_array(r, d)

    thresh = np.less(r, rt)
    for i in range(len(thresh)):
        thresh[i, i] = False

    mag = np.zeros(r.shape)
    mag[thresh] = k * (r[thresh] - rt)

    energy = np.sum(mag[thresh] / 2. * (r[thresh] - rt))
    return energy


def spring_force(atoms, k, rt):
    q = atoms.get_positions().astype(np.float32)
    n = len(atoms)
    d = np.zeros((n, n, 3), dtype=np.float32)
    get_d_array(d, q)
    r = np.zeros((n, n), dtype=np.float32)
    get_r_array(r, d)

    thresh = np.less(r, rt)

    for i in range(len(thresh)):
        thresh[i, i] = False

    mag = np.zeros(r.shape)
    mag[thresh] = k * (r[thresh] - rt)

    direction = np.zeros((n, n, 3))
    old_settings = np.seterr(all='ignore')
    for tz in range(3):
        direction[thresh, tz] = d[thresh, tz] / r[thresh] * mag[thresh]
    np.seterr(**old_settings)
    direction[np.isnan(direction)] = 0.0
    direction = np.sum(direction, axis=1)
    return direction


def com_spring_nrg(atoms, k, rt):
    com = atoms.get_center_of_mass()
    q = atoms.get_positions().astype(np.float32)
    disp = q - com
    dist = np.sqrt(np.sum(disp ** 2, axis=1))
    thresh = np.greater(dist, rt)
    mag = np.zeros(len(atoms))

    mag[thresh] = k * (dist[thresh] - rt)
    energy = np.sum(mag[thresh] / 2. * (dist[thresh] - rt))
    return energy


def com_spring_force(atoms, k, rt):
    com = atoms.get_center_of_mass()
    q = atoms.get_positions().astype(np.float32)
    disp = q - com
    dist = np.sqrt(np.sum(disp ** 2, axis=1))
    thresh = np.greater(dist, rt)
    mag = np.zeros(len(atoms))

    mag[thresh] = k * (dist[thresh] - rt)

    direction = np.zeros(q.shape)
    old_settings = np.seterr(all='ignore')
    for tz in range(3):
        direction[thresh, tz] = disp[thresh, tz] / dist[thresh] * mag[thresh]
    np.seterr(**old_settings)

    return direction * -1.


def att_spring_nrg(atoms, k, rt):
    q = atoms.get_positions().astype(np.float32)
    n = len(atoms)
    d = np.zeros((n, n, 3), dtype=np.float32)
    get_d_array(d, q)
    r = np.zeros((n, n), dtype=np.float32)
    get_r_array(r, d)

    thresh = np.greater(r, rt)
    for i in range(len(thresh)):
        thresh[i, i] = False

    mag = np.zeros(r.shape)
    mag[thresh] = k * (r[thresh] - rt)

    energy = np.sum(mag[thresh] / 2. * (r[thresh] - rt))
    return energy


def att_spring_force(atoms, k, rt):
    q = atoms.get_positions().astype(np.float32)
    n = len(atoms)
    d = np.zeros((n, n, 3), dtype=np.float32)
    get_d_array(d, q)
    r = np.zeros((n, n), dtype=np.float32)
    get_r_array(r, d)

    thresh = np.greater(r, rt)

    for i in range(len(thresh)):
        thresh[i, i] = False

    mag = np.zeros(r.shape)
    mag[thresh] = k * (r[thresh] - rt)

    direction = np.zeros((n, n, 3))
    old_settings = np.seterr(all='ignore')
    for tz in range(3):
        direction[thresh, tz] = d[thresh, tz] / r[thresh] * mag[thresh]
    np.seterr(**old_settings)
    direction[np.isnan(direction)] = 0.0
    direction = np.sum(direction, axis=1)
    return direction