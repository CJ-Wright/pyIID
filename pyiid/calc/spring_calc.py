from __future__ import print_function
from ase.calculators.calculator import Calculator
import numpy as np
from pyiid.experiments.elasticscatter.kernels.cpu_nxn import get_d_array, \
    get_r_array
from six.moves import xrange
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
        # Common parameters to all springs
        self.sp_type = sp_type
        self.k = k
        self.rt = rt

        # Depending on the spring type use different kernels
        self.nrg_func = spring_nrg
        self.f_func = spring_force
        self.v_nrg = voxel_spring_nrg
        self.atomwise_nrg = atomwise_spring_nrg
        if sp_type == 'com':
            self.nrg_func = com_spring_nrg
            self.f_func = com_spring_force
            self.v_nrg = voxel_com_spring_nrg
            self.atomwise_nrg = atomwise_com_spring_nrg
        if sp_type == 'att':
            self.nrg_func = att_spring_nrg
            self.f_func = att_spring_force
            self.v_nrg = voxel_att_spring_nrg
            self.atomwise_nrg = atomwise_att_spring_nrg

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
        return self.v_nrg(atoms, self.k, self.rt, resolution)

    def calculate_atomwise_energy(self, atoms):
        return self.atomwise_nrg(atoms, self.k, self.rt)


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


# TODO:Kernelize me Captain
def voxel_spring_nrg(atoms, k_const, rt, resolution):
    c = np.diagonal(atoms.get_cell())
    voxels = np.zeros(c / resolution)
    q = atoms.get_positions().astype(np.float32)
    im, jm, km = voxels.shape
    for i in xrange(im):
        x = (i + .5) * resolution
        for j in xrange(jm):
            y = (j + .5) * resolution
            for k in xrange(km):
                z = (k + .5) * resolution
                temp2 = 0.0
                for l in xrange(len(q)):
                    temp = np.sqrt(
                        (x - q[l, 0]) ** 2 +
                        (y - q[l, 1]) ** 2 +
                        (z - q[l, 2]) ** 2
                    )
                    if temp < rt:
                        temp2 += .5 * k_const * (temp - rt) ** 2
                voxels[i, j, k] += temp2
    return voxels * 2


def atomwise_spring_nrg(atoms, k, rt):
    q = atoms.get_positions().astype(np.float32)
    n = len(atoms)
    d = np.zeros((n, n, 3), dtype=np.float32)
    get_d_array(d, q)
    r = np.zeros((n, n), dtype=np.float32)
    get_r_array(r, d)

    nrg = .5 * k * (r - rt) ** 2
    print(nrg)
    nrg[np.where(r > rt)] = 0.0
    for i in xrange(len(nrg)):
        nrg[i, i] = 0.0
    return -np.sum(nrg, axis=0) * 2


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


# TODO: Kernelize me Captain
# TODO: This fails because the addition of an atom moves the center of mass
# XXX: We may just want to get rid of this class of spring, it is not useful
def voxel_com_spring_nrg(atoms, k_const, rt, resolution):
    c = np.diagonal(atoms.get_cell())
    voxels = np.zeros(c / resolution)
    com = atoms.get_center_of_mass()
    im, jm, km = voxels.shape
    for i in xrange(im):
        x = (i + .5) * resolution
        for j in xrange(jm):
            y = (j + .5) * resolution
            for k in xrange(km):
                z = (k + .5) * resolution
                temp = np.sqrt(
                    (x - com[0]) ** 2 +
                    (y - com[1]) ** 2 +
                    (z - com[2]) ** 2
                )
                if temp > rt:
                    voxels[i, j, k] += .5 * k_const * (temp - rt) ** 2
    return voxels * 2


def atomwise_com_spring_nrg(atoms, k, rt):
    com = atoms.get_center_of_mass()
    q = atoms.get_positions().astype(np.float32)
    disp = q - com
    dist = np.sqrt(np.sum(disp ** 2, axis=1))

    nrg = .5 * k * (dist - rt) ** 2
    nrg[np.where(dist < rt)] = 0.0
    return np.sum(nrg, axis=0) * 2


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


# TODO:Kernelize me Captain
def voxel_att_spring_nrg(atoms, k_const, rt, resolution):
    c = np.diagonal(atoms.get_cell())
    voxels = np.zeros(c / resolution)
    q = atoms.get_positions().astype(np.float32)
    im, jm, km = voxels.shape
    for i in xrange(im):
        x = (i + .5) * resolution
        for j in xrange(jm):
            y = (j + .5) * resolution
            for k in xrange(km):
                z = (k + .5) * resolution
                temp2 = 0.0
                for l in xrange(len(q)):
                    temp = np.sqrt(
                        (x - q[l, 0]) ** 2 +
                        (y - q[l, 1]) ** 2 +
                        (z - q[l, 2]) ** 2
                    )
                    if temp > rt:
                        temp2 += .5 * k_const * (temp - rt) ** 2
                voxels[i, j, k] += temp2
    return voxels * 2


def atomwise_att_spring_nrg(atoms, k, rt):
    q = atoms.get_positions().astype(np.float32)
    n = len(atoms)
    d = np.zeros((n, n, 3), dtype=np.float32)
    get_d_array(d, q)
    r = np.zeros((n, n), dtype=np.float32)
    get_r_array(r, d)

    nrg = .5 * k * (r - rt) ** 2
    nrg[np.where(r < rt)] = 0.0
    return -np.sum(nrg, axis=0) * 2
