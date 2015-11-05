from copy import deepcopy as dc
from time import time
import numpy as np
from ase.atom import Atom
from ase.units import *
from pyiid.sim import Ensemble

__author__ = 'christopher'


# TODO: can we work with PDF overlaps? What is the relationship between PDF
# overlap and the PES?

def add_atom(atoms, chem_potentials, beta,
             resolution=None,
             voxel_energy=False):
    # Current energy of the system
    e0 = atoms.get_potential_energy()

    # make the proposed system
    atoms_prime = dc(atoms)

    # make new atom
    new_symbol = np.random.choice(atoms.get_chemical_symbols())
    new_atom = Atom(new_symbol, [0, 0, 0])

    # If we don't specify a resolution use floating point uniform distribution
    if resolution is None:
        new_position = np.zeros(3)
        for i in xrange(3):
            new_position[i] = np.random.uniform(0, np.max(
                np.diagonal(atoms.get_cell())[i]))
    else:
        if voxel_energy:
            # Get the voxel energy using exponential weighting
            voxel_nrg = atoms.calc.calculate_voxel_energy(atoms, resolution)
            prob = np.exp(-beta * (voxel_nrg - e0))
            prob -= np.min(prob)
            prob /= np.sum(prob)
            qvr = np.random.choice(prob.size, p=prob.ravel())
            qv = np.asarray(np.unravel_index(qvr, prob.shape))
        else:
            # Use voxels for resolution
            c = np.int32(np.ceil(np.diagonal(atoms.get_cell()) / resolution))
            qvr = np.random.choice(np.product(c))
            qv = np.asarray(np.unravel_index(qvr, c))
        new_position = (qv + .5) * resolution

    # Set the atomic position based on sampling procedure
    new_atom.position = np.asarray(new_position)

    # append new atom to system
    atoms_prime.append(new_atom)

    # get new energy
    delta_energy = atoms_prime.get_potential_energy() - e0
    # get chemical potential
    mu = chem_potentials[new_symbol]
    # return atoms_prime
    # print atoms_prime.get_potential_energy(), atoms.get_potential_energy()
    # print delta_energy, np.exp(min([0, -1. * beta * delta_energy + beta * mu]))
    # calculate acceptance
    if np.random.random() < np.exp(
            min([0, -1. * beta * delta_energy + beta * mu])):
        return atoms_prime
    else:
        return None


def del_atom(atoms, chem_potentials, beta, atomwise=False):
    if len(atoms) == 1:
        return None
    # atoms.center()
    # make the proposed system
    atoms_prime = dc(atoms)
    e0 = atoms.get_potential_energy()
    if atomwise is False:
        del_atom_index = np.random.choice(range(len(atoms)))
    else:
        # Get atomic energies, weight using exponentials
        atomwise_nrg = atoms.calc.calculate_atomwise_energy(atoms)
        prob = np.exp(-1 * beta * (atomwise_nrg - e0))
        prob /= np.sum(prob)
        del_atom_index = np.random.choice(range(len(atoms)), p=prob)
    del_symbol = atoms_prime[del_atom_index].symbol

    # append new atom to system
    del atoms_prime[del_atom_index]

    # get new energy
    delta_energy = atoms_prime.get_potential_energy() - e0
    # get chemical potential
    # print delta_energy
    mu = chem_potentials[del_symbol]
    # calculate acceptance
    if np.random.random() < np.exp(
            min([0, -1. * beta * delta_energy - beta * mu
                 ])) and not np.isnan(delta_energy):
        return atoms_prime
    else:
        return None


class GrandCanonicalEnsemble(Ensemble):
    """
    Grand Canonical Monte Carlo simulation
    >>> from ase.cluster.octahedron import Octahedron
    >>> from pyiid.calc.spring_calc import Spring
    >>> atoms = Octahedron('Au', 3)
    >>> atoms.rattle(.1)
    >>> atoms.center()
    >>> calc = Spring(rt=2.5, k=200)
    >>> atoms.set_calculator(calc)
    >>> gc = GrandCanonicalEnsemble(atoms, {'Au': 0.0}, 3000)
    >>> traj = gc.run(10000)
    """

    def __init__(self, atoms, chemical_potentials, temperature=100,
                 restart=None, logfile=None, trajectory=None, seed=None,
                 verbose=False, scatter=None, target_pdf=None, resolution=None,
                 voxel_energy=False, atomwise=False):
        Ensemble.__init__(self, atoms, restart, logfile, trajectory, seed,
                          verbose)
        self.resolution = resolution
        self.beta = 1. / (temperature * kB)
        self.chem_pot = chemical_potentials
        self.metadata = {'rejected_additions': 0, 'accepted_removals': 0,
                         'accepted_additions': 0, 'rejected_removals': 0}
        self.scatter = scatter
        self.target_pdf = target_pdf
        self.voxel_energy = voxel_energy
        self.atomwise = atomwise

    def step(self):
        if self.random_state.uniform() >= .5:
            mv = 'remove'
            new_atoms = del_atom(self.traj[-1], self.chem_pot, self.beta,
                                 self.atomwise)
        else:
            mv = 'add'
            new_atoms = add_atom(self.traj[-1], self.chem_pot, self.beta,
                                 self.resolution, self.voxel_energy)
        if new_atoms is not None:
            if self.verbose:
                print '\t' + mv + ' atom accepted', len(new_atoms)

            if mv == 'add':
                self.metadata['accepted_additions'] += 1
            elif mv == 'remove':
                self.metadata['accepted_removals'] += 1

            self.traj.append(new_atoms)
            return [new_atoms]
        else:
            if self.verbose:
                print '\t' + mv + ' atom rejected', len(self.traj[-1])

            if mv == 'add':
                self.metadata['rejected_additions'] += 1
            elif mv == 'remove':
                self.metadata['rejected_removals'] += 1

            return None

    def estimate_simulation_duration(self, atoms, iterations):
        t2 = time()
        e = atoms.get_potential_energy()
        te = time() - t2

        total_time = 0.
        for i in xrange(iterations):
            total_time += te
        return total_time
