from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from copy import deepcopy as dc
from time import time
import numpy as np
from ase.atom import Atom
from ase.units import *
from pyiid.sim import Ensemble
from six.moves import xrange
__author__ = 'christopher'


def add_atom(atoms, chem_potentials, beta, random_state, resolution=None):
    """
    Perform a GCMC atomic addition

    Parameters
    ----------
    atoms: ase.Atoms object
        The atomic configuration
    chem_potentials: dict
        A dictionary of {"Chemical Symbol": mu} where mu is a float denoting
        the chemical potential
    beta: float
        The thermodynamic beta
    random_state: np.random.RandomState object
        The random state to be used
    resolution: float or ndarray, optional
        If used denote the resolution for the voxels

    Returns
    -------
    atoms or None:
        If the new configuration is accepted then the new atomic configuration
        is returned, else None

    """
    # make the proposed system
    atoms_prime = dc(atoms)

    # make new atom
    new_symbol = np.random.choice(chem_potentials.keys())
    e0 = atoms.get_potential_energy()
    if resolution is None:
        new_position = np.random.uniform(0, np.max(atoms.get_cell(), 0))
    else:
        c = np.int32(np.ceil(np.diagonal(atoms.get_cell()) / resolution))
        qvr = np.random.choice(np.product(c))
        qv = np.asarray(np.unravel_index(qvr, c))
        new_position = (qv + random_state.uniform(0, 1, 3)) * resolution
    new_atom = Atom(new_symbol, np.asarray(new_position))

    # append new atom to system
    atoms_prime.append(new_atom)

    # get new energy
    delta_energy = atoms_prime.get_potential_energy() - e0
    # get chemical potential
    mu = chem_potentials[new_symbol]
    # calculate acceptance
    if np.random.random() < np.exp(
            min([0, -1. * beta * delta_energy + beta * mu])):
        return atoms_prime
    else:
        return None


def del_atom(atoms, chem_potentials, beta, random_state):
    """

    Parameters
    ----------
    atoms: ase.Atoms object
        The atomic configuration
    chem_potentials: dict
        A dictionary of {"Chemical Symbol": mu} where mu is a float denoting
        the chemical potential
    beta: float
        The thermodynamic beta
    random_state: np.random.RandomState object
        The random state to be used

    Returns
    -------
    atoms or None:
        If the new configuration is accepted then the new atomic configuration
        is returned, else None


    """
    if len(atoms) <= 1:
        return None
    print(len(atoms))
    # make the proposed system
    atoms_prime = dc(atoms)
    e0 = atoms.get_potential_energy()
    del_atom_index = random_state.choice(range(len(atoms)))
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
                 verbose=False, resolution=None):
        Ensemble.__init__(self, atoms, restart, logfile, trajectory, seed,
                          verbose)
        self.beta = 1. / (temperature * kB)
        self.chem_pot = chemical_potentials
        self.metadata = {'rejected_additions': 0, 'accepted_removals': 0,
                         'accepted_additions': 0, 'rejected_removals': 0}
        self.resolution = resolution

    def step(self):
        if self.random_state.uniform() >= .5:
            mv = 'remove'
            new_atoms = del_atom(self.traj[-1], self.chem_pot, self.beta,
                                 self.random_state
                                 )
        else:
            mv = 'add'
            new_atoms = add_atom(self.traj[-1], self.chem_pot, self.beta,
                                 self.random_state, resolution=self.resolution
                                 )
        if new_atoms is not None:
            if self.verbose:
                print('\t' + mv + ' atom accepted', len(new_atoms))

            if mv == 'add':
                self.metadata['accepted_additions'] += 1
            elif mv == 'remove':
                self.metadata['accepted_removals'] += 1

            self.traj.append(new_atoms)
            return [new_atoms]
        else:
            if self.verbose:
                print('\t' + mv + ' atom rejected', len(self.traj[-1]))

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
