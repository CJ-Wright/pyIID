from __future__ import print_function

from copy import deepcopy as dc

import numpy as np
from ase.io.trajectory import Trajectory
from ase.optimize.optimize import Optimizer
from numpy.random import RandomState

from pyiid.adp import has_adp

__author__ = 'christopher'


def leapfrog(atoms, step, center=True):
    """
    Propagate the dynamics of the system via the leapfrog algorithm one step

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration for the system
    step: float
        The step size for the simulation, the new momentum/velocity is step *
        the force
    center: bool
        If true, center the atoms in the cell after moving them

    Returns
    -------
    ase.Atoms
        The new atomic positions and velocities
    """
    latoms = dc(atoms)

    adps = None
    if has_adp(latoms):
        adps = has_adp(latoms)

    latoms.set_momenta(latoms.get_momenta() + 0.5 * step * latoms.get_forces())
    if adps is not None:
        adps.set_momenta(adps.get_momenta() +
                         0.5 * step * adps.get_forces(latoms))

    latoms.set_positions(latoms.get_positions() +
                         step * latoms.get_velocities())
    if adps is not None:
        adps.set_positions(adps.get_positions() +
                           0.5 * step * adps.get_velocities())

    latoms.set_momenta(latoms.get_momenta() +
                       0.5 * step * latoms.get_forces())

    if adps is not None:
        adps.set_momenta(adps.get_momenta() +
                         0.5 * step * adps.get_forces(latoms))

    if center:
        latoms.center()
    return latoms


class Ensemble(Optimizer):
    def __init__(self, atoms, restart=None, logfile=None, trajectory=None,
                 seed=None,
                 verbose=False):
        Optimizer.__init__(self, atoms, restart, logfile, trajectory=None)
        atoms.get_forces()
        atoms.get_potential_energy()
        if seed is None:
            seed = np.random.randint(0, 2 ** 31)
        self.verbose = verbose
        self.random_state = RandomState(seed)
        self.starting_atoms = dc(atoms)
        self.pe = []
        self.metadata = {'seed': seed}
        self.traj = [dc(atoms)]
        print(self.traj[0].get_momenta())
        if trajectory is not None:
            self.trajectory = Trajectory(trajectory, mode='w')
            self.trajectory.write(self.traj[-1])
            if self.verbose:
                print('Trajectory written', len(self.traj))
        else:
            self.trajectory = None
        if verbose:
            print('trajectory file', self.trajectory)

    def check_eq(self, eq_steps, tol):
        ret = np.cumsum(self.pe, dtype=float)
        ret[eq_steps:] = ret[eq_steps:] - ret[:-eq_steps]
        ret = ret[eq_steps - 1:] / eq_steps
        return np.sum(np.gradient(ret[eq_steps:])) < tol

    def run(self, steps=100000000, eq_steps=None, eq_tol=None, **kwargs):
        self.metadata['planned iterations'] = steps
        i = 0
        while i < steps:
            # Check if we are at equilibrium, if we want that
            if eq_steps is not None:
                if self.check_eq(eq_steps, eq_tol):
                    break
            # Verboseness
            if self.verbose:
                print('iteration number', i)
            try:
                self.step()
                i += 1
            # If we blow up, write the last structure down and exit gracefully
            except KeyboardInterrupt:
                print('Interupted, returning data')
                return self.traj, self.metadata

        if self.trajectory is not None:
            self.trajectory.close()
        return self.traj, self.metadata

    def step(self):
        pass

    def estimate_simulation_duration(self, atoms, iterations):
        pass
