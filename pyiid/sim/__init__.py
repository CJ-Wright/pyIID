from copy import deepcopy as dc
from ase.optimize.optimize import Optimizer
import numpy as np
from numpy.random import RandomState
from ase.md.velocitydistribution import Stationary, ZeroRotation

__author__ = 'christopher'


def leapfrog(atoms, step, center=True):
    """
    Propagate the dynamics of the system via the leapfrog algorithm one step

    Parameters
    -----------
    atoms: ase.Atoms ase.Atoms
        The atomic configuration for the system
    step: float
        The step size for the simulation, the new momentum/velocity is step *
        the force

    Returns
    -------
    ase.Atoms
        The new atomic positions and velocities
    """
    latoms = dc(atoms)

    latoms.set_momenta(latoms.get_momenta() + 0.5 * step * latoms.get_forces())

    latoms.positions += step * latoms.get_velocities()

    latoms.set_momenta(latoms.get_momenta() + 0.5 * step * latoms.get_forces())
    if center:
        latoms.center()
    return latoms


class Ensemble(Optimizer):
    def __init__(self, atoms, restart, logfile, trajectory, seed=None,
                 verbose=False):
        Optimizer.__init__(self, atoms, restart, logfile, trajectory)
        atoms.get_forces()
        atoms.get_potential_energy()
        if seed is None:
            seed = np.random.randint(0, 2 ** 31)
        self.verbose = verbose
        self.random_state = RandomState(seed)
        self.starting_atoms = dc(atoms)
        self.traj = [dc(atoms)]
        self.pe = []

    def check_eq(self, eq_steps, tol):
        ret = np.cumsum(self.pe, dtype=float)
        ret[eq_steps:] = ret[eq_steps:] - ret[:-eq_steps]
        ret = ret[eq_steps - 1:] / eq_steps
        return np.sum(np.gradient(ret[eq_steps:])) < tol

    def run(self, steps=100000000, eq_steps=None, eq_tol=None, **kwargs):
        for i in xrange(steps):
            if eq_steps is not None:
                if self.check_eq(eq_steps, eq_tol):
                    break
            if self.verbose:
                print 'iteration number', i
            self.step()
        return self.traj

    def step(self):
        pass
