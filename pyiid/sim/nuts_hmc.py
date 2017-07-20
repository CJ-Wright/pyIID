from __future__ import print_function
from copy import deepcopy as dc
from ase.units import fs
import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pyiid.sim import leapfrog
from pyiid.sim import Ensemble
from ase.units import kB
from time import time

__author__ = 'christopher'
Emax = 200


def buildtree(input_atoms, u, v, j, e, e0, rs, beta=1):
    """
    Build the tree of samples for NUTS, recursively

    Parameters
    -----------
    input_atoms: ase.Atoms object
        The atoms for the tree
    u: float
        slice parameter, the baseline energy to compare against
    v: -1 or 1
        The direction of the tree leaves, negative means that we simulate
        backwards in time
    j: int
        The tree depth
    e: float
        The stepsize
    e0: float
        Current energy
    rs: numpy.random.RandomState object
        The random state object used to generate the random numbers.  Use of a
        unified random number generator with a known seed should help us to
        generate reproducible simulations
    beta: float
        Thermodynamic beta, a proxy for thermal energy
    Returns
    -------
    Many things
    """
    if j == 0:
        atoms_prime = leapfrog(input_atoms, v * e)
        neg_delta_energy = e0 - atoms_prime.get_total_energy()
        try:
            exp1 = np.exp(neg_delta_energy)
            exp2 = np.exp(Emax + neg_delta_energy)
        except:
            exp1 = 0
            exp2 = 0
        # print exp1, exp2
        # n_prime = int(u <= np.exp(-atoms_prime.get_total_energy()))
        # s_prime = int(u <= np.exp(Emax-atoms_prime.get_total_energy()))
        n_prime = int(u <= exp1)
        s_prime = int(u < exp2)
        return (atoms_prime, atoms_prime, atoms_prime, n_prime, s_prime,
                min(1, np.exp(-atoms_prime.get_total_energy() +
                              input_atoms.get_total_energy())), 1)
    else:
        (neg_atoms, pos_atoms, atoms_prime, n_prime, s_prime, a_prime,
         na_prime) = buildtree(input_atoms, u, v, j - 1, e, e0, rs, beta)
        if s_prime == 1:
            if v == -1:
                (neg_atoms, _, atoms_prime_prime, n_prime_prime, s_prime_prime,
                 app, napp) = buildtree(neg_atoms, u, v, j - 1, e, e0, rs,
                                        beta)
            else:
                (_, pos_atoms, atoms_prime_prime, n_prime_prime, s_prime_prime,
                 app, napp) = buildtree(pos_atoms, u, v, j - 1, e, e0, rs,
                                        beta)

            if rs.uniform() < float(n_prime_prime / (
                    max(n_prime + n_prime_prime, 1))):
                atoms_prime = atoms_prime_prime

            a_prime = a_prime + app
            na_prime = na_prime + napp

            datoms = pos_atoms.positions - neg_atoms.positions
            span = datoms.flatten()
            s_prime = s_prime_prime * (
                span.dot(neg_atoms.get_velocities().flatten()) >= 0) * (
                          span.dot(pos_atoms.get_velocities().flatten()) >= 0)
            n_prime = n_prime + n_prime_prime
        return (neg_atoms, pos_atoms, atoms_prime, n_prime, s_prime, a_prime,
                na_prime)


class NUTSCanonicalEnsemble(Ensemble):
    def __init__(self, atoms, restart=None, logfile=None, trajectory=None,
                 temperature=100, escape_level=13, accept_target=.65,
                 momentum=None,
                 seed=None, verbose=False):
        Ensemble.__init__(self, atoms, restart, logfile, trajectory, seed,
                          verbose)
        self.accept_target = accept_target
        self.temp = temperature
        self.thermal_nrg = self.temp * kB
        self.step_size = self._find_step_size(atoms, self.thermal_nrg)
        self.mu = np.log(10 * self.step_size)
        # self.ebar = 1
        self.sim_hbar = 0
        self.gamma = 0.05
        self.t0 = 10
        # self.k = .75
        self.metadata.update({
            'samples_total': 0,
            'accepted_samples': 0
        })
        self.escape_level = escape_level
        self.m = 0
        self.momentum = momentum
        if self.thermal_nrg:
            if self.verbose:
                print('thermal_nrg', self.thermal_nrg)
            MaxwellBoltzmannDistribution(atoms, temp=self.thermal_nrg,
                                         force_temp=True)
            if self.verbose:
                print('kinetic energy', atoms.get_kinetic_energy())
        elif self.momentum:
            atoms.set_momenta(self.random_state.normal(0, momentum, (
                len(atoms), 3)))
            if self.verbose:
                print('kinetic energy', atoms.get_kinetic_energy())
        else:
            print('Some thermal energy needed')

    def _find_step_size(self, input_atoms, thermal_nrg=None, momentum=None):
        """
        Find a suitable starting step size for the simulation

        Parameters
        -----------
        input_atoms: ase.Atoms object
            The starting atoms for the simulation
        thermal_nrg:
            The thermal energy for the simulation

        Returns
        -------
        float:
            The step size
        """
        atoms = dc(input_atoms)
        step_size = .5
        if thermal_nrg:
            MaxwellBoltzmannDistribution(atoms, temp=thermal_nrg,
                                         force_temp=True)
        elif momentum:
            atoms.set_momenta(self.random_state.normal(0, momentum, (
                len(atoms), 3)))
        else:
            print('Some thermal energy needed')

        atoms_prime = leapfrog(atoms, step_size)

        a = 2 * (np.exp(
            -1 * atoms_prime.get_total_energy() + atoms.get_total_energy()
        ) > 0.5) - 1

        while (np.exp(-1 * atoms_prime.get_total_energy() +
                          atoms.get_total_energy())) ** a > 2 ** -a:
            step_size *= 2 ** a
            print('trying step size', step_size)
            atoms_prime = leapfrog(atoms, step_size)
            if step_size < 1e-7 or step_size > 1e7:
                step_size = 1.
                break
        print('optimal step size', step_size)
        return step_size

    def _find_step_size(self, input_atoms, thermal_nrg=None, momentum=None):
        """
        Find a suitable starting step size for the simulation

        Parameters
        -----------
        input_atoms: ase.Atoms object
            The starting atoms for the simulation
        thermal_nrg:
            The thermal energy for the simulation

        Returns
        -------
        float:
            The step size
        """
        atoms = dc(input_atoms)
        step_size = .5
        if thermal_nrg:
            MaxwellBoltzmannDistribution(atoms, temp=thermal_nrg,
                                         force_temp=True)
        elif momentum:
            atoms.set_momenta(self.random_state.normal(0, 1, (
                len(atoms), 3)) * self.momentum)
        else:
            print('Some thermal energy needed')

        atoms_prime = leapfrog(atoms, step_size)

        a = 2 * (np.exp(
            -1 * atoms_prime.get_total_energy() + atoms.get_total_energy()
        ) > 0.5) - 1

        while (np.exp(-1 * atoms_prime.get_total_energy() +
                          atoms.get_total_energy())) ** a > 2 ** -a:
            step_size *= 2 ** a
            print('trying step size', step_size)
            atoms_prime = leapfrog(atoms, step_size)
            if step_size < 1e-7 or step_size > 1e7:
                step_size = 1.
                break
        print('optimal step size', step_size)
        return step_size

    def step(self):
        atoms = dc(self.traj[-1])
        new_configurations = []
        if self.verbose:
            print('\ttime step size', self.step_size / fs, 'fs')
        # sample r0
        if self.momentum is None:
            MaxwellBoltzmannDistribution(atoms, self.thermal_nrg,
                                         # force_temp=True
                                         )
        else:
            atoms.set_momenta(self.random_state.normal(0, 1, (
                len(atoms), 3)) * self.momentum)
        # re-sample u, note we work in post exponential units:
        # [0, exp(-H(atoms0)] <= exp(-H(atoms1) >>> [0, 1] <= exp(-deltaH)
        u = self.random_state.uniform(0, 1)

        # Note that because we need to calculate the difference between the
        # proposed energy and the current energy we declare it here,
        # preventing the need for multiple calls to the energy function
        e0 = atoms.get_total_energy()

        e = self.step_size
        n, s, j = 1, 1, 0
        neg_atoms = dc(atoms)
        pos_atoms = dc(atoms)
        while s == 1:
            v = self.random_state.choice([-1, 1])
            if v == -1:
                (neg_atoms, _, atoms_prime, n_prime, s_prime, a,
                 na) = buildtree(neg_atoms, u, v, j, e, e0, self.random_state,
                                 1 / self.thermal_nrg)
            else:
                (_, pos_atoms, atoms_prime, n_prime, s_prime, a,
                 na) = buildtree(pos_atoms, u, v, j, e, e0, self.random_state,
                                 1 / self.thermal_nrg)

            if s_prime == 1 and self.random_state.uniform() < min(
                    1, n_prime * 1. / n):
                self.traj += [atoms_prime]
                if self.trajectory is not None:
                    atoms_prime.get_forces()
                    self.trajectory.write(atoms_prime)
                    if self.verbose:
                        print('\t\t\tTrajectory written', len(self.traj))
                if self.verbose:
                    print('\t\t\tNew Potential Energy: {} eV'.format(
                        atoms_prime.get_potential_energy()))
                    print('\t\t\tNew Kinetic Energy: {} eV'.format(
                        atoms_prime.get_kinetic_energy()))
                    print('\t\t\tNew Temperature: {} K'.format(
                        atoms_prime.get_temperature()))
                self.metadata['accepted_samples'] += 1
                if self.verbose:
                    print('accepted configuration at ',
                          atoms_prime.get_potential_energy())
                new_configurations.extend([atoms_prime])
                atoms_prime.get_forces()
                atoms_prime.get_potential_energy()
                self.call_observers()
            n = n + n_prime
            span = pos_atoms.positions - neg_atoms.positions
            span = span.flatten()
            s = s_prime * (
                span.dot(neg_atoms.get_velocities().flatten()) >= 0) * (
                    span.dot(pos_atoms.get_velocities().flatten()) >= 0)
            j += 1
            if self.verbose:
                print('\t \tdepth', j, 'samples', 2 ** j)
            self.metadata['samples_total'] += 2 ** j
            # Prevent run away sampling, EXPERIMENTAL
            # If we have generated s**self.escape_level samples,
            # then we are moving too slowly and should start a new iter
            # hopefully with a larger step size
            if j >= self.escape_level:
                if self.verbose:
                    print('\t \t \tjmax emergency escape at {}'.format(j))
                s = 0
        w = 1. / (self.m + self.t0)
        self.sim_hbar = (1 - w) * self.sim_hbar + w * \
                                                  (self.accept_target - a / na)

        self.step_size = np.exp(self.mu - (self.m ** .5 / self.gamma) *
                                self.sim_hbar)

        self.m += 1
        if len(new_configurations) > 0:
            return new_configurations
        else:
            return None

    def estimate_simulation_duration(self, atoms, iterations):
        t0 = time()
        f = atoms.get_forces() * 2
        tf = time() - t0
        t2 = time()
        e = atoms.get_potential_energy() * 2
        te = time() - t2

        time_one_step = tf * 2 + te
        total_time = iterations * time_one_step * 2 ** self.escape_level
        return total_time
