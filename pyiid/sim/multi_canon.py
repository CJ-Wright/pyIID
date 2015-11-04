from __future__ import division
import numpy as np
from copy import deepcopy as dc
import importlib
from pyiid.sim import Ensemble
import math
__author__ = 'christopher'


class MultiCanonicalSimulation(Ensemble):
    def __init__(self, atoms, ensemble_list,
                 restart=None, logfile=None, trajectory=None,
                 ensemble_prob=None, seed=None, verbose=False):
        Ensemble.__init__(self, atoms, restart, logfile, trajectory, seed=seed,
                          verbose=verbose)
        self.ensembles = []

        # build the ensemble and init
        for ensemble in ensemble_list:
            self.ensembles.append(ensemble)

        if ensemble_prob is not None:
            self.prob = ensemble_prob
        else:
            self.prob = None
        self.metadata = {}
        for ensemble in self.ensembles:
            self.metadata[ensemble.__class__.__name__] = ensemble.metadata

    def step(self):
        j = np.random.choice(np.arange(len(self.ensembles)), p=self.prob)
        if self.verbose:
            print self.ensembles[j].__class__.__name__
        new_configs = self.ensembles[j].step()
        if new_configs:
            self.traj.extend(new_configs)
            for i in range(len(self.ensembles)):
                if i != j:
                    self.ensembles[i].traj.extend(new_configs)

    def estimate_simulation_duration(self, atoms, iterations):
        total_time = 0.
        for ensemble, p in zip(self.ensembles, self.prob):
            total_time += ensemble.estimate_simulation_duration(
                atoms, int(math.ceil(iterations * p)))
        return total_time