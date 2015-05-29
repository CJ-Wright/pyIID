__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms
from pyiid.wrappers.scatter import wrap_atoms


def setup_atoms(n, exp_dict=None):
    q = np.random.random((n, 3)) * 10
    atoms = Atoms('Au' + str(int(n)), q)
    wrap_atoms(atoms, exp_dict)
    return atoms


def generate_experiment():
    exp_dict = {}
    exp_keys = ['qmin', 'qmax', 'qbin', 'rmin', 'rmax', 'rstep']
    exp_ranges = [(0, 1.5), (19., 25.), (.8, .12), (0., 2.5), (30., 50.),
                  (.005, .015)]
    for n, k in enumerate(exp_keys):
        exp_dict[k] = np.random.uniform(exp_ranges[n][0], exp_ranges[n][1])
    return exp_dict