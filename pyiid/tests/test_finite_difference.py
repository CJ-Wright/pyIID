"""
This module tests the analytical F(Q) gradient using a small perturbation in
atom position.
"""
from __future__ import print_function
from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
__author__ = 'christopher'

dq = 5e-5


def finite_difference_grad(atoms, exp_dict):
    s = ElasticScatter(exp_dict, verbose=True)
    start_fq = s.get_fq(atoms)
    finite_difference_grad_fq = np.zeros((len(atoms), 3, len(start_fq)))
    for i in range(len(atoms)):
        for w in range(3):
            atoms2 = dc(atoms)
            atoms2[i].position[w] += dq
            fq2 = s.get_fq(atoms2)
            finite_difference_grad_fq[i, w, :] = (fq2 - start_fq) / dq
    return finite_difference_grad_fq
