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


if __name__ == '__main__':
    rt = 1e-5
    at = 2e-2
    import matplotlib.pyplot as plt

    # atoms = setup_atomic_square()[0]
    # atoms.adps = ADP(atoms, adps=np.random.random((len(atoms), 3)))
    atoms = Atoms('Au2', [[0, 0, 0], [3, 0, 0]])
    exp = None
    s = ElasticScatter(exp)
    # s.set_processor('CPU', 'nxn')
    a = finite_difference_grad(atoms, exp)
    b = s.get_grad_fq(atoms)
    print(a[0, 0] / b[0, 0])
    plt.plot(a[0, 0, :], label='fd')
    plt.plot(b[0, 0, :], label='analytical')
    plt.legend(loc='best')
    plt.show()
    # stats_check(a, b, rt, at)
    # assert_allclose(a, b, rtol=rt, atol=at)
