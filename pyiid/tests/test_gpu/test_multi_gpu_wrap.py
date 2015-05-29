__author__ = 'christopher'

import numpy as np
from numpy.testing import assert_allclose

from pyiid.wrappers.cpu_wrap import wrap_fq as serial_fq
from pyiid.wrappers.cpu_wrap import wrap_fq_grad as serial_grad_fq

from pyiid.wrappers.multi_gpu_wrap import wrap_fq as gpu_fq
from pyiid.wrappers.multi_gpu_wrap import wrap_fq_grad as gpu_grad_fq

from ase.atoms import Atoms
from pyiid.wrappers.scatter import wrap_atoms

n = 1500


def test_fq():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    wrap_atoms(atoms, exp_dict)

    sfq_ave = np.zeros(250)
    gfq_ave = np.zeros(250)
    sfq_ave += serial_fq(atoms)
    gfq_ave += gpu_fq(atoms)
    # for m in range(10):
    #     sfq_ave += serial_fq(atoms)
    #     gfq_ave += gpu_fq(atoms)
    # sfq_ave /= m
    # gfq_ave /= m
    assert_allclose(sfq_ave, gfq_ave, rtol=1e-3)

    return

# '''
def test_grad_fq():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    wrap_atoms(atoms, exp_dict)

    sfq = serial_grad_fq(atoms)
    gfq = gpu_grad_fq(atoms)
    assert_allclose(sfq, gfq, rtol=1e-1)

    return
# '''

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)

    '''
    import matplotlib.pyplot as plt
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    wrap_atoms(atoms)

    diff = []
    sfq = serial_fq(atoms)
    for i in range(10):
        gfq = gpu_fq(atoms)
        diff.append(np.abs(sfq - gfq)/(gfq+sfq)/2)
    plt.imshow(diff, aspect='auto', interpolation='None')
    plt.colorbar()

    # plt.plot(np.abs(sfq - gfq)/(gfq+sfq)/2, label='diff')
    plt.show()
    # plt.plot(sfq)
    # plt.plot(gfq)
    # plt.show()

    # sfq = serial_grad_fq(atoms)
    # gfq = gpu_grad_fq(atoms)

    # print np.abs(sfq-gfq) <= (0 + 1e-1*np.abs(gfq))
    # print gfq.shape
    # for i in range(3):
    #     plt.imshow((np.abs(sfq-gfq) <= (0 + 1e-1*np.abs(gfq)))[:, i, :], aspect='auto')
        # plt.imshow((np.abs((sfq-gfq)/(1e-1*np.abs(gfq))))[:, i, :], aspect='auto', interpolation='None')
        # plt.colorbar()
        # plt.show()
    # sfq_ave = sfq[0, 0, :]
    # gfq_ave = gfq[0,0,:]

    # plt.plot(sfq_ave, label='cpu')
    # plt.plot(gfq_ave, label='gpu')
    # plt.plot(np.abs(sfq_ave - gfq_ave)/(gfq_ave+sfq_ave)/2, label='diff')
    # plt.plot(np.abs(sfq_ave - gfq_ave), label='diff')
    # plt.legend()
    # plt.show()'''