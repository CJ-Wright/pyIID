import numpy as np
from .. import *
from pyiid.experiments.elasticscatter.kernels.master_kernel import \
    get_scatter_array, get_rw, get_chi_sq
__author__ = 'christopher'


def test_get_scatter_array():
    scatter_array = np.loadtxt('pyiid/tests/test_master/c60_scat.txt',
                               dtype=np.float32)
    ksa = np.zeros(scatter_array.shape)
    numbers = np.ones(len(ksa), dtype=np.int) * 6
    qbin = .1
    get_scatter_array(ksa, numbers, qbin)

    stats_check(scatter_array, ksa, rtol=1e-2)


def test_get_rw():
    x = np.arange(0, 2 * np.pi, .1)
    a = np.sin(x)
    b = np.cos(x)
    stats_check(get_rw(a, b)[0], 1)
    return


def test_get_rw2():
    x = np.arange(0, 2 * np.pi, .1)
    a = np.sin(x)
    b = np.sin(x)
    stats_check(0, get_rw(a, b)[0], atol=1e-15)
    return


def test_get_chi_sq():
    x = np.arange(0, 2 * np.pi, .1)
    a = np.sin(x)
    b = np.cos(x)
    stats_check(get_chi_sq(a, b)[0], 63.01399)
    return


def test_get_chi_sq2():
    x = np.arange(0, 2 * np.pi, .1)
    a = np.sin(x)
    b = np.sin(x)
    stats_check(0, get_chi_sq(a, b)[0], atol=1e-15)
    return


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
