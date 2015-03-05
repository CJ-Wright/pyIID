__author__ = 'christopher'

import numpy as np
from numpy.testing import assert_allclose
from copy import deepcopy as dc

from pyiid.wrappers.kernel_wrap import wrap_fq as serial_fq
from pyiid.wrappers.kernel_wrap import wrap_pdf as serial_pdf
from pyiid.wrappers.kernel_wrap import wrap_rw as serial_rw
from pyiid.wrappers.kernel_wrap import wrap_fq_grad as serial_grad_fq
from pyiid.wrappers.kernel_wrap import wrap_grad_rw as serial_grad_rw

from pyiid.wrappers.three_d_gpu_wrap import wrap_fq as gpu_fq
from pyiid.wrappers.three_d_gpu_wrap import wrap_pdf as gpu_pdf
from pyiid.wrappers.three_d_gpu_wrap import wrap_rw as gpu_rw
from pyiid.wrappers.three_d_gpu_wrap import wrap_fq_grad_gpu as gpu_grad_fq
from pyiid.wrappers.three_d_gpu_wrap import wrap_grad_rw as gpu_grad_rw

from ase.atoms import Atoms
from pyiid.wrappers.kernel_wrap import wrap_atoms, grad_pdf

# n = 50
# n = 10

def test_fq():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    wrap_atoms(atoms)

    sfq = serial_fq(atoms)
    gfq = gpu_fq(atoms)
    a = assert_allclose(sfq, gfq, atol=1e-5)

    return


def test_pdf0():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    wrap_atoms(atoms)

    spdf = serial_pdf(atoms)[0]
    gpdf= gpu_pdf(atoms)[0]
    assert_allclose(spdf, gpdf, atol=1e-5)

    return


def test_pdf1():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    wrap_atoms(atoms)

    spdf = serial_pdf(atoms, qmin=2.5)[0]
    gpdf= gpu_pdf(atoms, qmin=2.5)[0]
    assert_allclose(spdf, gpdf, atol=1e-5)

    return


def test_rw0():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)
    wrap_atoms(atoms)
    gobs = serial_pdf(atoms)[0]
    atoms.rattle(.1)

    sfq = serial_rw(atoms, gobs)[0]
    gfq = gpu_rw(atoms, gobs)[0]
    assert_allclose(sfq, gfq, atol=1e-5)

    return


def test_rw1():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)
    wrap_atoms(atoms)
    gobs = serial_pdf(atoms)[0]
    atoms.rattle(.1)

    sfq = serial_rw(atoms, gobs, qmin=2.5)[0]
    gfq = gpu_rw(atoms, gobs, qmin=2.5)[0]
    assert_allclose(sfq, gfq, atol=1e-5)

    return


def test_grad_fq():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)

    wrap_atoms(atoms)

    sfq = serial_grad_fq(atoms)
    gfq = gpu_grad_fq(atoms)
    assert_allclose(sfq, gfq, atol=1e-4)

    return


def test_grad_rw0():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)
    wrap_atoms(atoms)
    gobs = serial_pdf(atoms)[0]
    atoms.rattle(.1)

    sfq = serial_grad_rw(atoms, gobs, qmin=0.0)
    gfq = gpu_grad_rw(atoms, gobs, qmin=0.0)
    assert_allclose(sfq, gfq, atol=1e-3)

    return


def test_grad_rw1():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)
    wrap_atoms(atoms)
    gobs = serial_pdf(atoms)[0]
    atoms.rattle(.1)

    sfq = serial_grad_rw(atoms, gobs, qmin=2.5)
    gfq = gpu_grad_rw(atoms, gobs, qmin=2.5)
    assert_allclose(sfq, gfq, atol=1e-2)

    return


def test_grad_pdf1():
    
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au'+str(n), pos)
    wrap_atoms(atoms)
    gobs = serial_pdf(atoms)[0]
    atoms.rattle(.1)

    qmin = 2.5
    qbin = .1
    rmax = 40
    rstep = .01
    qmin_bin = int(qmin / qbin)

    sfq = serial_grad_fq(atoms)
    gfq = gpu_grad_fq(atoms)

    for tx in range(len(atoms)):
        for tz in range(3):
            sfq[tx, tz, :qmin_bin] = 0.
            gfq[tx, tz, :qmin_bin] = 0.

    assert_allclose(sfq, gfq, atol=1e-3)


    spdf = np.zeros((len(atoms), 3, rmax / rstep))
    gpdf = np.zeros((len(atoms), 3, rmax / rstep))

    grad_pdf(spdf, sfq, rstep, qbin, np.arange(0, rmax, rstep))
    grad_pdf(gpdf, gfq, rstep, qbin, np.arange(0, rmax, rstep))

    assert_allclose(spdf, gpdf, atol=1e-4)

    return

if __name__ == '__main__':
    import nose
    # '''
    for n in range(1, 10000, 100):
        test_fq()
        # nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
        print n
            # '''
    # n = 900
    # test_grad_rw1()
    # nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
