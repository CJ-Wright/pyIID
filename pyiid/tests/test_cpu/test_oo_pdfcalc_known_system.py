__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms
from numpy.testing import assert_allclose

from pyiid.wrappers.scatter import Scatter, wrap_atoms
from pyiid.tests import generate_experiment
from pyiid.testing.decorators import known_fail_if
from pyiid.calc.oo_pdfcalc import PDFCalc


def setup_atomic_configs():
    atoms1 = Atoms('Au4', [[0,0,0], [3,0,0], [0,3,0], [3,3,0]])
    wrap_atoms(atoms1)
    atoms2 = atoms1.copy()
    scale = .75
    atoms2.positions *= scale
    return atoms1, atoms2, scale


def test_rw():
    atoms1, atoms2, scale = setup_atomic_configs()
    scat = Scatter()
    gobs = scat.get_pdf(atoms1)
    calc = PDFCalc(gobs=gobs, scatter=scat, potential='rw')
    atoms2.set_calculator(calc)
    rw = atoms2.get_potential_energy()
    print rw
    assert rw >= .9


def test_chi_sq():
    atoms1, atoms2, scale = setup_atomic_configs()
    scat = Scatter()
    gobs = scat.get_pdf(atoms1)
    calc = PDFCalc(gobs=gobs, scatter=scat, potential='chi_sq')
    atoms2.set_calculator(calc)
    chi_sq = atoms2.get_potential_energy()
    print chi_sq
    assert chi_sq >= 1000
    # assert False


def test_grad_rw():
    atoms1, atoms2, scale = setup_atomic_configs()
    scat = Scatter()
    gobs = scat.get_pdf(atoms1)
    calc = PDFCalc(gobs=gobs, scatter=scat, potential='rw')
    atoms2.set_calculator(calc)
    forces = atoms2.get_forces()
    com = atoms2.get_center_of_mass()
    for i in range(len(atoms2)):
        dist = atoms2[i].position - com
        print dist, forces[i], np.cross(dist, forces[i])
        assert_allclose(np.cross(dist, forces[i]), np.zeros(3))


def test_grad_chi_sq():
    atoms1, atoms2, scale = setup_atomic_configs()
    scat = Scatter()
    gobs = scat.get_pdf(atoms1)
    calc = PDFCalc(gobs=gobs, scatter=scat, potential='chi_sq')
    atoms2.set_calculator(calc)
    forces = atoms2.get_forces()
    com = atoms2.get_center_of_mass()
    for i in range(len(atoms2)):
        dist = atoms2[i].position - com
        print dist, forces[i], np.cross(dist, forces[i])
        assert_allclose(np.cross(dist, forces[i]), np.zeros(3))

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)

    '''import matplotlib.pyplot as plt
    from pyiid.calc.oo_pdfcalc import wrap_rw
    from ase.visualize import view

    atoms1, atoms2, scale = setup_atomic_configs()
    scat = Scatter()
    gobs = scat.get_pdf(atoms1)
    gcalc = scat.get_pdf(atoms2)
    rw, scale = wrap_rw(gcalc, gobs)
    print rw, scale
    calc = calc = PDFCalc(gobs=gobs, scatter=scat, potential='rw')
    atoms2.set_calculator(calc)
    print atoms2.get_forces().shape
    print atoms2.get_forces()
    plt.plot(gobs)
    plt.plot(gcalc*scale)
    # plt.show()'''