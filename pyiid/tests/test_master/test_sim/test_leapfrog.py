__author__ = 'christopher'
from pyiid.tests import *
from pyiid.sim import leapfrog
from pyiid.tests import setup_atomic_square
from pyiid.calc.spring_calc import Spring
import numpy as np
from numpy.testing import assert_allclose

test_data = test_atom_squares


def test_gen_check_leapfrog_no_momentum():
    for v in test_data:
        yield check_leapfrog_no_momentum, v


def test_gen_check_leapfrog_momentum():
    for v in test_data:
        yield check_leapfrog_momentum, v


def test_gen_check_leapfrog_reversibility():
    for v in test_data:
        yield check_leapfrog_reversibility, v


def check_leapfrog_no_momentum(value):
    """
    Test leapfrog with null forces
    """
    atoms = value[0]
    calc = Spring(rt=1, k=100)
    atoms.set_calculator(calc)
    atoms2 = leapfrog(atoms, 1)
    assert_allclose(atoms.positions, atoms2.positions)


def check_leapfrog_momentum(value):
    """
    Test leapfrog with non-trivial momentum
    """
    atoms = value[0]
    calc = Spring(rt=1, k=100)
    atoms.set_momenta(np.ones((len(atoms), 3)))
    atoms.set_calculator(calc)
    atoms2 = leapfrog(atoms, 1, False)
    assert_allclose(atoms.positions, atoms2.positions - atoms.get_velocities())


def check_leapfrog_reversibility(value):
    """
    Test leapfrog with non-trivial momentum in reverse
    """
    atoms = value[0]
    calc = Spring(rt=1, k=100)
    atoms.set_momenta(np.ones((len(atoms), 3)))
    atoms.set_calculator(calc)
    atoms2 = leapfrog(atoms, 1)
    atoms3 = leapfrog(atoms2, -1)
    assert_allclose(atoms.positions, atoms3.positions)


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
