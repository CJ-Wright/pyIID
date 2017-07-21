from pyiid.tests import *
import numpy as np
from pyiid.calc.spring_calc import Spring
import pytest

__author__ = 'christopher'

test_data = tuple(product(test_atom_squares, test_spring_kwargs))


@pytest.mark.parametrize("a", test_data)
def test_meta(a):
    check_spring(a)
    check_grad_spring(a)


def check_spring(value):
    """
    Test spring for atomic square

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    atoms1, _ = value[0]
    calc = Spring(**value[1])
    atoms1.set_calculator(calc)
    assert atoms1.get_potential_energy() >= 100


def check_grad_spring(value):
    """
    Test gradient of spring for atomic square

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    atoms1, _ = setup_atomic_square()
    calc = Spring(**value[1])
    atoms1.set_calculator(calc)
    forces = atoms1.get_forces()
    com = atoms1.get_center_of_mass()
    for i in range(len(atoms1)):
        dist = atoms1[i].position - com
        print(i, dist, forces[i], np.cross(dist, forces[i]))
        # make certain the forces are not zero automatically
        assert np.any(forces[i])
        stats_check(np.cross(dist, forces[i]), np.zeros(3))
