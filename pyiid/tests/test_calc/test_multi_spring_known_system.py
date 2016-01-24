from pyiid.tests import *
import numpy as np
from pyiid.calc.spring_calc import Spring
from pyiid.calc.multi_calc import MultiCalc

__author__ = 'christopher'

test_data = tuple(product(test_atom_squares, test_spring_kwargs))


def test_gen_spring():
    for v in test_data:
        yield check_spring, v


def test_gen_grad_spring():
    for v in test_data:
        yield check_grad_spring, v


def check_spring(value):
    """
    Test two known square systems

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    atoms1, atoms2 = value[0]
    calc = MultiCalc(calc_list=[Spring(**value[1]), Spring(**value[1])])
    atoms2.set_calculator(calc)
    assert atoms2.get_potential_energy() >= 100


def check_grad_spring(value):
    """
    Test two square systems

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    atoms1, atoms2 = value[0]
    calc = MultiCalc(calc_list=[Spring(**value[1]), Spring(**value[1])])
    atoms2.set_calculator(calc)
    forces = atoms2.get_forces()
    com = atoms2.get_center_of_mass()
    for i in range(len(atoms2)):
        dist = atoms2[i].position - com
        # print i, dist, forces[i], np.cross(dist, forces[i])
        stats_check(np.cross(dist, forces[i]), np.zeros(3))


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
