from pyiid.tests import *
from pyiid.sim import leapfrog
from pyiid.calc.spring_calc import Spring
from pyiid.adp import has_adp
import numpy as np
__author__ = 'christopher'

test_data = test_atom_squares


@pytest.mark.parametrize("a", test_data)
def test_meta(a):
    check_leapfrog_no_momentum(a)
    check_leapfrog_momentum(a)
    check_leapfrog_reversibility(a)


def check_leapfrog_no_momentum(value):
    """
    Test leapfrog with null forces

    Parameters
    ----------
    value: tuple
        The values to use in the tests
    """
    atoms = value[0]
    calc = Spring(rt=1, k=100)
    atoms.set_calculator(calc)
    if has_adp(atoms):
        has_adp(atoms).set_calculator(calc)
    atoms2 = leapfrog(atoms, 1, False)
    stats_check(atoms.positions, atoms2.positions)
    if has_adp(atoms):
        stats_check(has_adp(atoms).get_positions(),
                    has_adp(atoms2).get_positions())


def check_leapfrog_momentum(value):
    """
    Test leapfrog with non-trivial momentum

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    atoms = value[0]
    calc = Spring(rt=1, k=100)
    atoms.set_momenta(np.ones((len(atoms), 3)))
    atoms.set_calculator(calc)
    if has_adp(atoms):
        has_adp(atoms).set_calculator(calc)
    atoms2 = leapfrog(atoms, 1, False)
    stats_check(atoms.positions, atoms2.positions - atoms.get_velocities())
    if has_adp(atoms):
        stats_check(has_adp(atoms).get_positions(),
                    has_adp(atoms2).get_positions() -
                    has_adp(atoms2).get_velocities())


def check_leapfrog_reversibility(value):
    """
    Test leapfrog with non-trivial momentum in reverse

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    atoms = value[0]
    calc = Spring(rt=1, k=100)
    atoms.set_momenta(np.ones((len(atoms), 3)))
    atoms.set_calculator(calc)
    if has_adp(atoms):
        has_adp(atoms).set_calculator(calc)
    atoms2 = leapfrog(atoms, 1, False)
    atoms3 = leapfrog(atoms2, -1, False)
    stats_check(atoms.positions, atoms3.positions)
    if has_adp(atoms):
        stats_check(has_adp(atoms).get_positions(),
                    has_adp(atoms2).get_positions())
