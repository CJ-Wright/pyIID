"""
Test pyIID's utilities
"""
from pyiid.tests import *
from ase.cluster import FaceCenteredCubic
from pyiid.utils import *
import pytest

__author__ = 'christopher'

atoms = FaceCenteredCubic('Au', [[1, 0, 0], [1, 1, 0], [1, 1, 1]], (2, 4, 2))


def test_tag_surface_atoms():
    tag_surface_atoms(atoms)
    assert np.sum(atoms.get_tags()) == 42


@pytest.mark.xfail(not srfit, reason='Need installed srfit for this stest')
# @known_fail_if(tf)
def test_get_angle_list():
    from scipy.special import comb
    angles = get_angle_list(atoms, 3.6)
    coord = get_coord_list(atoms, 3.6)
    assert len(angles) == int(sum([comb(n, 2) for n in coord]))


def test_get_coord_list():
    coord = get_coord_list(atoms, 3.6)
    assert len(coord) == len(atoms)


def test_get_bond_dist_list():
    bonds = get_bond_dist_list(atoms, 3.6)
    coord = get_coord_list(atoms, 3.6)
    assert len(bonds) == sum(coord)
