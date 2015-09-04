from ase.cluster import FaceCenteredCubic
from scipy.misc import comb
from pyiid.utils import *

__author__ = 'christopher'

atoms = FaceCenteredCubic('Au', [[1, 0, 0], [1, 1, 0], [1, 1, 1]], (2, 4, 2))


def test_tag_surface_atoms():
    tag_surface_atoms(atoms)
    assert np.sum(atoms.get_tags()) == 42


def test_get_angle_list():
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


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        # '-v'
        # '-x'
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)