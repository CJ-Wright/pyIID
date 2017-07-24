from __future__ import print_function
from pyiid.tests import *
import numpy as np
from pyiid.calc.spring_calc import Spring
from ase import Atom
from unittest import SkipTest
from builtins import range
import pytest

__author__ = 'christopher'


def check_meta(value):
    value[0](value[1:])


def check_voxel_nrg(value):
    if value[1]['sp_type'] == 'com':
        raise SkipTest(
            'the center of mass moves on addition/removal of atoms, '
            'making this very complex. COM will be depreciated soon')
    atoms = dc(value[0])
    resolution = 1.
    atoms.center(resolution)
    calc = Spring(**value[1])
    atoms.set_calculator(calc)
    e0 = atoms.get_potential_energy()
    voxel_nrg = atoms.calc.calculate_voxel_energy(atoms, resolution)

    voxel_nrg2 = np.zeros(voxel_nrg.shape)
    im, jm, km = voxel_nrg2.shape
    for i in range(im):
        x = (i + .5) * resolution
        for j in range(jm):
            y = (j + .5) * resolution
            for k in range(km):
                z = (k + .5) * resolution
                atoms2 = dc(atoms)
                atoms2 += Atom('Au', (x, y, z))
                voxel_nrg2[i, j, k] += atoms2.get_potential_energy() - e0
    stats_check(voxel_nrg, voxel_nrg2, rtol=2e-7)
    assert_allclose(voxel_nrg, voxel_nrg2, rtol=2e-7)


def check_atomwise_nrg(value):
    if value[1]['sp_type'] == 'com':
        raise SkipTest(
            'the center of mass moves on addition/removal of atoms, '
            'making this very complex. COM will be depreciated soon')
    atoms = dc(value[0])
    resolution = 1.
    atoms.center(resolution)
    calc = Spring(**value[1])
    atoms.set_calculator(calc)
    e0 = atoms.get_potential_energy()
    print(e0, e0 / 2.)
    voxel_nrg = atoms.calc.calculate_atomwise_energy(atoms)

    nrg2 = np.zeros(len(atoms))
    for atom in atoms:
        atoms2 = dc(atoms)
        del atoms2[atom.index]
        nrg2[atom.index] += atoms2.get_potential_energy() - e0
    stats_check(voxel_nrg, nrg2)
    assert_allclose(voxel_nrg, nrg2)


tests = [
    check_voxel_nrg,
    check_atomwise_nrg
]

test_data = tuple(
    product(tests,
            # test_atoms,
            [test_atom_squares[0][0]],
            test_spring_kwargs))


@pytest.mark.parametrize("a", test_data)
def test_meta(a):
    check_meta(a)
