from pyiid.tests import *
from ase.cluster import FaceCenteredCubic
from pyiid.utils import *
from ase.visualize import view

tf = False
try:
    from scipy.special import comb
except ImportError:
    tf = True
__author__ = 'christopher'


class TestUtils:
    def setUp(self):
        self.atoms = Atoms(
            FaceCenteredCubic('Au', [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                              (2, 4, 2)))

    def test_onion_tag(self):
        onion_tag(self.atoms)
        assert self.atoms.get_tags()[35] == 2
        assert len(self.atoms[[atom.index for atom in self.atoms if atom.tag == 0]]) == 42
    
    
    def test_tag_surface_atoms(self):
        tag_surface_atoms(self.atoms)
        assert np.sum(self.atoms.get_tags()) == 42
    
    
    @known_fail_if(tf)
    def test_get_angle_list(self):
        from scipy.special import comb
        angles = get_angle_list(self.atoms, 3.6)
        coord = get_coord_list(self.atoms, 3.6)
        assert len(angles) == int(sum([comb(n, 2) for n in coord]))
    
    
    def test_get_coord_list(self):
        coord = get_coord_list(self.atoms, 3.6)
        assert len(coord) == len(self.atoms)
    
    
    def test_get_bond_dist_list(self):
        bonds = get_bond_dist_list(self.atoms, 3.6)
        coord = get_coord_list(self.atoms, 3.6)
        assert len(bonds) == sum(coord)


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        '-x'
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
