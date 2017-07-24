from __future__ import print_function

from pyiid.sim.gcmc import GrandCanonicalEnsemble
from pyiid.tests import *

__author__ = 'christopher'

test_data = tuple(
    product(dc(test_atom_squares), [Spring(k=10, rt=2.5)], [None, .1]))


@pytest.mark.parametrize("a", test_data)
def test_nuts_dynamics(a):
    check_nuts(a)


def check_nuts(value):
    """
    Test NUTS simulation

    Parameters
    ----------
    value: tuple
        The values to use in the tests
    """
    ideal_atoms, _ = value[0]
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    calc = value[1]
    del ideal_atoms[-2:]
    n0 = len(ideal_atoms)
    ideal_atoms.set_calculator(calc)

    dyn = GrandCanonicalEnsemble(ideal_atoms, {'Au': 100.0}, temperature=1000,
                                 verbose=True, resolution=value[2], seed=seed)
    traj, metadata = dyn.run(10)

    pe_list = []
    n = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
        n.append(len(traj))
    print(len(traj))
    print('n max', np.max(n))
    print('n0', n0)
    del traj
    assert np.max(n) > n0
