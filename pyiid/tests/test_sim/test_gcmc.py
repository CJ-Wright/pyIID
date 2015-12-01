from pyiid.calc.calc_1d import Calc1D
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.sim.gcmc import GrandCanonicalEnsemble
from pyiid.tests import *
from pyiid.calc.spring_calc import Spring
from pyiid.calc.lennard_jones import LJ

__author__ = 'christopher'

test_nuts_data = list(
    product(dc(test_atom_squares), [Spring(k=10, rt=2.5)], [None, .1],
            [False, True, LJ(param_dict={('Au', 'Au'):[3., .4]})]))

for t in test_nuts_data:
    if t[2] is not None and t[3] is not False:
        test_nuts_data.remove(t)

def test_gcmc():
    for v in test_nuts_data:
        yield check_gcmc, v


def check_gcmc(value):
    """
    Test NUTS simulation

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    ideal_atoms, _ = value[0]
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    calc = value[1]
    del ideal_atoms[-2:]
    n0 = len(ideal_atoms)
    ideal_atoms.set_calculator(calc)

    dyn = GrandCanonicalEnsemble(ideal_atoms, {'Au': 100.0}, temperature=1000,
                                 verbose=True, resolution=value[2], seed=seed,
                                 voxel_weighting=value[3])
    traj, metadata = dyn.run(10)

    pe_list = []
    n = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
        n.append(len(traj))
    print len(traj)
    print 'n max', np.max(n)
    print 'n0', n0
    del traj
    assert np.max(n) > n0


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         # '--nocapture',
                         '-v',
                         # '-x'
                         ],
                   exit=False)
