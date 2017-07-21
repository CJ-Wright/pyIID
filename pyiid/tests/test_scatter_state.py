"""
Test that the elastic scatter system updates it's state when the number of
atoms change.
"""
# TODO: We should also check that the state updates if whe transmute elements
from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
from ase import Atom

__author__ = 'christopher'


def check_meta(value):
    value[0](value[1:])


def check_add_atom(value):
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp, verbose=True)
    scat.set_processor(proc, alg)

    assert scat._check_wrap_atoms_state(atoms) == False
    # Test a set of different sized ensembles
    ans1 = scat.get_fq(atoms)
    assert scat._check_wrap_atoms_state(atoms) == True
    # Check that Scatter gave back something
    assert ans1 is not None
    assert np.any(ans1)

    atoms2 = atoms + Atom('Au', [0, 0, 0])
    assert scat._check_wrap_atoms_state(atoms2) == False
    ans2 = scat.get_fq(atoms2)
    assert scat._check_wrap_atoms_state(atoms2) == True
    # Check that Scatter gave back something
    assert ans2 is not None
    assert np.any(ans2)

    assert not np.allclose(ans1, ans2)
    # make certain we did not give back the same pointer
    assert ans1 is not ans2
    # Check that all the values are not zero
    del atoms, exp, proc, alg, scat, ans1
    return


def check_del_atom(value):
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp, verbose=True)
    scat.set_processor(proc, alg)

    assert scat._check_wrap_atoms_state(atoms) == False
    # Test a set of different sized ensembles
    ans1 = scat.get_fq(atoms)
    assert scat._check_wrap_atoms_state(atoms) == True
    # Check that Scatter gave back something
    assert ans1 is not None
    assert np.any(ans1)

    atoms2 = dc(atoms)
    del atoms2[np.random.choice(len(atoms2))]
    assert scat._check_wrap_atoms_state(atoms2) == False
    ans2 = scat.get_fq(atoms2)
    assert scat._check_wrap_atoms_state(atoms2) == True
    # Check that Scatter gave back something
    assert ans2 is not None
    assert np.any(ans2)

    assert not np.allclose(ans1, ans2)
    # make certain we did not give back the same pointer
    assert ans1 is not ans2
    # Check that all the values are not zero
    del atoms, exp, proc, alg, scat, ans1
    return


tests = [
    check_add_atom,
    check_del_atom
]
test_data = list(product(
    tests,
    test_atoms,
    test_exp,
    proc_alg_pairs,
))

dels = []
for i, f in enumerate(test_data):
    if len(f[1]) > 200:
        dels.append(i)
dels.reverse()
for d in dels:
    del test_data[d]


@pytest.mark.parametrize("a", test_data)
def test_meta(a):
    check_meta(a)


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        '-x',
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
