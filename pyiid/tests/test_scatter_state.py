from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter

__author__ = 'christopher'
# TODO: need to add a bunch of tests here.
def check_meta(value):
    value[0](value[1:])

def check_add_atom(value):
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp, verbose=True)
    scat.set_processor(proc, alg)

    assert scat.check_state(atoms) != []
    # Test a set of different sized ensembles
    ans1 = scat.get_fq(atoms)
    assert scat.check_state(atoms) == []
    # Check that Scatter gave back something
    assert ans1 is not None
    assert np.any(ans1)

    atoms2 = atoms + Atom('Au', [0, 0, 0])
    assert scat.check_state(atoms2) != []
    ans2 = scat.get_fq(atoms2)
    assert scat.check_state(atoms2) == []
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

    assert scat.check_state(atoms) != []
    # Test a set of different sized ensembles
    ans1 = scat.get_fq(atoms)
    assert scat.check_state(atoms) == []
    # Check that Scatter gave back something
    assert ans1 is not None
    assert np.any(ans1)

    atoms2 = dc(atoms)
    del atoms2[np.random.choice(len(atoms2))]
    assert scat.check_state(atoms2) != []
    ans2 = scat.get_fq(atoms2)
    assert scat.check_state(atoms2) == []
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


def test_meta():
    for v in test_data:
            yield check_meta, v
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
