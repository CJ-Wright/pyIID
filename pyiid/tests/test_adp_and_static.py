from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter

__author__ = 'christopher'

# rtol = 4e-4
# atol = 4e-4
rtol = 5e-4
atol = 5e-5


# Actual Tests
def check_method(value):
    # set everything up
    atoms, adp_atoms = value[1]
    exp = value[2]
    assert np.all(atoms.get_positions() == adp_atoms.get_positions())
    scat = ElasticScatter(exp_dict=exp, verbose=True)
    proc1, alg1 = value[3]
    method_string = value[0]

    exp_method = getattr(scat, method_string)
    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = exp_method(atoms)

    # run algorithm 2
    ans2 = exp_method(adp_atoms)

    # test
    # assert np.any(ans1 != ans2)
    print(ans1 - ans2)
    assert np.all(ans1 == ans2)
    # make certain we did not give back the same pointer
    assert ans1 is not ans2
    # assert False

test_data = list(product(
    # tests,
    ['get_fq',
     # 'get_sq', 'get_iq', 'get_pdf', 'get_grad_fq', 'get_grad_pdf'
     ],
    zip(test_atoms, test_atoms),
    # test_adp_atoms,
    test_exp,
    proc_alg_pairs,

))


def test_meta():
    for v in test_data:
        yield check_method, v


if __name__ == '__main__':
    import nose
    print(test_data[0][1:])
    nose.runmodule(argv=[
        '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        # '-x',
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
