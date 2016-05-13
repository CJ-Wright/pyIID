"""
The results of atoms with ADPs should be different from those without ADPs.
"""
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
    method_string = value[0]
    # Make certain we are using an ADP and static system
    assert np.all(value[1][0].get_positions() == value[1][0].get_positions())

    exp = value[2]
    scat = ElasticScatter(exp_dict=exp, verbose=True)
    proc, alg = value[3]

    ans = []
    scat.set_processor(proc, alg)
    for atoms in value[1]:
        if method_string in elastic_scatter_adp_methods \
                and has_adp(atoms) is not None:
            # If we don't have the adps attached skip the test
            raise SkipTest()
        exp_method = getattr(scat, method_string)
        ans.append(np.nan_to_num(exp_method(atoms)))
        assert scat.processor == proc
        assert scat.alg == alg

    # test
    assert np.any(ans[0])
    assert np.any(ans[1])
    print(ans[0] - ans[1])
    assert np.any(ans[0] != ans[1])
    # make certain we did not give back the same pointer
    assert ans[0] is not ans[1]
    # assert False

test_data = list(product(
    elastic_scatter_methods,
    zip(test_atoms, test_adp_atoms),
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
