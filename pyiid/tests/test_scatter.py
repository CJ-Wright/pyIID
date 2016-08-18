from __future__ import print_function
"""
The results of an ElasticScatter experiment should be the same regardless of
the processor/algorithm used to obtain the results.
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
    """
    Check two processor, algorithm pairs against each other for FQ calculation
    :param value:
    :return:
    """
    # set everything up
    method_string = value[0]
    atoms, exp = value[1:3]
    scat = ElasticScatter(exp_dict=exp, verbose=True)

    ans = []
    for proc, alg in value[3]:
        scat.set_processor(proc, alg)
        exp_method = getattr(scat, method_string)
        # If we are going to check an ADP only method,
        # we better have atoms AND ADPS
        if method_string in elastic_scatter_adp_methods \
                and has_adp(atoms) is not None:
            # If we don't have the adps attached skip the test
            raise SkipTest()
        ans.append(exp_method(atoms))
        assert scat.processor == proc
        assert scat.alg == alg

    # test
    if not stats_check(ans[0], ans[1], rtol, atol):
        print(value)
    assert_allclose(ans[0], ans[1], rtol=rtol, atol=atol)

    # make certain we did not give back the same pointer
    assert ans[0] is not ans[1]
    # assert False

test_data = list(product(
    elastic_scatter_methods,
    test_atoms + test_adp_atoms,
    test_exp, comparison_pro_alg_pairs))


def test_meta():
    for v in test_data:
        yield check_method, v


if __name__ == '__main__':
    import nose

    print('number of test cases', len(test_data))
    nose.runmodule(argv=[
        '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        # '-x',
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
