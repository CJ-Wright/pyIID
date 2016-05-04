"""
Smoke test all the major ElasticScatter methods
"""
from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
import inspect

__author__ = 'christopher'


# ----------------------------------------------------------------------------
def check_method(value):
    """
    Smoke test for FQ

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    method_string = value[0]
    atoms, exp = value[1:3]
    proc, alg = value[3]
    noise = value[4]

    scat = ElasticScatter(exp_dict=exp, verbose=True)
    scat.set_processor(proc, alg)
    exp_method = getattr(scat, method_string)

    # Test a set of different sized ensembles
    if 'noise' in inspect.getargspec(exp_method)[0]:
        ans = exp_method(atoms, noise)
    elif 'atoms' in inspect.getargspec(exp_method)[0]:
        ans = exp_method(atoms)
    else:
        ans = exp_method()

    # Check that Scatter gave back something
    assert ans is not None

    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return


def check_scatter_consistancy(value):
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp, verbose=True)
    scat.set_processor(proc, alg)
    ans = scat.get_pdf(atoms)
    ans1 = scat.get_fq(atoms)
    anss = scat.get_scatter_vector()
    print ans1.shape, anss.shape, scat.exp['qmin'], scat.exp['qmax'], \
        scat.exp['qbin']
    print int(np.ceil(scat.exp['qmax'] / scat.exp['qbin'])) - int(
        np.ceil(scat.exp['qmin'] / scat.exp['qbin']))
    print atoms.get_array('F(Q) scatter').shape
    print (scat.exp['qmin'] - scat.exp['qmax']) / scat.exp['qbin']
    assert ans1.shape == anss.shape
    ans2 = scat.get_sq(atoms)
    assert ans2.shape == anss.shape
    ans3 = scat.get_iq(atoms)
    assert ans3.shape == anss.shape


test_data = tuple(product(
    elastic_scatter_methods + ['get_scatter_vector', 'get_r'],
    test_atoms + test_adp_atoms,
    test_exp,
    proc_alg_pairs,
    [None, rs.randint(10, 100) * .05]
))


def test_meta():
    for v in test_data:
        yield check_method, v


if __name__ == '__main__':
    import nose

    print('number of test cases', len(test_data))
    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        '-x',
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
