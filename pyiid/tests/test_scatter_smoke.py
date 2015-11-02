from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
from time import time

__author__ = 'christopher'



# ----------------------------------------------------------------------------
def check_meta(value):
    value[0](value[1:])

def check_scatter_fq(value):
    """
    Smoke test for FQ
    """
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)

    # Test a set of different sized ensembles
    ans = scat.get_fq(atoms)

    # Check that Scatter gave back something
    assert ans is not None

    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return


def check_scatter_sq(value):
    """
    Smoke test for SQ
    :param value:
    :return:
    """
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    ans = scat.get_sq(atoms)
    # Check that Scatter gave back something
    assert ans is not None
    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return


def check_scatter_iq(value):
    """
    Smoke test for IQ
    :param value:
    :return:
    """
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    ans = scat.get_iq(atoms)
    # Check that Scatter gave back something
    assert ans is not None
    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return


def check_scatter_pdf(value):
    """
    Smoke test for PDF
    :param value:
    :return:
    """
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    ans = scat.get_pdf(atoms)
    # Check that Scatter gave back something
    assert ans is not None
    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return


def check_scatter_grad_fq(value):
    """
    Smoke test for grad FQ
    :param value:
    :return:
    """
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    assert scat.check_state(atoms) != []
    ans = scat.get_grad_fq(atoms)
    # Check that Scatter gave back something
    assert ans is not None
    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return


def check_scatter_grad_pdf(value):
    """
    Smoke test for grad PDF
    :param value:
    :return:
    """
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    assert scat.check_state(atoms) != []
    ans = scat.get_grad_pdf(atoms)
    # Check that Scatter gave back something
    assert ans is not None
    # Check that all the values are not zero
    assert np.any(ans)
    del atoms, exp, proc, alg, scat, ans
    return


tests = [
    check_scatter_fq,
    check_scatter_sq,
    check_scatter_iq,
    check_scatter_pdf,
    check_scatter_grad_fq,
    check_scatter_grad_pdf
]
test_data = tuple(product(
    tests,
    test_atoms,
    test_exp,
    proc_alg_pairs,
))

def test_meta():
    for v in test_data:
            yield check_meta, v

'''
def test_gen_scatter_smoke_fq():
    for v in test_data:
        yield check_scatter_fq, v

def test_gen_scatter_smoke_grad_fq():
    for v in test_data:
        yield check_scatter_grad_fq, v

def test_gen_scatter_smoke_sq():
    for v in test_data:
        yield check_scatter_sq, v

def test_gen_scatter_smoke_iq():
    for v in test_data:
        yield check_scatter_iq, v

def test_gen_scatter_smoke_pdf():
    for v in test_data:
        yield check_scatter_pdf, v

def test_gen_scatter_smoke_grad_pdf():
    for v in test_data:
        yield check_scatter_grad_pdf, v
'''

if __name__ == '__main__':
    import nose
    print len(test_data)
    print len(test_data) * 6
    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        '-x',
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
