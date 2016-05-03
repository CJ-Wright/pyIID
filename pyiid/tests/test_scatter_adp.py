from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter

__author__ = 'christopher'

# rtol = 4e-4
# atol = 4e-4
rtol = 5e-4
atol = 5e-5


# Actual Tests
def check_meta(value):
    value[0](value[1:])


def check_scatter_fq(value):
    """
    Check two processor, algorithm pairs against each other for FQ calculation
    :param value:
    :return:
    """
    # set everything up
    atoms, exp = value[:2]
    scat = ElasticScatter(exp_dict=exp, verbose=True)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_fq(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_fq(atoms)

    # test
    if not stats_check(ans1, ans2, rtol, atol):
        print value
    assert_allclose(ans1, ans2, rtol=rtol, atol=atol)
    # make certain we did not give back the same pointer
    assert ans1 is not ans2
    # assert False


def check_scatter_grad_fq(value):
    """
    Check two processor, algorithm pairs against each other for gradient FQ
    calculation
    :param value:
    :return:
    """
    # set everything up
    atoms, exp = value[:2]
    scat = ElasticScatter(exp_dict=exp, verbose=True)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_grad_fq(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_grad_fq(atoms)

    # test
    if not stats_check(ans1, ans2, rtol, atol):
        print value
    assert_allclose(ans1, ans2, rtol=rtol, atol=atol)
    # make certain we did not give back the same pointer
    assert ans1 is not ans2


def check_scatter_sq(value):
    """
    Check two processor, algorithm pairs against each other for SQ calculation
    :param value:
    :return:
    """
    # set everything up
    atoms, exp = value[:2]
    scat = ElasticScatter(exp_dict=exp, verbose=True)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_sq(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_sq(atoms)

    # test
    stats_check(ans1, ans2, rtol, atol)
    assert_allclose(ans1, ans2, rtol=rtol, atol=atol)
    # make certain we did not give back the same pointer
    assert ans1 is not ans2


def check_scatter_iq(value):
    """
    Check two processor, algorithm pairs against each other for IQ calculation
    :param value:
    :return:
    """
    # set everything up
    atoms, exp = value[:2]
    scat = ElasticScatter(exp_dict=exp, verbose=True)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_iq(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_iq(atoms)

    # test
    stats_check(ans1, ans2, rtol, atol)
    assert_allclose(ans1, ans2, rtol=rtol, atol=atol)
    # make certain we did not give back the same pointer
    assert ans1 is not ans2


def check_scatter_pdf(value):
    """
    Check two processor, algorithm pairs against each other for PDF calculation
    :param value:
    :return:
    """
    # set everything up
    atoms, exp = value[:2]
    scat = ElasticScatter(exp_dict=exp, verbose=True)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_pdf(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_pdf(atoms)

    # test
    stats_check(ans1, ans2, rtol, atol)
    assert_allclose(ans1, ans2, rtol=rtol, atol=atol)
    # make certain we did not give back the same pointer
    assert ans1 is not ans2


def check_scatter_grad_pdf(value):
    """
    Check two processor, algorithm pairs against each other for gradient PDF
    calculation
    :param value:
    :return:
    """
    # set everything up
    atoms, exp = value[:2]
    scat = ElasticScatter(exp_dict=exp, verbose=True)
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]

    # run algorithm 1
    scat.set_processor(proc1, alg1)
    ans1 = scat.get_grad_pdf(atoms)

    # run algorithm 2
    scat.set_processor(proc2, alg2)
    ans2 = scat.get_grad_pdf(atoms)

    # test
    stats_check(ans1, ans2, rtol, atol)
    assert_allclose(ans1, ans2, rtol=rtol, atol=atol)
    # make certain we did not give back the same pointer
    assert ans1 is not ans2


tests = [
    check_scatter_fq,
    # check_scatter_sq,
    # check_scatter_iq,
    check_scatter_pdf,
    check_scatter_grad_fq,
    check_scatter_grad_pdf
]

test_data = list(product(
    tests,
    test_adp_atoms, test_exp, comparison_pro_alg_pairs))


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
        # '-x',
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
