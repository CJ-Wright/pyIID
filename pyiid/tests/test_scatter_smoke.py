from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
from time import time
from ase import Atom

__author__ = 'christopher'


# ----------------------------------------------------------------------------
def check_meta(value):
    value[0](value[1:])


def check_scatter_fq(value):
    """
    Smoke test for FQ

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    atoms, exp = value[0:2]
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp, verbose=True)
    scat.set_processor(proc, alg)

    assert scat.check_state(atoms) != []
    # Test a set of different sized ensembles
    ans = scat.get_fq(atoms)
    assert scat.check_state(atoms) == []

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

    scat = ElasticScatter(exp_dict=exp, verbose=True)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    assert scat.check_state(atoms) != []
    ans = scat.get_sq(atoms)
    assert scat.check_state(atoms) == []
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

    scat = ElasticScatter(exp_dict=exp, verbose=True)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    assert scat.check_state(atoms) != []
    ans = scat.get_iq(atoms)
    assert scat.check_state(atoms) == []
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

    scat = ElasticScatter(exp_dict=exp, verbose=True)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    assert scat.check_state(atoms) != []
    ans = scat.get_pdf(atoms)
    assert scat.check_state(atoms) == []
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

    scat = ElasticScatter(exp_dict=exp, verbose=True)
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

    scat = ElasticScatter(exp_dict=exp, verbose=True)
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


def check_scatter_fq_voxels(value):
    atoms, exp = dc(value[0:2])
    proc, alg = value[-1]

    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor(proc, alg)
    # Test a set of different sized ensembles
    res = 1.
    atoms.center(res)
    ans = scat.get_fq_voxels(atoms, Atom('Au', [0, 0, 0]), res)
    # Check that Scatter gave back something
    assert ans is not None
    # Check that all the values are not zero
    assert np.any(ans)
    ans2 = np.zeros(ans.shape)
    _, im, jm, km = ans.shape
    for i in xrange(im):
        x = (i + .5) * res
        for j in xrange(jm):
            y = (j + .5) * res
            for k in xrange(km):
                z = (k + .5) * res
                atoms2 = dc(atoms)
                atoms2 += Atom('Au', (x, y, z))
                ans2[:, i, j, k] = scat.get_fq(atoms2)
    assert np.any(ans2)
    stats_check(ans, ans2,
                rtol=3e-6,
                atol=3e-6
                )
    assert_allclose(ans, ans2,
                    rtol=3e-6,
                    atol=3e-6
                    )
    del atoms, exp, proc, alg, scat, ans
    return


tests = [
    check_scatter_fq,
    check_scatter_sq,
    check_scatter_iq,
    check_scatter_pdf,
    check_scatter_grad_fq,
    check_scatter_grad_pdf,
    check_scatter_fq_voxels,
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

if __name__ == '__main__':
    import nose

    print len(test_data)
    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        '-x',
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
