from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter, wrap_atoms
from pyiid.experiments.elasticscatter.kernels import (antisymmetric_reshape,
                                                      symmetric_reshape)
from pyiid.experiments.elasticscatter.kernels.cpu_nxn import \
    (get_d_array as nxn_d,
     get_r_array as nxn_r,
     get_normalization_array as nxn_norm,
     get_omega as nxn_omega,
     get_fq_inplace as nxn_fq)
from pyiid.experiments.elasticscatter.kernels.cpu_flat import \
    (get_d_array as k_d,
     get_r_array as k_r,
     get_normalization_array as k_norm,
     get_omega as k_omega,
     get_fq_inplace as k_fq)


def check_meta(value):
    value[0](value[1:])


def start(value):
    atoms, exp = value[:2]
    wrap_atoms(atoms, exp)
    q = atoms.get_positions().astype(np.float32)
    if value[2] == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')

    n, qmax_bin = scatter_array.shape
    k_max = n * (n - 1) / 2.
    return q, scatter_array, n, qmax_bin, k_max, 0


def d_comparison(value):
    input = start(value)
    q, scatter_array, n, qmax_bin, k_max, k_cov = input
    d1 = np.zeros((n, n, 3), np.float32)
    nxn_d(d1, q)
    d2 = np.zeros((k_max, 3), np.float32)
    k_d(d2, q, k_cov)
    stats_check(d1, antisymmetric_reshape(d2))
    assert_allclose(d1, antisymmetric_reshape(d2))
    print np.max(np.abs(d1 - antisymmetric_reshape(d2)))
    return d1, d2, input


def r_comparison(value):
    d1, d2, input = d_comparison(value)
    q, scatter_array, n, qmax_bin, k_max, k_cov = input
    r1 = np.zeros((n, n), np.float32)
    nxn_r(r1, d1)
    r2 = np.zeros(k_max, np.float32)
    k_r(r2, d2)
    stats_check(r1, symmetric_reshape(r2))
    assert_allclose(r1, symmetric_reshape(r2))
    print np.max(np.abs(r1 - symmetric_reshape(r2)))
    return r1, r2, input


def norm_comparison(value):
    input = start(value)
    q, scatter_array, n, qmax_bin, k_max, k_cov = input
    norm1 = np.zeros((n, n, qmax_bin), np.float32)
    nxn_norm(norm1, scatter_array)
    norm2 = np.zeros((k_max, qmax_bin), np.float32)
    k_norm(norm2, scatter_array, k_cov)
    stats_check(norm1, symmetric_reshape(norm2))
    assert_allclose(norm1, symmetric_reshape(norm2))
    print np.max(np.abs(norm1 - symmetric_reshape(norm2)))
    return norm1, norm2, input


def omega_comparison(value):
    s = ElasticScatter(value[1])
    if value[-1] == 'fq':
        qbin = s.exp['qbin']
    else:
        qbin = s.pdf_qbin
    r1, r2, input = r_comparison(value)
    norm1, norm2, input = norm_comparison(value)
    q, scatter_array, n, qmax_bin, k_max, k_cov = input
    omega1 = np.zeros((n, n, qmax_bin), np.float32)
    nxn_omega(omega1, r1, qbin)
    omega2 = np.zeros((k_max, qmax_bin), np.float32)
    k_omega(omega2, r2, qbin)
    stats_check(omega1, symmetric_reshape(omega2))
    assert_allclose(omega1, symmetric_reshape(omega2))
    print np.max(np.abs(omega1 - symmetric_reshape(omega2)))
    return omega1, omega2, input


tests = [
    # d_comparison, r_comparison, norm_comparison,
    omega_comparison
]

test_data = list(product(
    tests,
    test_atoms,
    test_exp,
    ['fq', 'pdf']))


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
