__author__ = 'christopher'
from pyiid.tests import *
from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.fqcalc import FQCalc

test_data = tuple(product(test_double_atoms, test_exp, test_potentials,
                          comparison_pro_alg_pairs))


def test_nrg():
    for v in test_data:
        yield check_nrg, v


def test_forces():
    for v in test_data:
        yield check_forces, v


def check_nrg(value):
    """
    Check two processor, algorithm pairs against each other for PDF energy
    :param value:
    :return:
    """
    # setup
    atoms1, atoms2 = value[0]
    scat = ElasticScatter()
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]
    scat.update_experiment(exp_dict=value[1])
    scat.set_processor(proc1, alg1)
    p, thresh = value[2]
    gobs = scat.get_fq(atoms1)

    calc = FQCalc(obs_data=gobs, scatter=scat, potential=p)
    atoms2.set_calculator(calc)
    ans1 = atoms2.get_potential_energy()

    scat.set_processor(proc2, alg2)
    calc = FQCalc(obs_data=gobs, scatter=scat, potential=p)
    atoms2.set_calculator(calc)
    ans2 = atoms2.get_potential_energy()
    print stats_check(ans1, ans2)
    assert_allclose(ans2, ans1,
                    # rtol=5e-4,
                    atol=1e-3)


def check_forces(value):
    """
    Check two processor, algorithm pairs against each other for PDF forces
    :param value:
    :return:
    """
    # setup
    atoms1, atoms2 = value[0]
    scat = ElasticScatter()
    proc1, alg1 = value[-1][0]
    proc2, alg2 = value[-1][1]
    scat.update_experiment(exp_dict=value[1])
    scat.set_processor(proc1, alg1)
    p, thresh = value[2]
    gobs = scat.get_fq(atoms1)

    calc = FQCalc(obs_data=gobs, scatter=scat, potential=p)
    atoms2.set_calculator(calc)
    ans1 = atoms2.get_forces()

    scat.set_processor(proc2, alg2)
    calc = FQCalc(obs_data=gobs, scatter=scat, potential=p)
    atoms2.set_calculator(calc)
    ans2 = atoms2.get_forces()
    print stats_check(ans1, ans2)
    if p == 'chi_sq':
        assert_allclose(ans2, ans1,
                        rtol=5e-3,
                        atol=1e-5
                        )
    else:
        assert_allclose(ans2, ans1,
                        rtol=5e-4,
                        atol=1e-7
                        )

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        # '-v'
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
