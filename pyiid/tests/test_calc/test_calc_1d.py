from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.calc.calc_1d import Calc1D

__author__ = 'christopher'

def check_meta(value):
    value[0](value[1:])


def check_nrg(value):
    """
    Check two processor, algorithm pairs against each other for PDF energy

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    rtol = 4e-6
    atol = 9e-6
    # setup
    atoms1, atoms2 = value[0]
    exp_dict = value[1]
    p, thresh = value[2]
    proc1, alg1 = value[3][0]
    proc2, alg2 = value[3][1]

    scat = ElasticScatter(verbose=True)
    scat.update_experiment(exp_dict)
    scat.set_processor(proc1, alg1)
    if value[4] == 'FQ':
        exp_func = scat.get_fq
        exp_grad = scat.get_grad_fq
    elif value[4] == 'PDF':
        exp_func = scat.get_pdf
        exp_grad = scat.get_grad_pdf
    target_data = exp_func(atoms1)
    calc = Calc1D(target_data=target_data,
                  exp_function=exp_func, exp_grad_function=exp_grad,
                  potential=p)
    atoms2.set_calculator(calc)
    ans1 = atoms2.get_potential_energy()

    scat.set_processor(proc2, alg2)
    calc = Calc1D(target_data=target_data,
                  exp_function=exp_func, exp_grad_function=exp_grad,
                  potential=p)
    atoms2.set_calculator(calc)
    ans2 = atoms2.get_potential_energy()
    stats_check(ans2, ans1, rtol, atol)


def check_forces(value):
    """
    Check two processor, algorithm pairs against each other for PDF forces
    :param value:
    :return:
    """
    # setup
    rtol = 1e-4
    atol = 6e-5
    atoms1, atoms2 = value[0]
    exp_dict = value[1]
    p, thresh = value[2]
    proc1, alg1 = value[3][0]
    proc2, alg2 = value[3][1]

    scat = ElasticScatter(verbose=True)
    scat.update_experiment(exp_dict)
    scat.set_processor(proc1, alg1)
    if value[4] == 'FQ':
        exp_func = scat.get_fq
        exp_grad = scat.get_grad_fq
    elif value[4] == 'PDF':
        exp_func = scat.get_pdf
        exp_grad = scat.get_grad_pdf
    target_data = exp_func(atoms1)
    calc = Calc1D(target_data=target_data,
                  exp_function=exp_func, exp_grad_function=exp_grad,
                  potential=p)

    atoms2.set_calculator(calc)
    ans1 = atoms2.get_forces()

    scat.set_processor(proc2, alg2)
    calc = Calc1D(target_data=target_data,
                  exp_function=exp_func, exp_grad_function=exp_grad,
                  potential=p)
    atoms2.set_calculator(calc)
    ans2 = atoms2.get_forces()
    stats_check(ans2, ans1,
                    rtol=rtol,
                    atol=atol
                    )

tests = [
    check_nrg,
    check_forces
]
test_experiment_types = ['FQ', 'PDF']
test_data = tuple(product(tests,
                          test_double_atoms, test_exp, test_potentials,
                          comparison_pro_alg_pairs, test_experiment_types))

def test_meta():
    for v in test_data:
            yield check_meta, v

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        '-x'
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
