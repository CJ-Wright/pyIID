from __future__ import print_function
from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.calc.calc_1d import Calc1D
import pytest

__author__ = 'christopher'


def check_meta(value):
    value[0](value[1:])


def check_nrg(value):
    """
    Check for PDF energy against known value
    :param value:
    :return:
    """
    # setup
    atoms1, atoms2 = value[0]
    exp_dict = value[1]
    p, thresh = value[2]
    proc1, alg1 = value[3]

    scat = ElasticScatter(verbose=True)
    scat.update_experiment(exp_dict)
    scat.set_processor(proc1, alg1)
    if value[4] == 'FQ':
        exp_func = scat.get_fq
        exp_grad = scat.get_grad_fq
    elif value[4] == 'PDF':
        exp_func = scat.get_pdf
        exp_grad = scat.get_grad_pdf
    else:
        exp_func = None
        exp_grad = None

    target_data = exp_func(atoms1)
    calc = Calc1D(target_data=target_data,
                  exp_function=exp_func, exp_grad_function=exp_grad,
                  potential=p)
    atoms2.set_calculator(calc)

    ans = atoms2.get_potential_energy()
    assert ans >= thresh
    del atoms1, atoms2, proc1, alg1, p, thresh, scat, target_data, calc, ans


def check_forces(value):
    """
    Check for PDF forces against known value
    :param value:
    :return:
    """
    # setup
    atoms1, atoms2 = value[0]
    exp_dict = value[1]
    p, thresh = value[2]
    proc1, alg1 = value[3]

    scat = ElasticScatter(verbose=True)
    scat.update_experiment(exp_dict)
    scat.set_processor(proc1, alg1)

    if value[4] == 'FQ':
        exp_func = scat.get_fq
        exp_grad = scat.get_grad_fq
    elif value[4] == 'PDF':
        exp_func = scat.get_pdf
        exp_grad = scat.get_grad_pdf
    else:
        exp_func = None
        exp_grad = None

    target_data = exp_func(atoms1)
    calc = Calc1D(target_data=target_data,
                  exp_function=exp_func, exp_grad_function=exp_grad,
                  potential=p)

    atoms2.set_calculator(calc)
    # print atoms2.get_potential_energy()
    forces = atoms2.get_forces()
    print(forces)
    com = atoms2.get_center_of_mass()
    for i in range(len(atoms2)):
        dist = atoms2[i].position - com
        stats_check(np.cross(dist, forces[i]), np.zeros(3), atol=1e-7)
        del dist
    del atoms1, atoms2, proc1, alg1, p, thresh, scat, target_data, calc, \
        forces, com


tests = [
    check_nrg,
    check_forces
]
test_experiment_types = ['FQ', 'PDF']

test_data = tuple(
    product(tests,
            test_atom_squares, test_exp, test_potentials, proc_alg_pairs,
            test_experiment_types))


@pytest.mark.parametrize("a", test_data)
def test_meta(a):
    check_meta(a)
