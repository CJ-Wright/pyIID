"""
Test that the forces we calculate from the PES using analytical results are the
same as when we use the ase numerical forces method.
"""
from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.calc.calc_1d import Calc1D

__author__ = 'christopher'

test_dynamics_data = tuple(product(test_atoms, test_calcs))


def test_gen_dynamics():
    for v in test_dynamics_data:
        yield check_n_forces, v

def check_n_forces(value):
    """
    Test numerical vs analytical forces

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    rtol = 1e-6
    atol = 6e-5
    ideal_atoms = value[0]
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    if isinstance(value[1], str):
        s = ElasticScatter(verbose=True)
        target_data = None
        exp_func = None
        exp_grad = None
        if value[1] == 'PDF':
            target_data = s.get_pdf(ideal_atoms)
            exp_func = s.get_pdf
            exp_grad = s.get_grad_pdf

        elif value[1] == 'FQ':
            target_data = s.get_fq(ideal_atoms)
            exp_func = s.get_fq
            exp_grad = s.get_grad_fq
        calc = Calc1D(target_data=target_data,
                      exp_function=exp_func, exp_grad_function=exp_grad,
                      potential='rw', conv=1)
    else:
        calc = value[1]
    ideal_atoms.positions *= 1.02

    ideal_atoms.set_calculator(calc)
    ans1 = ideal_atoms.get_forces()
    ans2 = calc.calculate_numerical_forces(ideal_atoms, d=5e-5)
    stats_check(ans2, ans1,
                rtol=rtol,
                atol=atol
                )


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         # '--nocapture',
                         '-v',
                         # '-x'
                         ],
                   exit=False)
