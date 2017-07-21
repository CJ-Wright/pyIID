from __future__ import print_function
from pyiid.tests import *
from pyiid.sim.dynamics import classical_dynamics
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.calc.calc_1d import Calc1D

__author__ = 'christopher'

test_data = tuple(product(test_atom_squares, test_calcs, [1, -1]))


@pytest.mark.parametrize("a", test_data)
def test_meta(a):
    check_dynamics(a)


def check_dynamics(value):
    """
    Test classical dynamics simulation, symplectic dynamics are look the same
    forward as reversed

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    ideal_atoms, _ = value[0]
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    if isinstance(value[1], str):
        s = ElasticScatter(verbose=True)
        if value[1] == 'PDF':
            target_data = s.get_pdf(ideal_atoms)
            exp_func = s.get_pdf
            exp_grad = s.get_grad_pdf
            adp_exp_grad = s.get_grad_adp_pdf
        elif value[1] == 'FQ':
            target_data = s.get_fq(ideal_atoms)
            exp_func = s.get_fq
            exp_grad = s.get_grad_fq
            adp_exp_grad = s.get_grad_adp_fq
        calc = Calc1D(target_data=target_data,
                      exp_function=exp_func, exp_grad_function=exp_grad,
                      potential='rw', conv=30)
        adp_calc = Calc1D(target_data=target_data,
                          exp_function=exp_func,
                          exp_grad_function=adp_exp_grad,
                          potential='rw', conv=30)
    else:
        calc = value[1]
        adp_calc = value[1]
    ideal_atoms.positions *= 1.02

    ideal_atoms.set_calculator(calc)
    if has_adp(ideal_atoms):
        has_adp(ideal_atoms).set_calculator(adp_calc)
    start_pe = ideal_atoms.get_potential_energy()
    e = value[2]
    traj = classical_dynamics(ideal_atoms, e, 5)

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    min_pe = np.min(pe_list)
    print(min_pe, start_pe, len(traj))
    print(pe_list)
    if start_pe != 0.0:
        assert min_pe < start_pe
