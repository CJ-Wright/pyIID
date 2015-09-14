from pyiid.tests import *
from pyiid.sim.dynamics import classical_dynamics
from pyiid.experiments.elasticscatter import ElasticScatter
import pyiid.calc.pdfcalc
from pyiid.calc.fqcalc import FQCalc
from pyiid.calc.calc_1d import Calc1D

__author__ = 'christopher'

test_dynamics_data = tuple(product(test_atom_squares, test_calcs, [1, -1]))


def test_gen_dynamics():
    for v in test_dynamics_data:
        yield check_dynamics, v


def check_dynamics(value):
    """
    Test classical dynamics simulation, symplectic dynamics are look the same
    forward as reversed
    """
    ideal_atoms, _ = value[0]
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    s = ElasticScatter()
    if value[1] == 'PDF':
        target_data = s.get_pdf(ideal_atoms)
        exp_func = s.get_pdf
        exp_grad = s.get_grad_pdf
        calc = Calc1D(target_data=target_data,
                      exp_function=exp_func, exp_grad_function=exp_grad,
                      potential='rw', conv=30)
    elif value[1] == 'FQ':
        target_data = s.get_pdf(ideal_atoms)
        exp_func = s.get_pdf
        exp_grad = s.get_grad_pdf
        calc = Calc1D(target_data=target_data,
                      exp_function=exp_func, exp_grad_function=exp_grad,
                      potential='rw', conv=30)
    else:
        calc = value[1]
    ideal_atoms.positions *= 1.02

    ideal_atoms.set_calculator(calc)
    start_pe = ideal_atoms.get_potential_energy()
    e = value[2]
    traj = classical_dynamics(ideal_atoms, e, 5)

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    min_pe = np.min(pe_list)
    print min_pe, start_pe, len(traj)
    print pe_list
    assert min_pe < start_pe


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         # '--nocapture',
                         '-v',
                         '-x'
                         ],
                   exit=False)
