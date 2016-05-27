from __future__ import print_function
from pyiid.calc.calc_1d import Calc1D
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.sim.nuts_hmc import NUTSCanonicalEnsemble
from pyiid.tests import *
from ase.visualize import view
from tempfile import NamedTemporaryFile
from ase.io.trajectory import TrajectoryReader
__author__ = 'christopher'

test_nuts_data = tuple(product(dc(test_atom_squares), test_calcs))


def test_nuts_dynamics():
    for v in test_nuts_data:
        yield check_nuts, v


def check_nuts(value):
    """
    Test NUTS simulation

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    traj_file = NamedTemporaryFile()
    ideal_atoms, _ = value[0]
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    s = ElasticScatter(verbose=True)
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

    nuts = NUTSCanonicalEnsemble(ideal_atoms, escape_level=4, verbose=True,
                                 seed=seed, trajectory=traj_file.name)
    traj, metadata = nuts.run(5)

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    min_pe = np.min(pe_list)
    print(len(traj))
    print(min_pe, start_pe)
    if start_pe != 0.0:
        if not min_pe < start_pe:
            view(traj)
        assert min_pe < start_pe
    read_traj = TrajectoryReader(traj_file.name)
    print(len(traj), len(read_traj))
    for atoms1, atoms2 in zip(read_traj, traj):
        assert_allclose(atoms1.positions, atoms2.positions)
    del traj

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         # '--nocapture',
                         '-v',
                         '-x'
                         ],
                   exit=False)
