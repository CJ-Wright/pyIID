from __future__ import print_function
from pyiid.calc.calc_1d import Calc1D
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.sim.nuts_hmc import NUTSCanonicalEnsemble
from pyiid.tests import *
from ase.visualize import view
from tempfile import NamedTemporaryFile, mkstemp
from ase.io.trajectory import TrajectoryReader, Trajectory

__author__ = 'christopher'



class TestNUTS:
    test_nuts_data = tuple(product(dc(test_atom_squares), test_calcs, [True, False]))

    def setUp(self):
        self.traj_file = NamedTemporaryFile(delete=False)

    def tearDown(self):
        os.remove(self.traj_file.name)

    def test_nuts_dynamics(self):
        for v in self.test_nuts_data:
            yield self.check_nuts, v

    def check_nuts(self, value):
        """
        Test NUTS simulation

        Parameters
        ----------
        value: list or tuple
            The values to use in the tests
        """
        print(self.traj_file)
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

        if value[2]:
            traj_name = self.traj_file.name
        else:
            traj_name = None
        nuts = NUTSCanonicalEnsemble(ideal_atoms, escape_level=4, verbose=True,
                                     seed=seed, trajectory=traj_name)
        traj, metadata = nuts.run(5)
        print(traj[0].get_momenta())
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

        self.traj_file.close()
        if value[2]:
            assert os.path.exists(self.traj_file.name)
            read_traj = TrajectoryReader(self.traj_file.name)
            print(len(traj), len(read_traj))
            assert len(traj) == len(read_traj)
            for i, (atoms1, atoms2) in enumerate(zip(read_traj, traj)):
                for att in ['get_positions', 'get_potential_energy',
                            'get_forces', 'get_momenta']:
                    print(i, att)
                    assert_allclose(*[getattr(a, att)() for a in [atoms1, atoms2]])
        del traj


class TestASE:
    test_nuts_data = tuple(product(dc(test_atom_squares), test_calcs))

    def setUp(self):
        self.traj_file = NamedTemporaryFile(delete=False)

    def tearDown(self):
        print(self.traj_file.name)
        os.remove(self.traj_file.name)
        print(os.path.exists(self.traj_file.name))

    def test_ase(self):
        for v in self.test_nuts_data:
            yield self.check_ase, v

    def check_ase(self, value):
        """
        Test NUTS simulation

        Parameters
        ----------
        value: list or tuple
            The values to use in the tests
        """
        print(self.traj_file)
        ideal_atoms, _ = value[0]
        write_traj = Trajectory(self.traj_file.name, mode='w')
        traj = []
        for i in range(3):
            write_traj.write(ideal_atoms)
            traj.append(ideal_atoms)

        self.traj_file.close()
        assert os.path.exists(self.traj_file.name)
        print(self.traj_file)
        read_traj = TrajectoryReader(self.traj_file.name)
        print(len(traj), len(read_traj))
        assert len(traj) == len(read_traj)
        del traj


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         # '--nocapture',
                         '-v',
                         '-x'
                         ],
                   exit=False)
