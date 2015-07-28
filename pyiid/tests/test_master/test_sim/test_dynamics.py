__author__ = 'christopher'
from pyiid.sim.dynamics import classical_dynamics
import numpy as np
from ase.atoms import Atoms

from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc
# from ase.visualize import view


def test_dynamics():
    """
    Test classical dynamics simulation
    :return:
    """
    ideal_atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    s = ElasticScatter()
    s.set_processor('CPU', 'flat')
    gobs = s.get_pdf(ideal_atoms)

    ideal_atoms.positions *= 1.02

    calc = PDFCalc(obs_data=gobs, scatter=s, conv=1, potential='rw')
    ideal_atoms.set_calculator(calc)

    e = 1.
    traj = classical_dynamics(ideal_atoms, e, 5)

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    min_pe = np.min(pe_list)
    del traj
    assert min_pe < .1

def test_reverse_dynamics():
    """
    Test classical dynamics simulation, symplectic dynamics are look the same
    forward as reversed
    """

    ideal_atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    s = ElasticScatter()
    s.set_processor('CPU', 'flat')
    gobs = s.get_pdf(ideal_atoms)

    ideal_atoms.positions *= 1.02

    calc = PDFCalc(obs_data=gobs, scatter=s, conv=1, potential='rw')
    ideal_atoms.set_calculator(calc)

    e = -1.
    traj = classical_dynamics(ideal_atoms, e, 5)
    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    min_pe = np.min(pe_list)
    # view(traj)
    del traj
    assert min_pe < .1

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest', '-v', '--nocapture'], exit=False)