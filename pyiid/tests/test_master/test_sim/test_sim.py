from pyiid.sim.nuts_hmc import nuts

from pyiid.tests import *
from pyiid.sim.dynamics import classical_dynamics
import numpy as np

from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.calc.fqcalc import FQCalc
__author__ = 'christopher'


test_data = tuple(product(test_atom_squares, test_calcs, [1, -1]))

def test_gen_dynamics():
    for v in test_data:
        yield check_dynamics, v
        yield check_nuts, v

def check_dynamics(value):
    """
    Test classical dynamics simulation, symplectic dynamics are look the same
    forward as reversed
    """
    ideal_atoms, _ = value[0]
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    if value[1] == 'PDF':
        s = ElasticScatter()
        gobs = s.get_pdf(ideal_atoms)
        calc = PDFCalc(obs_data=gobs, scatter=s, conv=30, potential='rw')
    elif value[1] == 'FQ':
        s = ElasticScatter()
        gobs = s.get_fq(ideal_atoms)
        calc = FQCalc(obs_data=gobs, scatter=s, conv=30, potential='rw')
    else:
        calc = value[1]
    ideal_atoms.positions *= 1.02

    ideal_atoms.set_calculator(calc)
    start_pe = ideal_atoms.get_potential_energy()
    e = value[2]
    traj = classical_dynamics(ideal_atoms, e, 10)

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    min_pe = np.min(pe_list)
    print min_pe, start_pe, len(traj)
    del traj
    assert min_pe < start_pe


def check_nuts(value):
    """
    Test NUTS simulation
    """
    ideal_atoms, _ = value[0]
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    if value[1] == 'PDF':
        s = ElasticScatter()
        gobs = s.get_pdf(ideal_atoms)
        calc = PDFCalc(obs_data=gobs, scatter=s, conv=30, potential='rw')
    elif value[1] == 'FQ':
        s = ElasticScatter()
        gobs = s.get_fq(ideal_atoms)
        calc = FQCalc(obs_data=gobs, scatter=s, conv=30, potential='rw')
    else:
        calc = value[1]
    ideal_atoms.positions *= 1.02

    ideal_atoms.set_calculator(calc)
    start_pe = ideal_atoms.get_potential_energy()

    traj, _, _ = nuts(ideal_atoms, .65, 3, 1.)

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    min_pe = np.min(pe_list)
    print len(traj)
    del traj
    print min_pe
    assert min_pe < start_pe

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         # '--nocapture'
                         ],
                   exit=False)