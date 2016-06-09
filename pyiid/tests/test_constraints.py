from pyiid.tests import *
from ase.calculators.calculator import Calculator
from ase import Atoms
from pyiid.constraints import SameMagnitude
from ase.cluster import FaceCenteredCubic
from ase.calculators.lj import LennardJones
from pyiid.utils import onion_tag
from ase.constraints import FixedLine

class UnevenCalc(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, **kwargs):
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        """Spring Calculator
        Parameters
        ----------
        atoms: Atoms object
            Contains positions, unit-cell, ...
        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces'
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these five: 'positions', 'numbers', 'cell',
            'pbc', 'charges' and 'magmoms'.
        """

        Calculator.calculate(self, atoms, properties, system_changes)

        # we shouldn't really recalc if charges or magmos change
        if len(system_changes) > 0:  # something wrong with this way
            if 'energy' in properties:
                self.calculate_energy(self.atoms)

            if 'forces' in properties:
                self.calculate_forces(self.atoms)
        for property in properties:
            if property not in self.results:
                if property is 'energy':
                    self.calculate_energy(self.atoms)

                if property is 'forces':
                    self.calculate_forces(self.atoms)

    def calculate_energy(self, atoms):
        """
        Calculate energy
        :param atoms:
        :return:
        """
        energy = 0
        self.results['energy'] = energy

    def calculate_forces(self, atoms):
        self.results['forces'] = np.asarray([
            [1, 0, 0],
            [2, 0, 0]
        ], dtype=float)

def test_constraint1():
    atoms = Atoms('Au2', [[0, 0, 0], [1, 0, 0]])
    calc = UnevenCalc()
    atoms.set_calculator(calc)
    f1 = atoms.get_forces()
    print(f1)
    sm = SameMagnitude([[0, 1]])
    atoms.set_constraint(sm)
    f2 = atoms.get_forces()
    print(f2)
    assert np.any(f1 != f2)
    
def test_constraint2():
    atoms = Atoms(
            FaceCenteredCubic('Au', [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                              (2, 4, 2)))
    calc3 = LennardJones()
    calc3.parameters.update({'sigma':2.8/1.122, 'epsilon':.005})

    onion_tag(atoms)
    fl = []
    for i, atom in enumerate(atoms):
        disp = atom.position - atoms.get_center_of_mass()
        disp[np.where((-1e-15 < disp) & (disp < 1e-15))] = 0.0
        if np.any(disp != 0.0):
            fl.append(FixedLine(i, disp))
    idxsl = []

    for a in set(atoms.get_tags()):
        idxsl.append([atom.index for atom in atoms if atom.tag == a])
    sm = SameMagnitude(idxs_l=idxsl)
    fl.append(sm)

    print(fl)
    atoms.set_constraint(fl)
    atoms.set_calculator(calc3)
    print(atoms.get_forces())
    print(atoms.get_forces())


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
