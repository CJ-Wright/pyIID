from pyiid.tests import *
from ase.calculators.calculator import Calculator
from ase import Atoms
from pyiid.constraints import SameMagnitude

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

def test_constraint():
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

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
