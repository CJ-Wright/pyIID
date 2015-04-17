__author__ = 'christopher'
from ase.calculators.calculator import Calculator
import numpy as np


class Spring(Calculator):
    """
    Class for doing PDF based RW calculations
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, k=10, rt=1.5, **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)

        from pyiid.wrappers.serial_cpu_wrap import spring_nrg, spring_force
        self.nrg_func = spring_nrg
        self.f_func = spring_force
        self.k = k
        self.rt = rt

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        """PDF Calculator

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
        energy = self.nrg_func(atoms, self.k, self.rt)

        self.energy_free = energy
        self.energy_zero = energy
        self.results['energy'] = energy

    def calculate_forces(self, atoms):
        self.results['forces'] = np.zeros((len(atoms), 3))

        forces = self.f_func(atoms, self.k, self.rt)

        self.results['forces'] = forces


if __name__ == '__main__':
    from ase.atoms import Atoms
    from ase.visualize import view
    from pyiid.calc.pdfcalc import PDFCalc
    from pyiid.wrappers.multi_gpu_wrap import wrap_pdf
    from pyiid.wrappers.serial_cpu_wrap import wrap_atoms
    from pyiid.kernels.serial_kernel import get_r_array, get_d_array

    ideal_atoms = Atoms('Au4', [[0,0,0], [1,0,0], [0, 1, 0], [1,1,0]])
    start_atoms = Atoms('Au4', [[0,0,0], [2,0,0], [0, 2., 0], [2,2,0]])
    # ideal_atoms = Atoms('Au2', [[0,0,0], [2.0,0,0]])
    rt = 2.5
    k = 10

    '''q = start_atoms.positions
    n = len(ideal_atoms)
    d = np.zeros((n, n, 3))
    get_d_array(d, q)
    r = np.zeros((n, n))
    get_r_array(r, d)

    thresh = np.less(r, rt)
    for i in range(len(thresh)):
        thresh[i,i] = False


    mag = np.zeros(r.shape)
    mag[thresh] = k * (r[thresh]-rt)
    # print mag
    direction = np.zeros(q.shape)
    for i in range(len(q)):
        for j in range(len(q)):
            if i != j:
                direction[i,:] += d[i,j,:]/r[i,j] * mag[i, j]

    print direction

    nrg = np.sum(mag[thresh]/2.*(r[thresh]-rt))
    print nrg'''

    calc = Spring(k=k, rt=rt)
    start_atoms.set_calculator(calc)
    f = start_atoms.get_forces()
    start_atoms.get_potential_energy()
    print f
    view(start_atoms)