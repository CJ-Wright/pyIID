__author__ = 'christopher'
from ase.calculators.calculator import Calculator
import numpy as np
from pyiid.kernel_wrap import wrap_rw, wrap_grad_rw, wrap_pdf


def get_rw_grad(atoms):
    print 'hi'
    print atoms
    return np.ones((len(atoms), 3))


def get_rw(atoms):
    print 'hi'
    print atoms
    return 1


class pdf_calc(Calculator):
    """
    Class for doing PDF based RW calculations
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                  label=None, atoms=None, gobs = None, Qmin=0.0, Qmax = 25.0, Qbin = None, rmin = 0.0, rmax = 40.0, rbin = .01, conv = 1.,**kwargs):
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        self.Qmin = Qmin
        self.Qmax = Qmax
        if Qbin is None:
            self.Qbin = np.pi/(rmax+6*2*np.pi/Qmax)
        self.rmin = rmin
        self.rmax = rmax
        self.rmin = rmin
        self.rbin = rbin
        self.rw_to_eV = conv
        if gobs is not None:
            self.gobs = gobs
        else:
            raise NotImplementedError('Need an experimental PDF')

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges','magmoms']):
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
        energy, scale, gcalc, FQ = wrap_rw(atoms, self.gobs, self.Qmax, self.Qmin, self.Qbin, self.rmax, self.rbin)
        self.energy_free = energy* self.rw_to_eV
        self.energy_zero = energy* self.rw_to_eV
        self.results['energy'] = energy* self.rw_to_eV

    def calculate_forces(self, atoms):
        self.results['forces'] = np.zeros((len(atoms), 3))
        forces = wrap_grad_rw(atoms, self.gobs, self.Qmax, self.Qmin, self.Qbin, self.rmax, self.rbin) * self.rw_to_eV
        self.results['forces'] = forces


if __name__ == '__main__':
    import cProfile
    from ase import Atoms
    atoms = Atoms
    atoms.get_potential_energy()
    import matplotlib.pyplot as plt
    # cProfile.run('''
    import ase.io as aseio
    from copy import deepcopy as dc
    atoms = aseio.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
    pdf, FQ = wrap_pdf(atoms, Qmin=2.5)
    calc = pdf_calc(gobs=pdf, Qmin=2.5)
    atoms.set_calculator(calc)

    # plt.show()
    atoms2 = dc(atoms)
    atoms2.rattle(stdev=.1)
    pdf2, FQ2 = wrap_pdf(atoms2, Qmin=2.5)
    atoms2.set_calculator(calc)
    # pdf2, FQ2 = wrap_pdf(atoms2)
    print 'start energy'
    # t0 = time.time()
    print(atoms2.get_total_energy())
    # t1 = time.time()
    # print(atoms2.get_forces())
    # t2 = time.time()
    # ''', sort='tottime')
    #     print('energy', t1-t0, 'forces', t2-t1)
    plt.plot(pdf)
    plt.plot(pdf2)
    # plt.plot(pdf-pdf2)
    plt.show()