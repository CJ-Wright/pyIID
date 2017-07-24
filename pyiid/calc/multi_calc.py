import numpy as np
from ase.calculators.calculator import Calculator

__author__ = 'christopher'


class MultiCalc(Calculator):
    """
    Class for doing multiple calculator energy calculations.
    Each of the energies and forces from the sub-calculators are summed
    together to produce the composite potential energy surface
    """
    # TODO: make this so the calculators run in parallel if possible
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, calc_list=None, **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)

        self.calc_list = calc_list

    def calculate(self, atoms=None, properties=None,
                  system_changes=None):
        """PDF Calculator

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

        if system_changes is None:
            system_changes = ['positions', 'numbers', 'cell',
                              'pbc', 'charges', 'magmoms']
        if properties is None:
            properties = ['energy']
        Calculator.calculate(self, atoms, properties, system_changes)

        # we shouldn't really recalc if charges or magmos change
        if len(system_changes) > 0:  # something wrong with this way
            if 'energy' in properties:
                self.calculate_energy(self.atoms)

            if 'forces' in properties:
                self.calculate_forces(self.atoms)
        for calc_property in properties:
            if calc_property not in self.results:
                if calc_property is 'energy':
                    self.calculate_energy(self.atoms)

                if calc_property is 'forces':
                    self.calculate_forces(self.atoms)

    def calculate_energy(self, atoms):
        """
        Calculate energy
        :param atoms:
        :return:
        """
        energy_list = []
        # TODO: parallelism this
        for calculator in self.calc_list:
            atoms.set_calculator(calculator)
            energy_list.append(atoms.get_potential_energy())

        energy = sum(energy_list)
        self.results['energy'] = energy

    def calculate_forces(self, atoms):
        self.results['forces'] = np.zeros((len(atoms), 3))
        forces = np.zeros((len(atoms), 3))

        # TODO: parallelism this
        for calculator in self.calc_list:
            atoms.set_calculator(calculator)
            forces[:, :] += atoms.get_forces()

        self.results['forces'] = forces

    def calculate_voxel_energy(self, atoms, resolution):
        c = np.diagonal(atoms.get_cell())
        voxel_energy = np.zeros(c / resolution)
        for calc in self.calc_list:
            try:
                voxel_energy += calc.calculate_voxel_energy(atoms, resolution)
            except AttributeError:
                # for voxel in voxels compute energy
                # need to know what kind of atom is being added
                '''
                im, jm, km = voxel_energy.shape
                for i in range(im):
                    x = (i + .5) * resolution
                    for j in range(jm):
                        y = (j + .5) * resolution
                        for k in range(km):
                            z = (k + .5) * resolution
                            atoms2 = dc(atoms)
                            atoms2 += Atom(element, (x, y, z))
                            voxel_energy[i, j, k] +=
                            atoms2.get_potential_energy()
                '''
                pass
        return voxel_energy

    def calculate_atomwise_energy(self, atoms):
        nrg = np.zeros(len(atoms))
        for calc in self.calc_list:
            try:
                nrg += calc.calculate_atomwise_energy(atoms)
            except AttributeError:
                # for every atom remove it and get the energy
                '''
                for atom in atoms:
                    atoms2 = dc(atoms)
                    del atoms2[atom.index]
                    nrg[atom.index] += atoms2.get_potential_energy()
                '''
                pass
        return nrg
