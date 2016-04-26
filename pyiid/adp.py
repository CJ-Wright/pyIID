import numpy as np

__author__ = 'christopher'


class ADP:
    def __init__(self, atoms, adps=None, adp_momenta=None,
                 adp_equivalency=None,
                 fixed_adps=None):
        if adps is None:
            adps = np.ones(atoms.positions.shape) * .005
        if adp_momenta is None:
            adp_momenta = np.zeros(atoms.positions.shape)
        if adp_equivalency is None:
            adp_equivalency = np.arange(len(atoms) * 3).reshape(
                (len(atoms), 3))
        if fixed_adps is None:
            fixed_adps = np.ones(atoms.positions.shape)
        self.adps = adps
        self.adp_momenta = adp_momenta
        self.adp_equivalency = adp_equivalency
        self.fixed_adps = fixed_adps
        self.calc = None

    def get_position(self):
        return self.adps

    def set_position(self, new_adps):
        delta_adps = new_adps - self.adps
        # Make all the equivalent adps the same
        unique_adps = np.unique(self.adp_equivalency)
        for i in unique_adps:
            delta_adps[np.where(self.adp_equivalency == i)] = np.mean(
                delta_adps[np.where(self.adp_equivalency == i)])
        # No changes for the fixed adps, where fixed is zero
        delta_adps *= self.fixed_adps
        self.adps += delta_adps

    def get_momenta(self):
        return self.adp_momenta

    def set_momenta(self, new_momenta):
        self.adp_momenta = new_momenta

    def get_forces(self, atoms):
        return self.calc.calculate_forces(atoms)

    def set_calc(self, calc):
        self.calc = calc

    def del_adp(self, index):
        for a in [self.adps, self.adp_momenta, self.adp_equivalency,
                  self.fixed_adps]:
            a = np.delete(a, index, 0)

    def add_adp(self, adp=None, adp_momentum=None, adp_equivalency=None,
                fixed_adp=None):
        if adp is None:
            adp = np.ones((1, 3)) * .005
        if adp_momentum is None:
            adp_momentum = np.zeros((1, 3))
        if adp_equivalency is None:
            adp_equivalency = np.arange(3) + np.max(self.adp_equivalency)
        if fixed_adp is None:
            fixed_adp = np.ones((1, 3))
        for a, b in zip(
                [self.adps, self.adp_momenta,
                 self.adp_equivalency, self.fixed_adps],
                [adp, adp_momentum, adp_equivalency, fixed_adp]):
            a = np.vstack([a, b])
