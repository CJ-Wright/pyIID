import numpy as np

__author__ = 'christopher'


def has_adp(atoms):
    for a in ['adps', 'adp']:
        if a in atoms.info.keys():
            return atoms.info[a]
    return None


class ADP:
    def __init__(self, atoms, adps=None, adp_momenta=None,
                 adp_equivalency=None,
                 fixed_adps=None):
        """
        Set up the atomic anisotropic displacement parameters.
        Parameters
        ----------
        atoms: ASE.atoms
            The atomic configuration
        adps: 2d array
            The array of ADP values for the atomic configuration
        adp_momenta: 2d array
            The "momentum" for each ADP, note that this is not real momentum
            just a mathematical construct for the Hamiltonian dynamics
        adp_equivalency: 2d array
            An array which describes which ADPS are forced to be equivalent
        fixed_adps: 2d array
            An array which describes which adps are fixed

        Returns
        -------

        """
        if adps is None:
            adps = np.ones(atoms.positions.shape) * .005
        if adp_momenta is None:
            adp_momenta = np.zeros(atoms.positions.shape)
        if adp_equivalency is None:
            adp_equivalency = np.arange(len(atoms) * 3).reshape(
                (len(atoms), 3))
        if fixed_adps is None:
            fixed_adps = np.ones(atoms.positions.shape)
        self.adps = adps.copy()
        self.adp_momenta = adp_momenta.copy()
        self.adp_equivalency = adp_equivalency.copy()
        self.fixed_adps = fixed_adps.copy()
        self.calc = None

    def get_positions(self):
        """
        Get the ADP values
        Returns
        -------
        2darray:
            The current ADPs
        """
        return self.adps.copy()

    def set_positions(self, new_adps):
        """
        Set the ADP positions in a manner consistent with the constraints
        Parameters
        ----------
        new_adps: 2darray
            The new adp values

        Returns
        -------

        """
        delta_adps = new_adps.copy() - self.adps
        # Make all the equivalent adps the same
        unique_adps = np.unique(self.adp_equivalency)
        for i in unique_adps:
            delta_adps[np.where(self.adp_equivalency == i)] = np.mean(
                delta_adps[np.where(self.adp_equivalency == i)])
        # No changes for the fixed adps, where fixed is zero
        delta_adps *= self.fixed_adps
        self.adps += delta_adps

    def get_momenta(self):
        """
        Get the current ADP momenta
        Returns
        -------
        2darray:
            The current momentum
        """
        return self.adp_momenta

    def set_momenta(self, new_momenta):
        """
        Set the ADP momentum
        Parameters
        ----------
        new_momenta: 2darray
            The new momentum values

        Returns
        -------

        """
        self.adp_momenta = new_momenta.copy()

    def get_velocities(self):
        return self.get_momenta()

    def get_forces(self, atoms):
        """
        Get the forces on the ADPs from the APD calculator
        ..note:: It may seem a bit odd to have forces working on the ADPs
        since they aren't particle positions but a better way to think of it as
        a description of how the ADPs should change, in both magnitude and
        direction to best minimize the potential energy of the system (however
        that is calculated).
        Parameters
        ----------
        atoms: ase.atoms
            The atomic configuration

        Returns
        -------
        2darray:
            The forces on each of the adps

        """
        return self.calc.get_forces(atoms)

    def set_calculator(self, calc):
        """
        Set the calculator for the ADPS, this will calculate the potential
        energy and forces associated with the ADPS
        Parameters
        ----------
        calc

        Returns
        -------

        """
        self.calc = calc

    def del_adp(self, index):
        """
        Delete an ADP from the system, usually this accompanies the deletion
        of an atom.
        Parameters
        ----------
        index: int
            The index of the ADP to be deleted

        Returns
        -------

        """
        for a in [self.adps, self.adp_momenta, self.adp_equivalency,
                  self.fixed_adps]:
            a = np.delete(a, index, 0)

    def add_adp(self, adp=None, adp_momentum=None, adp_equivalency=None,
                fixed_adp=None):
        """
        Add an ADP to the system, usually this accompanies the addition of an
        atom.
        Parameters
        ----------
        adp: 1darray
            The new adp values
        adp_momentum: 1darray
            The momentum of the new adp
        adp_equivalency: int
            Which adps this adp is equivalent to
        fixed_adp: int
            The fixed values for the adps.

        Returns
        -------

        """
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
