__author__ = 'christopher'
import numpy as np
from ase.constraints import FixConstraint


class RevHookean(FixConstraint):
    """Applies a Hookean restorative force between a pair of atoms, an atom
    and a point, or an atom and a plane."""

    def __init__(self, a1, a2, k, rt=None):
        """Forces two atoms to stay close together by applying no force if
        they are below a threshold length, rt, and applying a Hookean
        restorative force when the distance between them exceeds rt. Can also be
        used to tether an atom to a fixed point in space or to a distance above
        a plane.

        Parameters
        ----------
        a1 : int
           Index of atom 1
        a2 : one of three options
           1) index of atom 2
           2) a fixed point in cartesian space to which to tether a1
           3) a plane given as (A, B, C, D) in A x + B y + C z + D = 0.
        k : float
           Hooke's law (spring) constant to apply when distance
           exceeds threshold_length.
        rt : float
           The threshold length below which there is no force. The
           length is 1) between two atoms, 2) between atom and point.
           This argument is not supplied in case 3.

        If a plane is specified, the Hooke's law force is applied if the atom
        is on the normal side of the plane. For instance, the plane with
        (A, B, C, D) = (0, 0, 1, -7) defines a plane in the xy plane with a z
        intercept of +7 and a normal vector pointing in the +z direction.
        If the atom has z > 7, then a downward force would be applied of
        k * (atom.z - 7). The same plane with the normal vector pointing in
        the -z direction would be given by (A, B, C, D) = (0, 0, -1, 7).
        """

        if type(a2) == int:
            self._type = 'two atoms'
            self.indices = [a1, a2]
        self.threshold = rt
        self.spring = k

    def adjust_positions(self, oldpositions, newpositions):
        pass

    def adjust_forces(self, positions, forces):
        if self._type == 'two atoms':
            p1, p2 = positions[self.indices]
        displace = p2 - p1
        bondlength = np.linalg.norm(displace)
        if (bondlength < self.threshold) and (self.threshold > 0):
            magnitude = self.spring * (bondlength - self.threshold)
            direction = displace / np.linalg.norm(displace)
            if self._type == 'two atoms':
                forces[self.indices[0]] += direction * magnitude
                forces[self.indices[1]] -= direction * magnitude
            else:
                forces[self.index] += direction * magnitude

    def adjust_potential_energy(self, positions, forces):
        """Returns the difference to the potential energy due to an active
        constraint."""
        if self._type == 'plane':
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = ((A * x + B * y + C * z + D) /
                 np.sqrt(A**2 + B**2 + C**2))
            if d > 0:
                return 0.5 * self.spring * d**2
            else:
                return 0.
        else:
            raise NotImplementedError('Adjust potential energy only '
                                      'implemented for plane.')

    def __repr__(self):
        if self._type == 'two atoms':
            return 'Hookean(%d, %d)' % tuple(self.indices)
        elif self._type == 'point':
            return 'Hookean(%d) to cartesian' % self.index
        else:
            return 'Hookean(%d) to plane' % self.index

    def copy(self):
        if self._type == 'two atoms':
            return RevHookean(a1=self.indices[0], a2=self.indices[1],
                           rt=self.threshold, k=self.spring)
        elif self._type == 'point':
            return RevHookean(a1=self.index, a2=self.origin,
                           rt=self.threshold, k=self.spring)
        else:
            return RevHookean(a1=self.index, a2=self.plane,
                           k=self.spring)

if __name__ == '__main__':
    from ase.atoms import Atoms
    from ase.visualize import view
    from pyiid.calc.pdfcalc import PDFCalc
    from pyiid.wrappers.multi_gpu_wrap import wrap_pdf
    from pyiid.wrappers.serial_cpu_wrap import wrap_atoms
    ideal_atoms = Atoms('Au4', [[0,0,0], [1,0,0], [0, 1, 0], [1,1,0]])
    start_atoms = Atoms('Au4', [[0,0,0], [.9,0,0], [0, .9, 0], [.9,.9,0]])

    wrap_atoms(ideal_atoms)
    wrap_atoms(start_atoms)

    gobs = wrap_pdf(ideal_atoms)[0]

    calc = PDFCalc(gobs=gobs, qbin=.1, conv=1000, potential='rw')
    start_atoms.set_calculator(calc)
    q = ideal_atoms.positions

    '''
    c = []
    for i in range(len(start_atoms)):
        for j in range(i+1, len(start_atoms)):
            c.append(RevHookean(i, j, k=100, rt=1.1))
    start_atoms.set_constraint(c)
    # '''
    f = start_atoms.get_forces()
    e = start_atoms.get_total_energy()
    print e
    print f
    f2 = 2*f
    # view(start_atoms)