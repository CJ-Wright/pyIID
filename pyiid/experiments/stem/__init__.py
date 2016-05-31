from __future__ import print_function

"""
The basic idea of this module is to provide the STEM class, which simulates
STEM images from atomic configurations. Note that these simulations are very
simplistic, they make the following assumptions.
1. Atoms do not lose/gain/share electrons
2. Atoms' electron clouds are spherically symmetrical
3. Total electron density is a approximated by the atomic number
4. Electron density is radially shaped like a gaussian function

Thus we can compute the electron density for an atomic system by treating
each atom as a gaussian with intensity = f(0) and sigma = covalent radius
"""
import numpy as np
from scipy.stats import norm
from ase.data import covalent_radii
from builtins import range


def get_atomic_electron_density(atom, voxels, resolution):
    # make var/covar matrix
    sigma = np.zeros((3, 3))
    for i in range(3):
        sigma[i, i] = covalent_radii[atom.number]

    # make gaussian distribution centered on atom
    r = np.zeros(voxels.shape)
    q = atom.position
    im, jm, km, = voxels.shape
    for i in range(im):
        x = (i + .5) * resolution[0]
        for j in range(jm):
            y = (j + .5) * resolution[1]
            for k in range(km):
                z = (k + .5) * resolution[2]
                r[i, j, k] = 0
                for l, w in zip(range(3), [x, y, z]):
                    r[i, j, k] += (w - q[l]) ** 2

    # make gaussian
    # put gaussian on voxel grid
    voxel_gaussian = norm.pdf(r)

    # put e density in voxel
    ed = voxel_gaussian * atom.number
    return ed


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ase import Atoms
    from ase.visualize import view
    from ase.cluster import FaceCenteredCubic
    import ase.io as aseio

    # atoms = Atoms('Pt', [(0, 0, 0)])
    atoms = Atoms(
        FaceCenteredCubic('Pt', [[1, 0, 0], [1, 1, 0], [1, 1, 1]], (2, 3, 2)))
    # atoms = atoms[[atom.index for atom in atoms if atom.position[2]< 1.5]]
    view(atoms)
    atoms.set_cell(atoms.get_cell() * 1.2)
    atoms.center()
    cell = atoms.get_cell()
    print(cell, len(atoms))
    resolution = .3 * np.ones(3)
    c = np.diagonal(cell)
    v = tuple(np.int32(np.ceil(c / resolution)))
    voxels = np.zeros(v)
    ed = np.zeros(v)
    i = 0
    for atom in atoms:
        print(i)
        ed += get_atomic_electron_density(atom, voxels, resolution)
        i += 1
    print(ed[:, :, 0])
    plt.imshow(np.sum(ed, axis=2), cmap='viridis')
    plt.show()
