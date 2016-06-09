import numpy as np


class SameMagnitude:
    """
    Force atoms to have the same magnitude forces
    """
    def __init__(self, idxs_l):
        """

        Parameters
        ----------
        idxs_l: list of lists of ints
            A list of lists of atom indices, all indicies in the same inner
            list will have the same magnitude of force, calculated to be
            average magnitude for all the atoms in the group.
        """
        self.idxs_l = idxs_l

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):
        i = 0
        for idxs in self.idxs_l:
            f = forces[idxs, :]
            mags = np.sqrt(np.sum(f ** 2, axis=1))
            norm_f = np.nan_to_num(f.T / mags).T
            ave_mags = np.average(mags)
            forces[idxs, :] = norm_f * ave_mags
            i += 1
