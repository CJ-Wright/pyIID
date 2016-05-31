from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import numpy as np
from copy import deepcopy as dc
from pyiid.sim import leapfrog

__author__ = 'christopher'
print('No longer supported')

delta_max = 500


def nuts(atoms, e, iterations, wtraj=None):
    traj = [atoms]
    configuration_list = []
    try:
        for m in range(iterations):
            # XXX ERRORR HERE
            # initialize momenta
            rand_momenta = np.random.normal(0, 1, (len(atoms), 3))
            atoms.set_momenta(rand_momenta)

            # get slice variable
            u = np.random.uniform(0, np.exp(-1 * traj[-1].get_total_energy()))
            # u = np.random.uniform(0, traj[-1].get_total_energy())
            # print u

            # initialize negative and positive directions
            neg_atoms = dc(traj[-1])
            pos_atoms = dc(traj[-1])

            c = [dc(traj[-1])]
            j, s = 0, 1
            while s == 1:
                print('depth', j, 'samples', 2 ** j)
                v = np.random.choice([-1, 1])
                if v == -1:
                    # build tree in negative direction
                    neg_atoms, _, cp, sp = buildtree(neg_atoms, u, v, j, e)
                else:
                    _, pos_atoms, cp, sp = buildtree(pos_atoms, u, v, j, e)

                if sp == 1:
                    c += cp
                datoms = pos_atoms.positions - neg_atoms.positions
                s = sp * \
                    (np.dot(datoms.flatten(),
                            neg_atoms.get_momenta().flatten()) >= 0) * \
                    (np.dot(datoms.flatten(),
                            pos_atoms.get_momenta().flatten()) >= 0)
                j += 1
            print(m, len(c))
            configuration_list.append(c)
            energies = np.asarray([atoms.get_total_energy() for atoms in c])
            sample_pos = np.random.choice(range(len(c)),
                                          p=np.exp(-energies) / np.exp(
                                              -energies).sum())
            # sample_pos = np.random.choice(range(len(C)))
            # sample_pos = np.argmin(energies)
            # XXX ERROR HERE
            if wtraj is not None:
                wtraj.write(atoms)
            traj.append(c[sample_pos])
    except KeyboardInterrupt:
        if wtraj is not None:
            wtraj.write(atoms)
    return traj, configuration_list


def buildtree(input_atoms, u, v, j, e):
    if j == 0:
        atomsp = leapfrog(input_atoms, v * e)
        if u <= np.exp(-atomsp.get_total_energy()):
            # if u <= atomsp.get_total_energy():
            cp = [atomsp]
        else:
            # print 'Fail MH'
            cp = []
        sp = u < np.exp(delta_max - atomsp.get_total_energy())
        # sp = u < delta_max + atomsp.get_total_energy()
        if sp == 0:
            # print 'Fail slice test'
            pass
        return atomsp, atomsp, cp, sp
    else:
        neg_atoms, pos_atoms, cp, sp = buildtree(input_atoms, u, v, j - 1, e)
        if v == -1:
            neg_atoms, _, cpp, spp = buildtree(neg_atoms, u, v, j - 1, e)
        else:
            _, pos_atoms, cpp, spp = buildtree(pos_atoms, u, v, j - 1, e)
        datoms = pos_atoms.positions - neg_atoms.positions
        sp = sp * spp * (
            np.dot(datoms.flatten(), neg_atoms.get_momenta().flatten()) >= 0) \
             * (np.dot(datoms.flatten(),
                       pos_atoms.get_momenta().flatten()) >= 0)
        cp += cpp
        return neg_atoms, pos_atoms, cp, sp
