__author__ = 'christopher'

import numpy as np
from copy import deepcopy as dc

delta_max = 500


def nuts(atoms, e, M, wtraj=None):
    traj = [atoms]
    C_list = []
    try:
        for m in range(M):
            # XXX ERRORR HERE
            #initialize momenta
            rand_momenta = np.random.normal(0, 1, (len(atoms), 3))
            atoms.set_momenta(rand_momenta)

            #get slice variable
            u = np.random.uniform(0, np.exp(-1*traj[-1].get_total_energy()))
            # u = np.random.uniform(0, traj[-1].get_total_energy())
            # print u

            #initialize negative and positive directions
            neg_atoms = dc(traj[-1])
            pos_atoms = dc(traj[-1])

            C = [dc(traj[-1])]
            j, s = 0, 1
            while s == 1:
                print 'depth', j, 'samples', 2**j
                v = np.random.choice([-1, 1])
                if v == -1:
                    #build tree in negative direction
                    neg_atoms, _, Cp, sp = buildtree(neg_atoms, u, v, j, e)
                else:
                    _, pos_atoms, Cp, sp = buildtree(pos_atoms, u, v, j, e)

                if sp == 1:
                    C += Cp
                datoms = pos_atoms.positions - neg_atoms.positions
                s = sp * (np.dot(datoms.flatten(), neg_atoms.get_momenta().flatten()) >= 0) * (np.dot(datoms.flatten(), pos_atoms.get_momenta().flatten()) >= 0)
                j += 1
            print m, len(C)
            C_list.append(C)
            energies = np.asarray([atoms.get_total_energy() for atoms in C])
            sample_pos = np.random.choice(range(len(C)), p=np.exp(-energies)/np.exp(-energies).sum())
            # sample_pos = np.random.choice(range(len(C)))
            # sample_pos = np.argmin(energies)
            # XXX ERROR HERE
            if wtraj is not None:
                wtraj.write(atoms)
            traj.append(C[sample_pos])
    except KeyboardInterrupt:
        if wtraj is not None:
            wtraj.write(atoms)
    return traj, C_list


def buildtree(input_atoms, u, v, j, e):
    if j == 0:
        atomsp = leapfrog(input_atoms, v*e)
        if u <= np.exp(-atomsp.get_total_energy()):
        # if u <= atomsp.get_total_energy():
            Cp = [atomsp]
        else:
            # print 'Fail MH'
            Cp = []
        sp = u < np.exp(delta_max - atomsp.get_total_energy())
        # sp = u < delta_max + atomsp.get_total_energy()
        if sp == 0:
            # print 'Fail slice test'
            pass
        return atomsp, atomsp, Cp, sp
    else:
        neg_atoms, pos_atoms, Cp, sp = buildtree(input_atoms, u, v, j-1, e)
        if v == -1:
            neg_atoms, _, Cpp, spp = buildtree(neg_atoms, u, v, j-1, e)
        else:
            _,pos_atoms, Cpp, spp = buildtree(pos_atoms, u, v, j-1, e)
        datoms = pos_atoms.positions - neg_atoms.positions
        sp = sp*spp*(np.dot(datoms.flatten(), neg_atoms.get_momenta().flatten()) >= 0) * (np.dot(datoms.flatten(), pos_atoms.get_momenta().flatten()) >= 0)
        Cp += Cpp
        return neg_atoms, pos_atoms, Cp, sp


def leapfrog(atoms, step):
    """
    Propagate the dynamics of the system via the leapfrog algorithm one step

    Parameters
    -----------
    atoms: ase.Atoms ase.Atoms
        The atomic configuration for the system
    step: float
        The step size for the simulation, the new momentum/velocity is step *
        the force

    Returns
    -------
    ase.Atoms
        The new atomic positions and velocities
    """
    latoms = dc(atoms)
    
    latoms.set_momenta(latoms.get_momenta() + 0.5 * step * latoms.get_forces())

    latoms.positions += step * latoms.get_velocities()

    latoms.set_momenta(latoms.get_momenta() + 0.5 * step * latoms.get_forces())
    return latoms


if __name__ == '__main__':
    import os
    from copy import deepcopy as dc
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.visualize import view
    from ase.io.trajectory import PickleTrajectory
    import ase.io as aseio
    import time
    import datetime
    import math
    from ase.atoms import Atoms

    from pyiid.sim.hmc import run_hmc
    from pyiid.wrappers.multi_gpu_wrap import wrap_rw, wrap_pdf
    from pyiid.calc.pdfcalc import PDFCalc
    from pyiid.wrappers.serial_cpu_wrap import wrap_atoms
    from pyiid.utils import tag_surface_atoms, build_sphere_np, load_gr_file, time_est
    
    ideal_atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    # start_atoms = Atoms('Au4', [[-1, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    start_atoms = dc(ideal_atoms)
    start_atoms.positions *= 1.05
    
    wrap_atoms(ideal_atoms)
    wrap_atoms(start_atoms)
    
    gobs, fq = wrap_pdf(ideal_atoms, qbin=.1)
    
    calc = PDFCalc(gobs=gobs, conv=100, potential='rw', qbin=.1)
    # calc = PDFCalc(gobs=gobs, conv=1000, qbin=.1)
    start_atoms.set_calculator(calc)
    print start_atoms.get_potential_energy()
    
    e = .8e-2
    M = 10
    
    traj, C_list = nuts(start_atoms, e, M)
    for c in C_list:
        for atoms in c:
            f = atoms.get_forces()
            f2 = f*2
        view(c)

    # raw_input('')
    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
        f = atoms.get_forces()
        f2 = f*2
    min_pe = np.argmin(pe_list)
    view(traj)
    rw, scale, _, _ = wrap_rw(traj[-1], gobs)
    print rw, scale
    r = np.arange(0, 40, .01)
    plt.plot(r, gobs, label='gobs')
    plt.plot(r, wrap_pdf(traj[0], qbin=.1)[0]*wrap_rw(traj[0],gobs, qbin=.1)[1], label='ginitial')
    plt.plot(r, wrap_pdf(traj[min_pe], qbin=.1)[0]*wrap_rw(traj[min_pe],gobs, qbin=.1)[1], label='gfinal')
    plt.legend()
    plt.show()