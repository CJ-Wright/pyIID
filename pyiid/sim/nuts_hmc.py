__author__ = 'christopher'
from numbapro import autojit

from ase.atoms import Atoms as atoms
from copy import deepcopy as dc
import numpy as np

Emax = 500

@autojit(target='cpu')
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


def find_step_size(input_atoms):
    atoms = dc(input_atoms)
    step_size = .8e-2
    '''
    atoms.set_momenta(np.random.normal(0, 1, (len(atoms), 3)))

    atoms_prime = leapfrog(atoms, step_size)

    a = 2 * (np.exp(-1*atoms_prime.get_total_energy() + atoms.get_total_energy()) > 0.5) - 1
    while (np.exp(-1*atoms_prime.get_total_energy() + atoms.get_total_energy())) ** a > 2 ** -a:
        step_size *= 2 ** a
        atoms_prime = leapfrog(atoms, step_size)
    '''
    print step_size
    return step_size


def nuts(atoms, accept_target, iterations, wtraj=None):
    traj = [atoms]

    atoms0 = dc(atoms)
    m = 0

    step_size = find_step_size(atoms)
    mu = np.log(10*step_size)
    # ebar = 1
    Hbar = 0
    gamma = 0.05
    t0 = 10
    # k = .75
    print 'start hmc'
    try:
        while m <= iterations:
            print 'step', step_size
            rand_momenta = np.random.normal(0, 1, (len(atoms), 3))
            atoms.set_momenta(rand_momenta)
            u = np.random.uniform(0, np.exp(-1.*traj[-1].get_total_energy()))

            e = step_size
            n, s, j = 1, 1, 0
            neg_atoms = dc(atoms)
            pos_atoms = dc(atoms)
            while s == 1:
                v = np.random.choice([-1, 1])
                if v == -1:
                    neg_atoms, _, atoms_prime, n_prime, s_prime, a, na = buildtree(neg_atoms, u, v, j, e, atoms0)
                else:
                    _, pos_atoms, atoms_prime, n_prime, s_prime, a, na = buildtree(pos_atoms, u, v, j, e, atoms0)

                if s_prime == 1 and np.random.uniform() < min(1, n_prime * 1. / n):
                    traj += [atoms_prime]
                    if wtraj is not None:
                        wtraj.write(atoms_prime)
                    atoms = atoms_prime
                n = n + n_prime
                span = pos_atoms.positions - neg_atoms.positions
                span = span.flatten()
                s = s_prime * (span.dot(neg_atoms.get_velocities().flatten()) >= 0) * (span.dot(pos_atoms.get_velocities().flatten()) >= 0)
                j += 1
            print 'iteration', m, 'depth', j, 'samples', 2**j
            w = 1. / (m + t0)
            Hbar = (1 - w) * Hbar + w * (accept_target - a / na)

            step_size = np.exp(mu - (m ** .5 / gamma) * Hbar)

            m += 1
    except KeyboardInterrupt:
        pass
    return traj


# @autojit(target='cpu')
def buildtree(input_atoms, u, v, j, e, atoms0):
    if j == 0:
        atoms_prime = leapfrog(input_atoms, v*e)
        n_prime = int(u <= np.exp(-atoms_prime.get_total_energy()))
        s_prime = int(u <= np.exp(Emax-atoms_prime.get_total_energy()))

        return atoms_prime, atoms_prime, atoms_prime, n_prime, s_prime, min(1, np.exp(-atoms_prime.get_total_energy() + input_atoms.get_total_energy())), 1
    else:
        neg_atoms, pos_atoms, atoms_prime, n_prime, s_prime, a_prime, na_prime = buildtree(
            input_atoms, u, v, j - 1, e, atoms0)
        if s_prime == 1:
            if v == -1:
                neg_atoms, _, atoms_prime_prime, n_prime_prime, s_prime_prime, app, napp = buildtree(
                    neg_atoms, u, v, j - 1, e, atoms0)
            else:
                _, pos_atoms, atoms_prime_prime, n_prime_prime, s_prime_prime, app, napp = buildtree(
                    pos_atoms, u, v, j - 1, e, atoms0)

            if np.random.uniform() < float(n_prime_prime / (max(n_prime + n_prime_prime, 1))):
                atoms_prime = atoms_prime_prime

            a_prime = a_prime + app
            na_prime = na_prime + napp

            datoms = pos_atoms.positions - neg_atoms.positions
            span = datoms.flatten()
            s_prime = s_prime_prime * (span.dot(neg_atoms.get_velocities().flatten()) >= 0) * (span.dot(pos_atoms.get_velocities().flatten()) >= 0)
            n_prime = n_prime + n_prime_prime
        return neg_atoms, pos_atoms, atoms_prime, n_prime, s_prime, a_prime, na_prime


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

    traj = nuts(start_atoms, 0.65, M)
    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
        f = atoms.get_forces()
        f2 = f*2
    min_pe = np.argmin(pe_list)
    view(traj)
    rw, scale, _, _ = wrap_rw(traj[-1], gobs)
    print traj[-1].get_potential_energy(),rw, scale
    r = np.arange(0, 40, .01)
    plt.plot(r, gobs, label='gobs')
    plt.plot(r, wrap_pdf(traj[0], qbin=.1)[0]*wrap_rw(traj[0],gobs, qbin=.1)[1], label='ginitial')
    plt.plot(r, wrap_pdf(traj[min_pe], qbin=.1)[0]*wrap_rw(traj[-1],gobs, qbin=.1)[1], label='gfinal')
    plt.legend()
    plt.show()