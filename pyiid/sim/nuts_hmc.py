from pyiid.sim import leapfrog

__author__ = 'christopher'

from ase.atoms import Atoms as atoms
from copy import deepcopy as dc
import numpy as np

Emax = 200


def find_step_size(input_atoms):
    atoms = dc(input_atoms)
    step_size = 1.5
    # '''
    atoms.set_momenta(np.random.normal(0, 1, (len(atoms), 3)))

    atoms_prime = leapfrog(atoms, step_size)

    a = 2 * (np.exp(
        -1 * atoms_prime.get_total_energy() + atoms.get_total_energy()) > 0.5) - 1

    while (np.exp(
                    -1 * atoms_prime.get_total_energy() + atoms.get_total_energy())) ** a > 2 ** -a:
        print 'initial step size', a
        step_size *= 2 ** a
        atoms_prime = leapfrog(atoms, step_size)
    # '''
    print step_size
    return step_size


def nuts(atoms, accept_target, iterations, p_scale=1, wtraj=None):
    if wtraj is not None:
        atoms.set_momenta(np.random.normal(0, p_scale, (len(atoms), 3)))
        atoms.get_forces()
        atoms.get_potential_energy()
        wtraj.write(atoms)
    traj = [atoms]

    atoms0 = dc(atoms)
    m = 0

    step_size = find_step_size(atoms)
    mu = np.log(10 * step_size)
    # ebar = 1
    Hbar = 0
    gamma = 0.05
    t0 = 10
    # k = .75
    samples_total = 0
    print 'start hmc'
    try:
        while m <= iterations:
            print 'step', step_size
            atoms.set_momenta(np.random.normal(0, p_scale, (len(atoms), 3)))
            # u = np.random.uniform(0, np.exp(-1.*traj[-1].get_total_energy()))
            e0 = traj[-1].get_total_energy()
            u = np.random.uniform(0, 1)

            e = step_size
            n, s, j = 1, 1, 0
            neg_atoms = dc(atoms)
            pos_atoms = dc(atoms)
            while s == 1:
                v = np.random.choice([-1, 1])
                if v == -1:
                    neg_atoms, _, atoms_prime, n_prime, s_prime, a, na = buildtree(
                        neg_atoms, u, v, j, e, atoms0, e0)
                else:
                    _, pos_atoms, atoms_prime, n_prime, s_prime, a, na = buildtree(
                        pos_atoms, u, v, j, e, atoms0, e0)

                if s_prime == 1 and np.random.uniform() < min(1,
                                                              n_prime * 1. / n):
                    traj += [atoms_prime]
                    if wtraj is not None:
                        atoms_prime.get_forces()
                        atoms_prime.get_potential_energy()
                        wtraj.write(atoms_prime)
                    atoms = atoms_prime
                n = n + n_prime
                span = pos_atoms.positions - neg_atoms.positions
                span = span.flatten()
                s = s_prime * (
                    span.dot(neg_atoms.get_velocities().flatten()) >= 0) * (
                        span.dot(pos_atoms.get_velocities().flatten()) >= 0)
                j += 1
                print 'iteration', m, 'depth', j, 'samples', 2 ** j
                samples_total += 2 ** j
            w = 1. / (m + t0)
            Hbar = (1 - w) * Hbar + w * (accept_target - a / na)

            step_size = np.exp(mu - (m ** .5 / gamma) * Hbar)

            m += 1
    except KeyboardInterrupt:
        if m == 0:
            m = 1
        pass
    print '\n\n\n'
    print 'number of leapfrog samples', samples_total
    print 'number of successful leapfrog samples', len(traj) - 1
    print 'percent of good leapfrog samples', float(
        len(traj) - 1) / samples_total * 100, '%'
    print 'number of leapfrog per iteration, average', float(samples_total) / m
    print 'number of good leapfrog per iteration, average', float(
        len(traj) - 1) / m
    return traj, samples_total, float(samples_total) / m


# @autojit(target='cpu')
def buildtree(input_atoms, u, v, j, e, atoms0, e0):
    if j == 0:
        atoms_prime = leapfrog(input_atoms, v * e)
        neg_delta_energy = e0 - atoms_prime.get_total_energy()
        try:
            exp1 = np.exp(neg_delta_energy)
            exp2 = np.exp(Emax + neg_delta_energy)
        except:
            exp1 = 0
            exp2 = 0
        # print exp1, exp2
        # n_prime = int(u <= np.exp(-atoms_prime.get_total_energy()))
        # s_prime = int(u <= np.exp(Emax-atoms_prime.get_total_energy()))
        n_prime = int(u <= exp1)
        s_prime = int(u < exp2)
        return atoms_prime, atoms_prime, atoms_prime, n_prime, s_prime, min(1,
                                                                            np.exp(
                                                                                -atoms_prime.get_total_energy() + input_atoms.get_total_energy())), 1
    else:
        neg_atoms, pos_atoms, atoms_prime, n_prime, s_prime, a_prime, na_prime = buildtree(
            input_atoms, u, v, j - 1, e, atoms0, e0)
        if s_prime == 1:
            if v == -1:
                neg_atoms, _, atoms_prime_prime, n_prime_prime, s_prime_prime, app, napp = buildtree(
                    neg_atoms, u, v, j - 1, e, atoms0, e0)
            else:
                _, pos_atoms, atoms_prime_prime, n_prime_prime, s_prime_prime, app, napp = buildtree(
                    pos_atoms, u, v, j - 1, e, atoms0, e0)

            if np.random.uniform() < float(
                            n_prime_prime / (max(n_prime + n_prime_prime, 1))):
                atoms_prime = atoms_prime_prime

            a_prime = a_prime + app
            na_prime = na_prime + napp

            datoms = pos_atoms.positions - neg_atoms.positions
            span = datoms.flatten()
            s_prime = s_prime_prime * (
                span.dot(neg_atoms.get_velocities().flatten()) >= 0) * (
                          span.dot(pos_atoms.get_velocities().flatten()) >= 0)
            n_prime = n_prime + n_prime_prime
        return neg_atoms, pos_atoms, atoms_prime, n_prime, s_prime, a_prime, na_prime


if __name__ == '__main__':
    from copy import deepcopy as dc
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.visualize import view
    from ase.atoms import Atoms

    from pyiid.wrappers.elasticscatter import ElasticScatter
    from pyiid.calc.pdfcalc import PDFCalc
    from pyiid.calc import wrap_rw

    ideal_atoms = Atoms('Sn2O2', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    ideal_atoms.set_masses([1, 1, 1, 1])
    # ideal_atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    start_atoms = dc(ideal_atoms)
    start_atoms.positions *= 1.02
    s = ElasticScatter()
    gobs = s.get_pdf(ideal_atoms)

    calc = PDFCalc(obs_data=gobs, scatter=s, conv=100, potential='rw')
    start_atoms.set_calculator(calc)
    print start_atoms.get_potential_energy()

    e = .8e-2
    M = 10

    traj, _, _ = nuts(start_atoms, 0.65, M, p_scale=1)

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
        f = atoms.get_forces()
        f2 = f * 2
    min_pe = np.argmin(pe_list)
    view(traj)
    print traj[-1].get_potential_energy()
    r = np.arange(0, 40, .01)
    plt.plot(r, gobs, 'b-', label='ideal')
    plt.plot(r, s.get_pdf(traj[0]) *
             wrap_rw(s.get_pdf(traj[0]), gobs)[1], 'k-',
             label='start')
    plt.plot(r, s.get_pdf(traj[-1]) *
             wrap_rw(s.get_pdf(traj[-1]), gobs)[1], 'r-',
             label='final')
    plt.legend()
    plt.show()
