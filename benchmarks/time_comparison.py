__author__ = 'christopher'
from ase.atoms import Atoms
import ase.io as aseio

from pyiid.calc.calc_1d import Calc1D
from pyiid.utils import build_sphere_np

import matplotlib.pyplot as plt
import time
from copy import deepcopy as dc
from pyiid.experiments.elasticscatter import ElasticScatter
import numpy as np

scat = ElasticScatter()
atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
pdf = scat.get_pdf(atoms)

type_list = []
time_list = []
benchmarks = [
    ('GPU', 'flat'),
    ('CPU', 'flat'),
]
sizes = np.arange(10, 55, 5)
pes_speed = None
fq_speed = None

atoms_list = [
    build_sphere_np('/mnt/work-data/dev/pyIID/benchmarks/1100138.cif',
                    float(i) / 2) for i in sizes]
number_of_atoms = np.asarray([len(a) for a in atoms_list])
print(sizes)
print(number_of_atoms)
hdr = ''

# Prep everything so we don't need to recompute their scattering power
_ = [scat._wrap_atoms(a) for a in atoms_list]

for proc, alg in benchmarks:
    print(proc, alg)

    scat.set_processor(proc, alg)
    type_list.append((proc, alg))
    nrg_l = []
    f_l = []
    fq_l = []
    g_fq_l = []
    for atoms, s in zip(atoms_list, sizes):
        print(len(atoms), s)
        calc = Calc1D(target_data=pdf, exp_function=scat.get_pdf,
                      exp_grad_function=scat.get_grad_pdf)
        atoms.set_calculator(calc)

        print('start grad fq')
        atoms.rattle()
        s = time.time()
        scat.get_grad_fq(atoms)
        f = time.time()
        g_fq_l.append(f - s)

        print('start forces')
        atoms.rattle()
        s = time.time()
        force = atoms.get_forces()
        # scat.get_grad_fq(atoms)
        f = time.time()
        f_l.append(f - s)

        print('start fq')
        atoms.rattle()
        s = time.time()
        scat.get_fq(atoms)
        f = time.time()
        fq_l.append(f - s)

        print('start potential energy')
        atoms.rattle()
        s = time.time()
        nrg = atoms.get_potential_energy()
        f = time.time()
        nrg_l.append(f - s)

    hdr += ', {0}_energy, {0}_forces'.format(proc)
    if pes_speed is None:
        pes_speed = np.vstack((np.asarray(nrg_l), np.asarray(f_l)))
        fq_speed = np.vstack((np.asarray(fq_l), np.asarray(g_fq_l)))
    else:
        pes_speed = np.vstack((pes_speed, np.asarray(nrg_l), np.asarray(f_l)))
        fq_speed = np.vstack((fq_speed, np.asarray(fq_l), np.asarray(g_fq_l)))

names = ['GPU', 'GPU', 'CPU', 'CPU']
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

colors = ['b', 'b', 'r', 'r']
lines = ['o', 's'] * 2
calc_type = ['energy', 'force'] * 2
for i in range(len(names)):
    ax1.semilogy(sizes, pes_speed[i], color=colors[i], marker=lines[i],
                 label='{0} {1}'.format(names[i], calc_type[i]))

ax1.legend(loc='best')
ax1.set_xlabel('NP diameter in Angstrom')
ax1.set_ylabel('Elapsed running time (s)')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(number_of_atoms)
ax2.set_tick_params(which='major', pad=20)
ax2.set_xlabel('Number of Atoms')
# plt.savefig('/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/new_figures/speed_log.eps', bbox_inches='tight', transparent=True)
# plt.savefig('/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/new_figures/speed_log.png', bbox_inches='tight', transparent=True)
plt.show()

calc_type = ['F(Q)', 'Grad F(Q)'] * 2
for i in range(len(names)):
    ax1.semilogy(sizes, fq_speed[i], color=colors[i], marker=lines[i],
                 label='{0} {1}'.format(names[i], calc_type[i]))

ax1.legend(loc='best')
ax1.set_xlabel('NP diameter in Angstrom')
ax1.set_ylabel('Elapsed running time (s)')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(number_of_atoms)
ax2.set_xlabel('Number of Atoms')
# plt.savefig('/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/new_figures/speed_log.eps', bbox_inches='tight', transparent=True)
# plt.savefig('/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/new_figures/speed_log.png', bbox_inches='tight', transparent=True)
plt.show()

# '''
pes_speed = pes_speed.T
fq_speed = fq_speed.T
np.savetxt(
    '/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/new_figures/sizes_speed.txt',
    sizes)
np.savetxt(
    '/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/new_figures/n_atoms_speed.txt',
    number_of_atoms)
np.savetxt(
    '/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/new_figures/pes_speed.txt',
    pes_speed, header=hdr)
np.savetxt(
    '/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/new_figures/fq_speed.txt',
    fq_speed, header=hdr)
# '''
