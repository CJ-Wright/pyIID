from ase.atoms import Atoms
import ase.io as aseio

from pyiid.experiments.elasticscatter import wrap_atoms
from pyiid.calc.calc_1d import Calc1D
from pyiid.utils import build_sphere_np

import matplotlib.pyplot as plt
from pprint import pprint
import time
from copy import deepcopy as dc
from collections import OrderedDict
import pickle
import traceback
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.tests import *
__author__ = 'christopher'

exp = None
scat = ElasticScatter(
    # verbose=True
)
atoms = Atoms('Au4', [[0,0,0], [3,0,0], [0,3,0], [3,3,0]])
pdf = scat.get_pdf(atoms)

type_list = []
time_list = []
# benchmarks = [
#     ('CPU', 'flat'),
#     ('Multi-GPU', 'flat')
# ]
benchmarks = proc_alg_pairs
colors=['b', 'r', 'k', 'g']
sizes = range(10, 45, 5)
print sizes, benchmarks
for proc, alg in benchmarks:
    print proc, alg
    number_of_atoms = []
    scat.set_processor(proc, alg)
    type_list.append((proc, alg))
    nrg_l = []
    f_l = []
    try:
        for i in sizes:
            atoms = build_sphere_np('/mnt/work-data/dev/pyIID/benchmarks/1100138.cif', float(i) / 2)
            atoms.rattle()
            print len(atoms), i/10.
            number_of_atoms.append(len(atoms))
            calc = Calc1D(target_data=pdf, exp_function=scat.get_pdf,
                          exp_grad_function=scat.get_grad_pdf,
                          exp_voxel_function=scat.get_pdf_voxels,
                          conv=1, potential='rw')
            atoms.set_calculator(calc)

            s = time.time()
            # nrg = atoms.get_potential_energy()
            scat.get_fq(atoms)
            f = time.time()

            nrg_l.append(f-s)

            s = time.time()
            # force = atoms.get_forces()
            scat.get_grad_fq(atoms)
            f = time.time()
            f_l.append(f-s)
    except:
        print traceback.format_exc()
        break
    time_list.append((nrg_l, f_l))

# for i in range(len(benchmarks)):
#     for j, calc_type in enumerate(['energy', 'force']):
#         f_str = benchmarks[i][0]+'_'+benchmarks[i][0]+'_'+calc_type+'.pkl'
#         with open(f_str, 'w') as f:
#             pickle.dump(time_list[i][j], f)
'''
for i in range(len(benchmarks)):
    for j, (calc_type, line) in enumerate(zip(['energy', 'force'], ['o', 's'])):
        plt.plot(sizes,time_list[i][j], color=colors[i], marker=line, label= '{0} {1}'.format(benchmarks[i][0], calc_type))
plt.legend(loc='best')
plt.xlabel('NP diameter in Angstrom')
plt.ylabel('time (s) [lower is better]')
plt.savefig('speed3.eps', bbox_inches='tight', transparent=True)
plt.savefig('speed3.png', bbox_inches='tight', transparent=True)
plt.show()
'''
names = [a + ' ' + b for (a, b) in benchmarks]
print len(names)
print names
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

for i in range(len(benchmarks)):
    for j, (calc_type, line) in enumerate(zip(['energy', 'force'], ['o', 's'])):
        ax1.semilogy(sizes,time_list[i][j], color=colors[i], marker=line, label= '{0} {1}'.format(names[i], calc_type))

ax1.legend(loc='best')
ax1.set_xticks(sizes)
ax1.set_xlabel('NP diameter in Angstrom')
ax1.set_ylabel('Elapsed running time (s)')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(number_of_atoms)
ax2.set_xlabel('Number of Atoms')
# plt.savefig('/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/figures/speed_log.eps', bbox_inches='tight', transparent=True)
# plt.savefig('/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/figures/speed_log.png', bbox_inches='tight', transparent=True)
plt.show()