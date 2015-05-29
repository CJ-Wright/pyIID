__author__ = 'christopher'
from ase.atoms import Atoms
import ase.io as aseio

from pyiid.wrappers.master_wrap import wrap_rw, wrap_pdf
from pyiid.wrappers.scatter import wrap_atoms
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.utils import build_sphere_np

import matplotlib.pyplot as plt
from pprint import pprint
import time
from copy import deepcopy as dc
from collections import OrderedDict
import pickle

atoms = Atoms('Au4', [[0,0,0], [3,0,0], [0,3,0], [3,3,0]])

wrap_atoms(atoms, exp_dict)
atoms2 = dc(atoms)
pdf = wrap_pdf(atoms, qmin=0.0, qbin=.1)


atoms.positions *= .95

# Time for Rw
gpu_3_d = []
cpu = []

pu = [
    'multi_gpu',
    'cpu'
]

cpu_sizes = []
gpu_sizes = []
cpu_f = []
cpu_e = []
multi_gpu_f = []
multi_gpu_e = []
try:
    for i in range(10, 105, 5):
        atoms = build_sphere_np('/mnt/work-data/dev/pyIID/benchmarks/1100138.cif', float(i) / 2)
        wrap_atoms(atoms, exp_dict)
        atoms.rattle()
        print len(atoms), i/10.
        calc = PDFCalc(gobs=pdf, qmin=0.0, qbin=.1, conv=1, potential='rw')
        atoms.set_calculator(calc)
        s = time.time()
        nrg = atoms.get_potential_energy()
        f = time.time()
        multi_gpu_e.append(f-s)
        s = time.time()
        force = atoms.get_forces()
        f = time.time()
        multi_gpu_f.append(f-s)
        gpu_sizes.append(i/10.)
except :
    pass
'''
try:
    for i in range(10, 100, 5):
        atoms = build_sphere_np('/mnt/work-data/dev/pyIID/benchmarks/1100138.cif', float(i) / 2)
        wrap_atoms(atoms)
        atoms.rattle()
        print len(atoms), i/10.
        calc = PDFCalc(gobs=pdf, qmin=0.0, conv=1, qbin=.1, processor='cpu', potential='rw')
        atoms.set_calculator(calc)
        s = time.time()
        nrg = atoms.get_potential_energy()
        f = time.time()
        cpu_e.append(f-s)
        s = time.time()
        force = atoms.get_forces()
        f = time.time()
        cpu_f.append(f-s)
        cpu_sizes.append(i/10.)
except :
    pass
'''
print multi_gpu_e
AAA
f_name_list = [
    # ('cpu_e.txt', cpu_e), ('cpu_f.txt', cpu_f),
    ('gpu_e.txt', multi_gpu_e), ('gpu_f.txt', multi_gpu_f)
]
for f_str, lst in f_name_list:
    with open(f_str, 'w') as f:
        pickle.dump(lst, f)
#
# plt.plot(cpu_sizes, cpu_e, 'bo', label='cpu energy')
# plt.plot(sizes, cpu_f, 'bs', label='cpu force')

plt.plot(gpu_sizes, multi_gpu_e, 'ro', label='GPU energy')
plt.plot(gpu_sizes, multi_gpu_f, 'rs', label='GPU force')
plt.legend(loc=2)
plt.xlabel('NP diameter in Angstrom')
plt.ylabel('time (s) [lower is better]')
plt.title('Scaling of algorithm')
plt.savefig('gpu_speed.eps', bbox_inches='tight', transparent=True)
plt.savefig('gpu_speed.png', bbox_inches='tight', transparent=True)
plt.show()

plt.semilogy(gpu_sizes, multi_gpu_e, 'ro', label='GPU energy')
plt.semilogy(gpu_sizes, multi_gpu_f, 'rs', label='GPU force')
plt.legend(loc=2)
plt.xlabel('NP diameter in Angstrom')
plt.ylabel('time (s) [lower is better]')
plt.title('Scaling of algorithm')
plt.savefig('gpu_speed_log.eps', bbox_inches='tight', transparent=True)
plt.savefig('gpu_speed_log.png', bbox_inches='tight', transparent=True)
plt.show()