__author__ = 'christopher'
from ase.atoms import Atoms
import ase.io as aseio

from pyiid.wrappers.serial_cpu_wrap import wrap_atoms
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.wrappers.multi_gpu_wrap import wrap_rw, wrap_pdf
from pyiid.utils import build_sphere_np

import matplotlib.pyplot as plt
from pprint import pprint
import time
from copy import deepcopy as dc
from collections import OrderedDict

atoms = Atoms('Au4', [[0,0,0], [3,0,0], [0,3,0], [3,3,0]])

wrap_atoms(atoms)
atoms2 = dc(atoms)
pdf, fq = wrap_pdf(atoms, qmin=0.0, qbin=.1)


atoms.positions *= .95

# Time for Rw
gpu_3_d = []
gpu_2_d = []
cpu = []

pu = [
    'multi_gpu',
    'cpu'
]
try:
    super_results_d = OrderedDict()
    for i in range(10, 50, 5):
        atoms = build_sphere_np('/mnt/work-data/dev/pyIID/benchmarks/1100138.cif', float(i) / 2)
        wrap_atoms(atoms)
        atoms.rattle()
        print len(atoms), i/10.
        results_d = {}
        for processor in pu:
            sub_d = {}
            calc = PDFCalc(gobs=pdf, qmin=0.0, conv=1, qbin=.1, processor=processor, potential='rw')
            atoms.set_calculator(calc)
            s = time.time()
            nrg = atoms.get_potential_energy()
            f = time.time()
            sub_d['time for energy'] = f-s
            sub_d['energy'] = nrg
            s = time.time()
            force = atoms.get_forces()
            f = time.time()
            sub_d['time for force'] = f-s
            # sub_d['force'] = force
            results_d[processor] = sub_d
            atoms._del_calculator
        super_results_d[i/10.] = results_d
except KeyboardInterrupt:
    pass

# pprint(super_results_d)

sizes = []
cpu_f = []
cpu_e = []
multi_gpu_f = []
multi_gpu_e = []

for key, value in super_results_d.iteritems():
    sizes.append(key)
    cpu_f.append(value['cpu']['time for force'])
    cpu_e.append(value['cpu']['time for energy'])

    multi_gpu_e.append(value['multi_gpu']['time for energy'])
    multi_gpu_f.append(value['multi_gpu']['time for force'])

# print sizes, gpu3_d_e

plt.plot(sizes, cpu_f, 'bo', label='cpu energy')
plt.plot(sizes, cpu_e, 'bs', label='cpu force')

plt.plot(sizes, multi_gpu_e, 'ro', label='GPU energy')
plt.plot(sizes, multi_gpu_f, 'rs', label='GPU force')
plt.legend(loc=2)
plt.xlabel('NP size in nm')
plt.ylabel('time (s) [lower is better]')
plt.title('Scaling of algorithm')
plt.savefig('speed.eps', bbox_inches='tight', transparent=True)
plt.show()