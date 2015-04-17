__author__ = 'christopher'
import os
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
from ase.io.trajectory import PickleTrajectory
import ase.io as aseio

from pyiid.sim.hmc import simulate_dynamics
from pyiid.wrappers.gpu_wrap import wrap_rw, wrap_pdf
from pyiid.calc.pdfcalc_gpu import PDFCalc
from pyiid.wrappers.serial_cpu_wrap import wrap_atoms


atoms_file = '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/2-AuNP-DFT.d/SizeVariation.d/Au55.xyz'
atoms_file_no_ext = os.path.splitext(atoms_file)[0]
atomsio = aseio.read(atoms_file)
# scatter_array = np.loadtxt('/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/c60_scat.txt', dtype=np.float32)

wrap_atoms(atomsio)

qmax = 25
qbin = .1
n = len(atomsio)

qmax_bin = int(qmax / qbin)
# atomsio.set_array('scatter', scatter_array)

atoms = dc(atomsio)
pdf, fq = wrap_pdf(atoms, qmin=2.5, qbin=.1)
# plt.plot(pdf)
atoms.positions *= 1.05
# atoms.positions *= .95
# atoms.rattle(.1)
rw, scale, apdf, afq = wrap_rw(atoms, pdf, qmin=2.5, qbin=.1)
# view(atoms)
# print scale
# plt.plot(apdf*scale)
# plt.show()

calc = PDFCalc(gobs=pdf, qmin=2.5, conv=1, qbin=.1)
atoms.set_calculator(calc)
rwi = atoms.get_potential_energy()
print rwi
# atoms.set_velocities(np.zeros((len(atoms), 3)))
atoms.set_velocities(np.random.normal(0, 1, (len(atoms), 3))/3/len(atoms))
# atoms.set_velocities(np.random.normal(0, 1, (len(atoms), 3))/3)
# atoms.set_velocities(np.random.normal(0, 1, (len(atoms), 3)))

pe_list = []
traj = simulate_dynamics(atoms, .05, 100)
for atoms in traj:
    pe_list.append(atoms.get_potential_energy())
# print((rwi - traj[-1].get_potential_energy()))

wtraj = PickleTrajectory(atoms_file_no_ext+'_gpu_simulate_rattle'+'.traj', 'w')
for atoms in traj:
    wtraj.write(atoms)
# plt.plot(pe_list)
# plt.show()

nuts = []
for atoms in traj:
    dth = atoms.positions - traj[0].positions
    dthf = dth.flatten()
    r = atoms.get_velocities()
    rf = r.flatten()
    nuts.append(np.dot(dthf.T, rf))

plt.plot(nuts)
plt.show()
min = np.argmin(pe_list)
print(min, pe_list[min])

r = np.arange(0, 40, .01)
view(traj)
plt.plot(r, pdf, label='ideal')
plt.plot(r, wrap_pdf(traj[min], qmin= 2.5)[0], label='best')
plt.plot(r, wrap_pdf(traj[0], qmin= 2.5)[0], 'k', label='start')
plt.legend()
plt.xlabel('r (A)')
plt.show()
# '''