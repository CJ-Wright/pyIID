__author__ = 'christopher'
import os
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
from ase.io.trajectory import PickleTrajectory
import ase.io as aseio

from pyiid.sim.hmc import run_hmc
from pyiid.wrappers.serial_cpu_wrap import wrap_rw, wrap_pdf
from pyiid.calc.pdfcalc import PDFCalc


atoms_file = '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/C60.xyz'
atoms_file_no_ext = os.path.splitext(atoms_file)[0]
atomsio = aseio.read(atoms_file)
scatter_array = np.loadtxt('/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/c60_scat.txt', dtype=np.float32)

qmax = 25
qbin = .1
n = len(atomsio)

qmax_bin = int(qmax / qbin)
atomsio.set_array('scatter', scatter_array)

atoms = dc(atomsio)
pdf, fq = wrap_pdf(atoms, qmin=2.5, qbin=.1)
# plt.plot(pdf)
# plt.show()
iatoms = dc(atoms)
atoms.positions *= 1.001
fatoms = dc(atoms)
total_move = fatoms.get_positions() - iatoms.get_positions()
fatoms.set_velocities(-total_move)
# view(iatoms)
print'fatoms'
view(fatoms)
rw, scale, apdf, afq = wrap_rw(atoms, pdf, qmin=2.5, qbin=.1)

calc = PDFCalc(gobs=pdf, qmin=2.5, conv=1, qbin=.1)
atoms.set_calculator(calc)
rwi = atoms.get_potential_energy()
print(rwi)
atoms.set_velocities(np.zeros((len(atoms), 3)))

pe_list = []
traj, accept_list, move_list = run_hmc(atoms, 100, .1, 10, 0.9, 0, .9,
                                       1.02, .98, .001, .65)

for atoms in traj:
    pe_list.append(atoms.get_potential_energy())
print(rwi - traj[-1].get_potential_energy())

wtraj = PickleTrajectory(atoms_file_no_ext+'_cpu_hmc_dilate'+'.traj', 'w')
for atoms in traj:
    wtraj.write(atoms)
alp = 500
# '''
i = 0

plt.plot(pe_list)
plt.show()