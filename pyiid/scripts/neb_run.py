__author__ = 'christopher'
import os
from copy import deepcopy as dc

import numpy as np
import ase.io as aseio
from ase.neb import NEB
from ase.optimize.bfgs import BFGS

from pyiid.wrappers.gpu_wrap import wrap_rw, wrap_pdf
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.wrappers.serial_cpu_wrap import wrap_atoms


atoms_file = '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/2-AuNP-DFT.d/SizeVariation.d/Au55.xyz'
atoms_file_no_ext = os.path.splitext(atoms_file)[0]
atomsio = aseio.read(atoms_file)
wrap_atoms(atomsio)

qmax = 25
qbin = .1
n = len(atomsio)

qmax_bin = int(qmax / qbin)
# atomsio.set_array('scatter', scatter_array)

atoms = dc(atomsio)
pdf, fq = wrap_pdf(atoms, qmin=2.5, qbin=.1)
# atoms.positions *= 1.05
atoms.positions *= .95
# atoms.rattle(.1)
rw, scale, apdf, afq = wrap_rw(atoms, pdf, qmin=2.5, qbin=.1)


calc = PDFCalc(gobs=pdf, qmin=2.5, conv=1, qbin=.1)
# atoms.set_velocities(np.random.normal(0, 1, (len(atoms), 3))/3/len(atoms))
atoms.set_momenta(np.zeros((len(atoms), 3)))

traj = [atomsio]
traj += [atomsio.copy() for i in range(30)]
traj += [atoms]

neb = NEB(traj)
neb.interpolate()
pe_list = []
a_list2 = []
for image in neb.images:
    image.set_calculator(calc)
    pe_list.append(image.get_potential_energy())
    image.get_forces()
    a_list2 += [image]


qn = BFGS(neb, trajectory=atoms_file_no_ext+'_gpu_neb_contract'+'.traj')
qn.run(fmax=0.05)
# view(traj)
# view(neb.images)