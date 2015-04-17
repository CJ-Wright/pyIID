__author__ = 'christopher'
import sys
sys.path.extend(['/mnt/work-data/dev/pyIID'])
from pyiid.wrappers.three_d_gpu_wrap import wrap_fq, wrap_fq_grad_gpu
import ase.io as aseio
import os
from pyiid.wrappers.serial_cpu_wrap import wrap_atoms

atoms_file = '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/2-AuNP-DFT.d/SizeVariation.d/Au55.xyz'
atoms_file_no_ext = os.path.splitext(atoms_file)[0]
atomsio = aseio.read(atoms_file)
wrap_atoms(atomsio)
atomsio *= (14, 1, 1)
print len(atomsio)

# fq = wrap_fq(atomsio)
grad = wrap_fq_grad_gpu(atomsio)
# print grad