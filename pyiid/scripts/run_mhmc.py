__author__ = 'christopher'
import time

import ase.io as io
from ase.visualize import view
import matplotlib.pyplot as plt
import numpy as np
from pyiid.utils import load_gr_file
from pyiid.potential_core import Debye_srreal_U
from pyiid.alg import MHMC
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from pyiid.utils import convert_atoms_to_stru


r, gr = load_gr_file('/home/christopher/shared/PDFdata/July/X17A/1/FinalSum_1v2.gr')
atoms = io.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')

i = 0
t0 = time.clock()

move_list =[]

current_atoms = atoms[:]
new_atoms = current_atoms
traj =[new_atoms]

dpc = DebyePDFCalculator()
dpc.qmax = 25
dpc.rmin = 0.00
dpc.rmax = 40.01

gcalc = dpc(convert_atoms_to_stru(new_atoms), qmin=2.5)[1]
scale = np.max(gcalc)/np.max(gr)
print scale
gr = gr*scale
# plt.plot(r, gcalc, r, gr), plt.show()

current_U = Debye_srreal_U(new_atoms, gr)
print current_U
u_list =[current_U]
while i < 200:
    new_atoms, move_type, current_U = MHMC(new_atoms,  Debye_srreal_U,
                                     current_U, gr, 5, .01)
    traj += [new_atoms]
    move_list.append(move_type)
    u_list.append(current_U)
    # print i, current_U, move_type
    i += 1
t1 = time.clock()

print 'time = ', (t1-t0), ' sec'
print u_list[-1]
dist_from_crystal = []
q_crystal = current_atoms.get_positions()

for atoms in traj:
    q1 = atoms.get_positions()
    # print q1
    W = q1-q_crystal
    # print W
    d = 0
    for element in W:
        d += np.dot(element,element)
    d = np.sqrt(d)
    dist_from_crystal.append(d)
x_line = np.arange(i+1)
# print '\n\n\n'
# print len(x_line), len(dist_from_crystal), len(u_list)
plt.plot(x_line, dist_from_crystal, 'k', x_line, u_list, 'r'), plt.show()
baseline = np.min(gr)*1.1

dpc = DebyePDFCalculator()
dpc.qmax = 25
dpc.rmin = 0.00
dpc.rmax = 40.01

gcalc = dpc(convert_atoms_to_stru(traj[-1]), qmin=2.5)[1]
gdiff = gr-gcalc


plt.figure()
plt.plot(r, gr, 'bo', label="G(r) data")
plt.plot(r, gcalc, 'r-', label="G(r) fit")
plt.plot(r, gdiff + baseline, 'g-', label="G(r) diff")
plt.plot(r, np.zeros_like(r) + baseline, 'k:')
plt.xlabel(r"$r (\AA)$")
plt.ylabel(r"$G (\AA^{-2})$")
plt.legend()

plt.show()
view(traj)