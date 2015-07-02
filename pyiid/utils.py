__author__ = 'christopher'

import numpy as np
# from diffpy.Structure.structure import Structure
# from diffpy.Structure.atom import Atom as dAtom
from ase.atoms import Atoms as AAtoms
import ase.io as aseio
from ase.lattice.surface import add_adsorbate
from asap3.analysis.particle import FullNeighborList, CoordinationNumbers, \
    GetLayerNumbers
from itertools import combinations

import math
# import tkFileDialog
from pyiid.kernels.cpu_kernel import get_d_array, get_r_array
from copy import deepcopy as dc
import time
import datetime

'''
def convert_atoms_to_stru(atoms):
    """
    Convert between ASE and Diffpy structural objects

    Parameters:
    -----------
    atoms: ase.Atoms object

    Return:
    diffpy.Structure object:
    """
    diffpy_atoms = []
    symbols = atoms.get_chemical_symbols()
    q = atoms.get_positions()
    tags = atoms.get_tags()
    for symbol, xyz, tag, in zip(symbols, q, tags):
        d_atom = dAtom(symbol, xyz=xyz,
                       label=tag, occupancy=1)
        diffpy_atoms.append(d_atom)
    stru = Structure(diffpy_atoms)
    return stru


def update_stru(new_atoms, stru):
    aatomq = new_atoms.get_positions()
    datomq = np.reshape([datom.xyz for datom in stru], (len(new_atoms), 3))
    # aatome = new_atoms.get_chemical_symbols()
    # datome = np.array([datom.element for datom in stru])
    changedq = np.in1d(aatomq, datomq).reshape((len(new_atoms), 3))

    changed_array = np.sum(changedq, 1) != 3
    stru[changed_array].xyz = new_atoms[changed_array].get_positions()
    # for i in len(changed_array):
    #     if changed_array[i] == True:
    #         stru[i]._set_xyz_cartn(new_atoms[i].position)
    # changed_list = []
    # for i in len(new_atoms):
    #     if np.sum(changedq[i, :]) != 3:
    #         changed_list.append(i)
    # for j in changed_list:
    #     stru[j]._set_xyz_cartn(new_atoms[j].position)
    return stru
'''

def load_gr_file(gr_file=None, skiplines=None, rmin=None, rmax=None, **kwargs):
    """
    Load gr files produced from PDFgetx3
    """
    exp_keys = ['qmax', 'qbin', 'qmin', 'rmin', 'rmax', 'rstep']
    # TODO: also give back the filename
    if skiplines is None:
        with open(gr_file) as f:
            exp_dict = {}
            lines = f.readlines()
            record = False
            for num, line in enumerate(lines, 1):
                if "# End of config" in line:
                    record = False
                if record is True and line.startswith(tuple(exp_keys)):
                    # print 'line = ', line
                    key, val = line.split(' = ')
                    if key in exp_keys:
                        exp_dict[key] = float(val)
                if '# PDF calculation setup' in line:
                    record = True
                if '#### start data' in line:
                    skiplines = num + 2
                    break
    data = np.loadtxt(gr_file, skiprows=skiplines)
    r = data[:-1, 0]
    gr = data[:-1, 1]
    if rmax is not None:
        r = r[:math.ceil(rmax / exp_dict['rstep'])]
        gr = gr[:math.ceil(rmax / exp_dict['rstep'])]
        exp_dict['rmax'] = rmax

    if rmin is not None:
        r = r[math.ceil(rmin / exp_dict['rstep']):]
        gr = gr[math.ceil(rmin / exp_dict['rstep']):]
        exp_dict['rmin'] = rmin

    return r, gr, exp_dict


def convert_stru_to_atoms(stru):
    symbols = []
    xyz = []
    tags = []
    for d_atom in stru:
        symbols.append(d_atom.element)
        xyz.append(d_atom.xyz)
        tags.append(d_atom.label)
    # print symbols
    # print np.array(xyz)
    # print tags
    atoms = AAtoms(symbols, np.array(xyz), tags=tags)
    return atoms


def build_sphere_np(file, radius):
    """
    Build a spherical nanoparticle
    :param file: ASE loadable atomic positions
    :param radius: Radius of particle in Angstroms
    :return:
    """
    atoms = aseio.read(file)
    cell_dist = atoms.get_cell()
    multiple = np.ceil(2 * radius / cell_dist.diagonal()).astype(int)
    atoms = atoms.repeat(multiple)
    com = atoms.get_center_of_mass()
    atoms.translate(-com)
    del atoms[[atom.index for atom in atoms
               if np.sqrt(np.dot(atom.position, atom.position)) >=
               np.sqrt(radius ** 2)]]
    atoms.center()
    atoms.set_pbc((False, False, False))
    return atoms


def tag_surface_atoms(atoms, tag=1, tol=0.5):
    """
    Find which are the surface atoms in a nanoparticle.
    ..note: We define the surface as a collection of atoms for which the
    dot product between the displacement from the center of mass and all
    the interatomic distances are negative.  Thus, all the interatomic
    displacement vectors point inward or perpendicular.  This only works for
    true crystals so we will include some tolerance to account for surface
    disorder.
    :param atoms:
    :return:
    """
    '''
    n = len(atoms)
    d = np.zeros((n, n, 3))
    q = atoms.positions
    get_d_array(d, q)
    r = np.zeros((n, n))
    get_r_array(r, d)
    for i in range(n):
        r[i,i] = np.inf
        r[i,i] = np.inf
        r[i,i] = np.inf

    rmin = np.min(r)
    print rmin
    s_idx = GetLayerNumbers(atoms, rmin)
    for i in range(len(atoms)):
        if s_idx[i] == 1:
            atoms[i].tag = tag
    '''
    n = len(atoms)
    q = atoms.get_positions()
    d_array = np.zeros((n, n, 3))
    get_d_array(d_array, q)
    com = atoms.get_center_of_mass()
    for i in range(n):
        pos = atoms[i].position
        disp = pos - com
        disp /= np.linalg.norm(disp)
        dot = np.zeros(n)
        for j in range(n):
            if j != i:
                dot[j] = np.dot(disp, d_array[i, j, :]/np.linalg.norm(d_array[i, j, :]))
        if np.all(np.less_equal(dot, np.zeros(len(dot))+tol)):
            atoms[i].tag = tag
    # '''


def add_ligands(ligand, surface, distance, coverage, head, tail):
    atoms = dc(surface)
    tag_surface_atoms(atoms)
    for atom in atoms:
        if atom.tag == 1 and np.random.random() < coverage:
            pos = atom.position
            com = surface.get_center_of_mass()
            disp = pos - com
            norm_disp = disp / np.sqrt(np.dot(disp, disp))
            l_length = ligand[tail].position - ligand[head].position
            norm_l_length = l_length / np.sqrt(np.dot(l_length, l_length))
            ads = dc(ligand)
            ads.rotate(norm_l_length, a=norm_disp)
            ads.translate(-ads[head].position)
            ads.translate(pos + distance * norm_disp)
            atoms += ads
    return atoms


def get_angle_list(atoms, cutoff, element=None, tag=None):
    """
    Get all the angles in the NP
    :param atoms:
    :param cutoff:
    :return:
    """
    n_list = list(FullNeighborList(cutoff, atoms))
    angles = []
    for i in range(len(atoms)):
        z = list(combinations(n_list[i], 2))
        for a in z:
            if (element is not None and atoms[i].symbol != element) or \
                    (tag is not None and atoms[i].tag != tag):
                break
            angles.append(np.rad2deg(atoms.get_angle([a[0], i, a[1]])))
    return np.nan_to_num(np.asarray(angles))


def time_est(atoms, HD_iter, HMC_iter):
    """
    Estimate the amount of time to complete a simulation
    :param atoms:
    :param HD_iter:
    :param HMC_iter:
    :return:
    """
    s = time.time()
    nrg = atoms.get_potential_energy()
    f = time.time()
    nrg_t = f - s
    s = time.time()
    force = atoms.get_forces()
    f = time.time()
    ft = f - s
    total = HD_iter * ft * HMC_iter + nrg_t * HMC_iter
    print str(datetime.timedelta(seconds=math.ceil(total)))
    print 'finished by' + \
          str(datetime.datetime.today() +
              datetime.timedelta(seconds=math.ceil(total)))
    return total


def get_coord_list(atoms, cutoff, element=None, tag=None):
    """
    Get all the angles in the NP
    :param atoms:
    :param cutoff:
    :return:
    """
    if isinstance(atoms, list):
        coord_l = []
        for atms in atoms:
            a = CoordinationNumbers(atms, cutoff)
            if element is not None and tag is not None:
                coord_l.append(a[(np.asarray(atoms.get_chemical_symbols()) == element)
                         & (atoms.get_tags() == tag)])
            elif element is not None:
                coord_l.append(a[np.asarray(atoms.get_chemical_symbols()) == element])
            elif tag is not None:
                coord_l.append(a[atoms.get_tags() == tag])
            else:
                coord_l.append(a)
        c = np.asarray(coord_l)
        return np.average(c, axis=0), np.std(c, axis=0)

    else:
        a = CoordinationNumbers(atoms, cutoff)
        if element is not None and tag is not None:
            return a[(np.asarray(atoms.get_chemical_symbols()) == element)
                     & (atoms.get_tags() == tag)]
        elif element is not None:
            return a[np.asarray(atoms.get_chemical_symbols()) == element]
        elif tag is not None:
            return a[atoms.get_tags() == tag]
        else:
            return a


def get_bond_dist_list(atoms, cutoff, element=None, tag=None):
    """
    Get all the angles in the NP
    :param atoms:
    :param cutoff:
    :return:
    """
    n_list = list(FullNeighborList(cutoff, atoms))
    bonds = []
    for i in range(len(atoms)):
        for a in n_list[i]:
            if (element is not None and atoms[i].symbol != element) or \
                    (tag is not None and atoms[i].tag != tag):
                break
            bonds.append(atoms.get_distance(i, a))
    return np.nan_to_num(np.asarray(bonds))
