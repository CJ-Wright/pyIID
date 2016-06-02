import numpy as np
import math
from itertools import combinations
from copy import deepcopy as dc
from ase.atoms import Atoms as AAtoms
import ase.io as aseio
from pyiid.asa import calculate_asa, get_neighbor_list, get_coordination

__author__ = 'christopher'


def convert_stru_to_atoms(stru):
    symbols = []
    xyz = []
    tags = []
    for d_atom in stru:
        symbols.append(d_atom.element)
        xyz.append(d_atom.xyz)
        tags.append(d_atom.label)
    atoms = AAtoms(symbols, np.array(xyz), tags=tags)
    return atoms


def build_sphere_np(file_name, radius):
    """
    Build a spherical nanoparticle
    :param file_name: ASE loadable atomic positions
    :param radius: Radius of particle in Angstroms
    :return:
    """
    atoms = aseio.read(file_name)
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


def tag_surface_atoms(atoms, tag=1, probe=1.4, cutoff=None):
    """
    Find which are the surface atoms in a nanoparticle.

    Parameters
    ----------
    atoms: ase.atoms object
        The atomic configuration
    tag: int
        The number with which to tag the surface atoms
    probe: float, optional
        Radius of the probe molecule, default is 1.4 A the radius of water
    cutoff: float
        Bond cutoff, defaults to van der Waals radius
    """
    calculate_asa(atoms, probe, tag=tag, cutoff=cutoff)


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

    Parameters
    ----------
    atoms: ase.Atoms objecct
        The atomic configuration
    cutoff: float
        Bond length cutoff
    element: str, optional
        Limit the list to only this element
    tag: int
        Limt the list to only this tag

    Returns
    -------
    ndarray:
        The list of bond angles in degrees
    """
    n_list = list(get_neighbor_list(cutoff, atoms))
    angles = []
    for i in range(len(atoms)):
        z = list(combinations(n_list[i], 2))
        for a in z:
            if (element is not None and atoms[i].symbol != element) or \
                    (tag is not None and atoms[i].tag != tag):
                break
            angles.append(np.rad2deg(atoms.get_angle([a[0], i, a[1]])))
    return np.nan_to_num(np.asarray(angles))


def get_coord_list(atoms, cutoff, element=None, tag=None):
    """
    Get all the angles in the NP

    Parameters
    ----------
    atoms: ase.Atoms objecct
        The atomic configuration
    cutoff: float
        Bond length cutoff
    element: str, optional
        Limit the list to only this element
    tag: int
        Limt the list to only this tag

    Returns
    -------
    ndarray:
        The list of coordination nubmers
    """
    if isinstance(atoms, list):
        coord_l = []
        for atms in atoms:
            a = get_coordination(cutoff, atms)
            if element is not None and tag is not None:
                coord_l.append(
                    a[(np.asarray(atoms.get_chemical_symbols()) == element) &
                      (atoms.get_tags() == tag)])
            elif element is not None:
                coord_l.append(
                    a[np.asarray(atoms.get_chemical_symbols()) == element])
            elif tag is not None:
                coord_l.append(a[atoms.get_tags() == tag])
            else:
                coord_l.append(a)
        c = np.asarray(coord_l)
        return np.average(c, axis=0), np.std(c, axis=0)

    else:
        a = get_coordination(cutoff, atoms)
        if element is not None and tag is not None:
            return a[(np.asarray(atoms.get_chemical_symbols()) == element) &
                     (atoms.get_tags() == tag)]
        elif element is not None:
            return a[np.asarray(atoms.get_chemical_symbols()) == element]
        elif tag is not None:
            return a[atoms.get_tags() == tag]
        else:
            return a


def get_bond_dist_list(atoms, cutoff, element=None, tag=None):
    """
    Get all the angles in the NP

    Parameters
    ----------
    atoms: ase.Atoms objecct
        The atomic configuration
    cutoff: float
        Bond length cutoff
    element: str, optional
        Limit the list to only this element
    tag: int
        Limt the list to only this tag

    Returns
    -------
    ndarray:
        The list of bond distances
    """
    n_list = list(get_neighbor_list(cutoff, atoms))
    bonds = []
    for i in range(len(atoms)):
        for a in n_list[i]:
            if (element is not None and atoms[i].symbol != element) or \
                    (tag is not None and atoms[i].tag != tag):
                break
            bonds.append(atoms.get_distance(i, a))
    return np.nan_to_num(np.asarray(bonds))


def onion_tag(atoms, probe=1.4, cutoff=None):
    tags = np.zeros(len(atoms))
    tag_counter = 0
    loop_atoms = atoms.copy()

    remaining = np.arange(len(atoms))
    finished_atoms = []

    while len(atoms) >= len(loop_atoms):
        if len(loop_atoms) == 1:
            tags[remaining] = tag_counter
            break
        # Get me all the indexes of the surface atoms in the current config
        _, _, surface_atom_idxs = calculate_asa(loop_atoms, probe=probe)

        surface_atom_idxs = np.asarray(surface_atom_idxs)

        # remaining acts as a lookup table for the original idx
        for sidx in surface_atom_idxs:
            tags[remaining[sidx]] = tag_counter

        # Add the surface atoms to the finished atoms
        finished_atoms.extend(remaining[surface_atom_idxs].tolist())

        # Remove the surface atoms
        del loop_atoms[surface_atom_idxs]

        # The remaining atoms idx are the ones not in finished atoms
        remaining = np.asarray(
            [r for r in remaining.tolist() if r not in finished_atoms])

        tag_counter += 1
    atoms.set_tags(tags)
