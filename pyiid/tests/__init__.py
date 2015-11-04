"""
Note there is only one CPU nxn comparison test, the CPU nxn code is
rather slow, thus we test it against the flattened Multi core CPU code,
which is much faster.
Then we run all nuts_benchmarks against the CPU flat kernels.
Thus it is imperative that the flat CPU runs with no errors.
"""
import numpy as np
from ase.atoms import Atoms
from numpy.testing import assert_allclose
from itertools import *
import os
from copy import deepcopy as dc
import random
from pyiid.testing.decorators import *

from pyiid.experiments.elasticscatter import wrap_atoms
from pyiid.calc.spring_calc import Spring
from pyiid.adp import ADP

srfit = False
try:
    from diffpy.Structure.structure import Structure
    from diffpy.Structure.atom import Atom as dAtom
    from diffpy.srreal.pdfcalculator import DebyePDFCalculator

    srfit = True
except:
    pass
__author__ = 'christopher'

if srfit:
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


def setup_atoms(n):
    """
    Generate a configuration of n gold atoms with random positions
    """
    q = np.random.random((n, 3)) * 10
    atoms = Atoms('Au' + str(int(n)), q)
    return atoms


def setup_double_atoms(n):
    """
    Generate two configuration of n gold atoms with random positions
    """
    q = np.random.random((n, 3)) * 10
    atoms = Atoms('Au' + str(int(n)), q)

    q2 = np.random.random((n, 3)) * 10
    atoms2 = Atoms('Au' + str(int(n)), q2)
    return atoms, atoms2


def generate_experiment():
    """
    Generate elastic scattering experiments which are reasonable but random
    """
    exp_dict = {}
    exp_keys = ['qmin', 'qmax', 'qbin', 'rmin', 'rmax', 'rstep']
    exp_ranges = [(0, 1.5), (19., 25.), (.8, .12), (0., 2.5), (30., 50.),
                  (.005, .015)]
    for n, k in enumerate(exp_keys):
        exp_dict[k] = np.random.uniform(exp_ranges[n][0], exp_ranges[n][1])
    exp_dict['sampling'] = random.choice(['full', 'ns'])
    return exp_dict


def setup_atomic_square():
    """
    Setup squares of 4 gold atoms with known positions
    :return:
    """
    atoms1 = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    atoms2 = atoms1.copy()
    scale = .75
    atoms2.positions *= scale
    return atoms1, atoms2


def stats_check(ans1, ans2, rtol=1e-7, atol=0):
    if np.any(np.abs(ans2 - ans1) >= atol + rtol * np.abs(ans2)):
        print 'bulk statistics:'
        print 'max', np.max(np.abs(ans2 - ans1)),
        print 'min', np.min(np.abs(ans2 - ans1)),
        print 'men', np.mean(np.abs(ans2 - ans1)),
        print 'med', np.median(np.abs(ans2 - ans1)),
        print 'std', np.std(np.abs(ans2 - ans1))

        if isinstance(ans1, type(np.asarray([1]))):
            print 'normalized max', np.max(np.abs(ans2 - ans1)) / ans2[
                np.unravel_index(np.argmax(np.abs(ans2 - ans1)), ans2.shape)]
            fails = np.where(np.abs(ans1 - ans2) >= atol + rtol * np.abs(ans2))

            print 'percentage of failed tests', ans1[fails].size / float(
                ans1.size) * 100., '%'
            if ans1[fails].size <= 251:
                print '\n allclose failures'
                print zip(ans1[fails].tolist(), ans2[fails].tolist())
                print '\n allclose internals'
                print zip(np.abs(ans1[fails] - ans2[fails]).tolist(),
                          (atol + rtol * np.abs(ans2[fails])).tolist())
            else:
                print 'large number of failed tests'
            print '\n', 'without atol rtol = ', '\n'
            print np.abs(ans1[fails] - ans2[fails]) / np.abs(ans2[fails])
            print np.nanmax(
                np.abs(ans1[fails] - ans2[fails]) / np.abs(ans2[fails]))
            print 'without rtol atol = ', '\n'
            print np.abs(ans1[fails] - ans2[fails])
            print np.nanmax(np.abs(ans1[fails] - ans2[fails]))
            print '\n', 'with current atol rtol = ', '\n'
            print (np.abs(ans1[fails] - ans2[fails]) - atol) / np.abs(
                ans2[fails])
            print np.nanmax((np.abs(ans1[fails] - ans2[fails]) - atol) / np.abs(
                ans2[fails]))
            print 'with current rtol atol = ', '\n'
            print np.abs(ans1[fails] - ans2[fails]) - rtol * np.abs(
                ans2[fails])
            print np.nanmax(
                np.abs(ans1[fails] - ans2[fails]) - rtol * np.abs(ans2[fails]))
        else:
            print np.abs(ans1 - ans2)
            print atol + rtol * np.abs(ans2)
            print 'without rtol atol ='
            print rtol * np.abs(ans2)
            print 'without atol rtol =', np.abs(ans1 - ans2) / np.abs(ans2)
            print '\n', 'with current atol rtol = ', '\n'
            print (np.abs(ans1 - ans2) - atol) / np.abs(ans2)
            print np.max((np.abs(ans1 - ans2) - atol) / np.abs(ans2))
            print 'with current rtol atol = ', '\n'
            print np.abs(ans1 - ans2) - rtol * np.abs(ans2)
            print np.max(np.abs(ans1 - ans2) - rtol * np.abs(ans2))
        return False
    else:
        return True


# Setup lists of test variables for combinations
test_exp = [None]
test_atom_squares = [setup_atomic_square()]
test_potentials = [
    ('rw', .9),
    # ('chi_sq', 1)
]
# test_qbin = [.1]
test_spring_kwargs = [{'k': 100, 'rt': 5., 'sp_type': 'rep'},
                      {'k': 100, 'rt': 1., 'sp_type': 'com'},
                      {'k': 100, 'rt': 1., 'sp_type': 'att'}]

test_calcs = [Spring(**t_kwargs) for t_kwargs in test_spring_kwargs]
test_calcs.extend(['FQ', 'PDF'])

ns = [10]
travis = False
if os.getenv('TRAVIS'):
    travis = True
    num_exp = 1
    proc_alg_pairs = list(product(['CPU'], ['nxn', 'flat-serial', 'flat']))
    comparison_pro_alg_pairs = list(combinations(proc_alg_pairs, 2))

    if bool(os.getenv('NUMBA_DISABLE_JIT')):
        pass
    else:
        # Use a slightly bigger test set, since we are using the JIT
        ns = [10, 100, 400]
        num_exp = 3

else:
    ns = [
        10,
        # 15,
        100,
        # 400,
        1000
    ]
    num_exp = 1
    proc_alg_pairs = {
        ('CPU', 'nxn'),
        ('CPU', 'flat-serial'),
        ('CPU', 'flat'),
        ('Multi-GPU', 'flat'),
    }
    comparison_pro_alg_pairs = [
        (('CPU', 'nxn'), ('CPU', 'flat-serial')),
        (('CPU', 'flat-serial'), ('CPU', 'flat')),
        (('CPU', 'nxn'), ('CPU', 'flat')),
        (('CPU', 'flat'), ('Multi-GPU', 'flat')),

    ]
    # comparison_pro_alg_pairs = list(combinations(proc_alg_pairs, 2))

test_exp.extend([generate_experiment() for i in range(num_exp)])
test_atoms = [setup_atoms(int(n)) for n in ns]
test_double_atoms = [setup_double_atoms(int(n)) for n in ns]

test_atoms_adp = []
for atoms in test_atoms:
    adps = ADP(atoms, np.random.normal(0, .1, atoms.positions.shape))
    new_atoms = dc(atoms)
    new_atoms.adps = adps
    test_atoms_adp.append(new_atoms)
test_atoms.extend(test_atoms_adp)
# test_atoms = test_atoms_adp
