"""
Test the consistency of the system. Also test our PDF against SrFit
"""
from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
import pytest

__author__ = 'christopher'

test_data = test_atoms * 10


def test_consistency():
    outs = [[] for i in range(len(test_atoms))]
    s = ElasticScatter()
    for i, atoms in enumerate(test_data):
        fq = s.get_fq(atoms)
        outs[i % len(test_atoms)].append(fq)
    for j in range(len(test_atoms)):
        for a, b in permutations(outs[j], 2):
            stats_check(a, b, rtol=2e-7, atol=1e-7)


@pytest.mark.xfail(not srfit, reason='Need installed srfit for this stest')
# @known_fail_if(not srfit)
def test_consistency2():
    outs = [[] for i in range(len(test_atoms))]
    s = ElasticScatter(verbose=True)
    for i, atoms in enumerate(test_data):
        stru = convert_atoms_to_stru(atoms)
        srfit_calc = DebyePDFCalculator()
        srfit_calc.qmin = s.exp['qmin']
        srfit_calc.qmax = s.exp['qmax']
        srfit_calc.qstep = s.exp['qbin']
        srfit_calc(stru)
        stats_check(s.get_scatter_vector(), srfit_calc.qgrid)
        fq = srfit_calc.fq
        outs[i % len(test_atoms)].append(fq)
    for j in range(len(test_atoms)):
        for a, b in permutations(outs[j], 2):
            stats_check(a, b)


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        '-x',
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
