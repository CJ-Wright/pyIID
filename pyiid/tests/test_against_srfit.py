from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter

__author__ = 'christopher'

local_test_atoms = setup_atomic_square()[0] * 3
test_data = tuple(product([local_test_atoms], [None]))


# TODO move this back to a test generator, leave the decorator on the check
@known_fail_if(not srfit)
def test_fq_against_srfit():
    # unpack the atoms and experiment
    atoms = local_test_atoms
    exp = None

    # get the pyIID F(Q)
    s = ElasticScatter(exp)
    # s.set_processor('CPU', 'nxn')

    # get the SrFit F(Q)
    stru = convert_atoms_to_stru(atoms)
    srfit_calc = DebyePDFCalculator()
    srfit_calc.qmin = s.exp['qmin']
    srfit_calc.qmax = s.exp['qmax']
    srfit_calc.qstep = s.exp['qbin']
    srfit_calc(stru)
    stats_check(s.get_scatter_vector(), srfit_calc.qgrid)
    ans1 = s.get_fq(atoms)
    ans2 = srfit_calc.fq
    stats_check(ans1, ans2, rtol=1e-4, atol=5e-6)
    del srfit_calc
    stats_check(ans1, ans2, rtol=1e-4, atol=5e-6)


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        # '-v'
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
