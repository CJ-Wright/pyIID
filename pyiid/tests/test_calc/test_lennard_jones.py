from pyiid.tests import *
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.calc.lennard_jones import LJ

__author__ = 'christopher'


def check_meta(value):
    value[0](value[1:])


def check_nrg(value):
    """
    Check two processor, algorithm pairs against each other for PDF energy

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    rtol = 4e-6
    atol = 9e-6
    # setup
    atoms = value[0]
    calc = LJ(param_dict=value[1])
    atoms.set_calculator(calc)
    ans = atoms.get_potential_energy()
    assert ans is not None
    assert np.any(ans)


def check_voxel_nrg(value):
    """
    Check two processor, algorithm pairs against each other for PDF energy

    Parameters
    ----------
    value: list or tuple
        The values to use in the tests
    """
    rtol = 4e-6
    atol = 9e-6
    # setup
    atoms = value[0]
    calc = LJ(param_dict=value[1])
    ans = calc.calculate_voxel_energy(atoms, Atom('Au', [0, 0, 0]), .1)
    assert ans is not None
    assert np.any(ans)


tests = [
    check_nrg,
    check_voxel_nrg
]
test_experiment_types = ['FQ', 'PDF']
test_data = tuple(product(tests,
                          test_atoms, [{('Au', 'Au'): [2.8, 10]}]))


def test_meta():
    for v in test_data:
        yield check_meta, v


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[
        # '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        '-x'
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
