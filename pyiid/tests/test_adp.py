from pyiid.tests import *
from pyiid.adp import ADP
__author__ = 'christopher'

rtol = 5e-4
atol = 5e-5


# Actual Tests
def check_method(value):
    atoms = value[0]
    a = atoms.info['adps'].get_positions().copy()
    atoms.info['adps'].set_positions(a + np.ones(a.shape))
    b = atoms.info['adps'].get_positions()
    assert_allclose(a, b - np.ones(a.shape))


test_data = list(product(test_adp_atoms))

def test_meta():
    for v in test_data:
        yield check_method, v


if __name__ == '__main__':
    import nose

    print('number of test cases', len(test_data))
    nose.runmodule(argv=[
        '-s',
        '--with-doctest',
        # '--nocapture',
        '-v',
        '-x',
    ],
        # env={"NOSE_PROCESSES": 1, "NOSE_PROCESS_TIMEOUT": 599},
        exit=False)
