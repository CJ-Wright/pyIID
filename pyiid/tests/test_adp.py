from pyiid.tests import *
from pyiid.adp import ADP
import pytest
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


@pytest.mark.parametrize("a", test_data)
def test_meta(a):
    check_method(a)
