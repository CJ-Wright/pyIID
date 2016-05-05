__author__ = 'christopher'
import nose
from pyiid.testing.noseclasses import KnownFailure
from numba import cuda
from nose_exclude import NoseExclude
import os
import random
plugins = [KnownFailure, NoseExclude]
env = {
    "NOSE_WITH_COVERAGE": 1,
    'NOSE_COVER_PACKAGE': 'pyiid',
    'NOSE_COVER_HTML': 1,
    'NOSE_VERBOSE': 2,
    'NOSE_PROCESS_TIMEOUT': 599,
    'NOSE_EXCLUDE_DIRS': 'old_files/'
}

from nose.plugins import multiprocess
multiprocess._instantiate_plugins = plugins


def run():
    nose.main(addplugins=[x() for x in plugins],
              # argv=['-x'],
              env=env)


if __name__ == '__main__':
    try:
        os.environ["PYIID_TEST_SEED"] = str(int(random.random() * 2 ** 32))
        print('seed:', os.environ["PYIID_TEST_SEED"])
        run()
    finally:
        os.environ["PYIID_TEST_SEED"] = str(0)
