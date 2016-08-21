#!/usr/bin/env python
from __future__ import print_function
import nose
from pyiid.testing.noseclasses import KnownFailure
from numba import cuda
from nose_exclude import NoseExclude
import os
import random

import sys
import pytest

__author__ = 'christopher'

# plugins = [KnownFailure, NoseExclude]
# env = {
#     "NOSE_WITH_COVERAGE": 1,
#     'NOSE_COVER_PACKAGE': 'pyiid',
#     'NOSE_COVER_HTML': 1,
#     'NOSE_VERBOSE': 2,
#     'NOSE_PROCESS_TIMEOUT': 599,
# }
#
# from nose.plugins import multiprocess
#
# multiprocess._instantiate_plugins = plugins
#
#
# def run():
#     nose.main(addplugins=[x() for x in plugins],
#               # argv=['-x'],
#               env=env)


# if __name__ == '__main__':
#     try:
#         os.environ["SHORT_TEST"] = "1"
#         os.environ["PYIID_TEST_SEED"] = str(int(random.random() * 2 ** 32))
#         print('seed:', os.environ["PYIID_TEST_SEED"])
#         run()
#     finally:
#         os.environ["SHORT_TEST"] = "0"
#         os.environ["PYIID_TEST_SEED"] = str(0)
#         exit()


if __name__ == '__main__':
    # show output results from every test function
    args = ['-v']
    # show the message output for skipped and expected failure tests
    args.append('-rxs')
    args.extend(sys.argv[1:])
    try:
        os.environ["SHORT_TEST"] = "1"
        os.environ["PYIID_TEST_SEED"] = str(int(random.random() * 2 ** 32))
        print('seed:', os.environ["PYIID_TEST_SEED"])
        results = pytest.main(args)
    finally:
        os.environ["SHORT_TEST"] = "0"
        os.environ["PYIID_TEST_SEED"] = str(0)
        # call pytest and exit with the return code from pytest so that
        # travis will fail correctly if tests fail
        sys.exit(results)
