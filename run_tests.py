#!/usr/bin/env python
import sys

import os
import pytest
import random

__author__ = 'christopher'

if __name__ == '__main__':
    # show output results from every test function
    # show the message output for skipped and expected failure tests
    args = ['-v', '-rxs']
    args.extend(sys.argv[1:])
    try:
        os.environ["PYIID_TEST_SEED"] = str(int(random.random() * 2 ** 32))
        print('seed:', os.environ["PYIID_TEST_SEED"])
        results = pytest.main(args)
    finally:
        os.environ["PYIID_TEST_SEED"] = str(0)
        # call pytest and exit with the return code from pytest so that
        # travis will fail correctly if tests fail
        sys.exit(results)
