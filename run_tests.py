__author__ = 'christopher'
import nose
from pyiid.testing.noseclasses import KnownFailure

plugins = [KnownFailure]
env = {"NOSE_WITH_COVERAGE": 1,
       'NOSE_COVER_PACKAGE': 'pyiid',
       'NOSE_COVER_HTML': 1}

from nose.plugins import multiprocess
multiprocess._instantiate_plugins = plugins


def run():

    nose.main(addplugins=[x() for x in plugins], env=env)


if __name__ == '__main__':
    print 'coveralls test'
    run()